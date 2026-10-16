"""Coordinator with real (in-process) workers over TLS loopback (spec Phase 4)."""

from __future__ import annotations

import asyncio
import json
import os
import signal
from collections.abc import Awaitable, Callable
from pathlib import Path

import numpy as np
import pytest

from mdc.common import protocol as P
from mdc.common.credentials import JoinCode, read_join_file
from mdc.common.events import read_events
from mdc.coordinator.coordinator import Coordinator, JobFailed
from mdc.coordinator.job_builder import build_image_job_from_array, build_synthetic_job
from mdc.coordinator.registry import WorkerState
from mdc.tasks.image_pipeline import reference
from mdc.tasks.row_stats import row_stats
from mdc.worker.faults import FaultSettings
from tests.helpers.raw_worker import connect_raw
from tests.helpers.worker_harness import running_worker

pytestmark = [pytest.mark.integration, pytest.mark.timeout(90)]

StartCoordinator = Callable[..., Awaitable[Coordinator]]


def small_image(seed: int = 0, shape: tuple[int, int] = (96, 128)) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=shape, dtype=np.uint8)


def small_job(seed: int = 0):  # type: ignore[no-untyped-def]
    return build_image_job_from_array(small_image(seed), 32, 1, 2, 64)


async def wait_for(cond: Callable[[], bool], timeout: float = 10.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not cond():
        if loop.time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.05)


def join_for(enrolled: Callable[..., Path], name: str) -> JoinCode:
    return read_join_file(enrolled(name))


async def test_round_robin_job_with_two_workers(
    coordinator: StartCoordinator, enrolled: Callable[..., Path]
) -> None:
    coord = await coordinator()
    a, b = join_for(enrolled, "w-a"), join_for(enrolled, "w-b")
    async with running_worker(a, coord.port), running_worker(b, coord.port):
        assert await coord.wait_for_workers(2, 15) == 2
        job = small_job()
        report = await coord.run_job(job, timeout_s=60)
    assert report["output_correct"] is True and report["differing_pixels"] == 0
    assert report["attempts_ok"] == report["attempts_total"] == len(job.tasks) - 1
    per = report["per_worker"]
    assert per["w-a"]["tasks_ok"] > 0 and per["w-b"]["tasks_ok"] > 0
    assert report["n_workers_remote"] == 2
    out = coord.output_dir
    for name in (
        "output.png",
        "reference.png",
        "report.json",
        "events.jsonl",
        "snapshot.json",
        "config_effective.yaml",
    ):
        assert (out / name).exists(), name
    types = [e["type"] for e in read_events(out / "events.jsonl")]
    for expected in (
        "coordinator_started",
        "worker_connected",
        "worker_state",
        "job_started",
        "attempt_sent",
        "attempt_result",
        "task_done",
        "job_finished",
    ):
        assert expected in types, expected
    sent = next(e for e in read_events(out / "events.jsonl") if e["type"] == "attempt_sent")
    assert {
        "attempt_id",
        "task_ids",
        "n_tiles",
        "worker_id",
        "purpose",
        "predicted_s",
        "timeout_s",
        "bytes_in",
    } <= set(sent)
    saved = json.loads((out / "report.json").read_text())
    assert saved["output_correct"] is True


async def test_local_policy_uses_only_the_local_worker(
    coordinator: StartCoordinator, enrolled: Callable[..., Path]
) -> None:
    coord = await coordinator("scheduler.policy=local", local_worker=True)
    async with running_worker(join_for(enrolled, "idle"), coord.port):
        await coord.wait_for_workers(1, 20)
        assert coord.local_active()
        report = await coord.run_job(small_job(1), timeout_s=60)
    assert report["output_correct"] is True
    assert report["per_worker"]["local"]["trusted"] is True
    assert report["per_worker"]["idle"]["tasks_ok"] == 0
    assert report["per_worker"]["local"]["tasks_ok"] == report["attempts_ok"]


async def test_synthetic_job(coordinator: StartCoordinator, enrolled: Callable[..., Path]) -> None:
    coord = await coordinator()
    async with running_worker(join_for(enrolled, "syn"), coord.port):
        await coord.wait_for_workers(1, 15)
        report = await coord.run_job(build_synthetic_job(6, 500, seed=3), timeout_s=60)
    assert report["output_correct"] is True
    assert report["job_type"] == "synthetic"


async def test_worker_lost_mid_job_tasks_are_reassigned(
    coordinator: StartCoordinator, enrolled: Callable[..., Path]
) -> None:
    coord = await coordinator()
    steady = join_for(enrolled, "steady")
    doomed = join_for(enrolled, "doomed")
    async with (
        running_worker(steady, coord.port),
        running_worker(doomed, coord.port, faults=FaultSettings(drop_prob=1.0), lab_mode=True) as (
            doomed_worker,
            _t,
            _r,
        ),
    ):
        await coord.wait_for_workers(2, 15)
        job_task = asyncio.create_task(coord.run_job(small_job(2), timeout_s=60))
        # The dropper swallows its tasks; disconnecting it makes them "lost".
        await wait_for(lambda: bool(coord.registry.get(doomed.worker_id).inflight))
        doomed_worker.stop()
        report = await job_task
    assert report["output_correct"] is True
    assert report["attempts_lost"] >= 1
    record = coord.registry.get(doomed.worker_id)
    assert record is not None and record.state in {WorkerState.DEAD, WorkerState.LEFT}


async def test_reconnect_keeps_the_same_record(
    coordinator: StartCoordinator, enrolled: Callable[..., Path]
) -> None:
    coord = await coordinator()
    code = join_for(enrolled, "phone")
    async with running_worker(code, coord.port):
        await coord.wait_for_workers(1, 15)
        first = coord.registry.get(code.worker_id)
        assert first is not None
        first.completed = 7  # history that must survive the reconnect
        session1 = first.session_id
    await wait_for(lambda: coord.registry.get(code.worker_id).state is WorkerState.DEAD)
    async with running_worker(code, coord.port):
        await wait_for(lambda: coord.registry.get(code.worker_id).state is WorkerState.ACTIVE)
        again = coord.registry.get(code.worker_id)
        assert again is first and again.completed == 7 and again.session_id != session1
    assert len(coord.registry) == 1


async def test_bye_marks_worker_left(
    coordinator: StartCoordinator, enrolled: Callable[..., Path]
) -> None:
    coord = await coordinator()
    raw = await connect_raw(join_for(enrolled, "leaver"), coord.port)
    assert raw.welcomed
    await raw.send(P.make_message(P.BYE, reason="going home"))
    await wait_for(lambda: any(r.state is WorkerState.LEFT for r in coord.registry))
    await raw.close()


async def test_second_session_replaces_the_first(
    coordinator: StartCoordinator, enrolled: Callable[..., Path]
) -> None:
    coord = await coordinator()
    code = join_for(enrolled, "dup")
    first = await connect_raw(code, coord.port)
    second = await connect_raw(code, coord.port)
    assert first.welcomed and second.welcomed
    assert await first.closed(5)  # the old session is closed by the server
    await wait_for(lambda: coord.registry.get(code.worker_id).state is WorkerState.ACTIVE)
    await second.close()


async def test_local_worker_is_restarted(coordinator: StartCoordinator) -> None:
    coord = await coordinator(local_worker=True)
    await wait_for(coord.local_active, 20)
    assert coord.local_worker is not None and coord.local_worker.pid is not None
    os.kill(coord.local_worker.pid, signal.SIGKILL)
    await wait_for(lambda: coord.local_worker.restarts >= 1, 10)
    await wait_for(coord.local_active, 20)
    types = [e["type"] for e in coord.events.recent()]
    assert "local_worker_restarted" in types


async def test_job_timeout_and_double_start(coordinator: StartCoordinator) -> None:
    coord = await coordinator()  # no workers at all: the job cannot progress
    with pytest.raises(JobFailed, match="within"):
        await coord.run_job(small_job(), timeout_s=0.5)
    job_task = asyncio.create_task(coord.run_job(small_job(), timeout_s=1.0))
    await asyncio.sleep(0.1)
    with pytest.raises(JobFailed, match="already running"):
        await coord.run_job(small_job(), timeout_s=1.0)
    with pytest.raises(JobFailed):
        await job_task


# ------------------------------------------------------ result validation


async def _job_with_raw_worker(coord: Coordinator, code: JoinCode):  # type: ignore[no-untyped-def]
    raw = await connect_raw(code, coord.port)
    assert raw.welcomed
    job = build_image_job_from_array(small_image(5, (40, 40)), 40, 1, 1, 64)  # one tile
    task = asyncio.create_task(coord.run_job(job, timeout_s=20))
    assign = await raw.recv(P.TASK_ASSIGN)
    return raw, job, task, assign


def _result(assign: P.Message, payload: bytes = b"", **fields: object) -> P.Message:
    base = {
        "task_id": assign.header["task_id"],
        "attempt_id": assign.header["attempt_id"],
        "status": "ok",
        "error": None,
        "compute_ms": 5.0,
        "queue_ms": 0.1,
        "t_recv_w": 1.0,
        "t_send_w": 1.1,
    }
    base.update(fields)
    return P.make_message(P.TASK_RESULT, payload, **base)


@pytest.mark.parametrize("variant", ["error_status", "wrong_shape", "integrity", "json_for_image"])
async def test_bad_results_are_rejected_and_retried(
    coordinator: StartCoordinator, enrolled: Callable[..., Path], variant: str
) -> None:
    coord = await coordinator()
    code = join_for(enrolled, f"bad-{variant.replace('_', '-')}")
    raw, _job, job_task, assign = await _job_with_raw_worker(coord, code)
    if variant == "error_status":
        bad = _result(assign, status="error", error="boom")
    elif variant == "wrong_shape":
        fields, payload = P.pack_array(np.zeros((5, 5), dtype=np.uint8))
        bad = _result(assign, payload, **fields)
    elif variant == "integrity":
        fields, payload = P.pack_array(np.zeros((40, 40), dtype=np.uint8), "none")
        fields["sha256"] = "0" * 64
        bad = _result(assign, payload, **fields)
    else:
        bad = _result(assign, json_result={"digest": "a" * 64})
    await raw.send(bad)
    retry = await raw.recv(P.TASK_ASSIGN)  # the task comes back
    assert retry.header["attempt_id"] != assign.header["attempt_id"]
    data = P.unpack_array(retry.header, retry.payload)
    out = reference(small_image(5, (40, 40)), 1, 1)
    assert data.shape == (44, 44)
    fields, payload = P.pack_array(out)
    await raw.send(_result(retry, payload, **fields))
    row = await raw.recv(P.TASK_ASSIGN)  # stage 2
    await raw.send(_result(row, json_result=row_stats(out, 64)))
    report = await job_task
    assert report["output_correct"] is True
    assert report["attempts_error"] == 1
    await raw.close()


async def test_row_stats_pixel_count_is_checked(
    coordinator: StartCoordinator, enrolled: Callable[..., Path]
) -> None:
    coord = await coordinator()
    raw, _job, job_task, assign = await _job_with_raw_worker(coord, join_for(enrolled, "rs"))
    out = reference(small_image(5, (40, 40)), 1, 1)
    fields, payload = P.pack_array(out)
    await raw.send(_result(assign, payload, **fields))
    row = await raw.recv(P.TASK_ASSIGN)
    lie = {"hist": [10] + [0] * 255, "edge_pixels": 0, "pixels": 10}  # valid schema, wrong row
    await raw.send(_result(row, json_result=lie))
    again = await raw.recv(P.TASK_ASSIGN)
    assert again.header["kind"] == "row_stats"
    job_task.cancel()
    with pytest.raises((asyncio.CancelledError, JobFailed)):
        await job_task
    await raw.close()


async def test_compute_times_are_clamped(
    coordinator: StartCoordinator, enrolled: Callable[..., Path]
) -> None:
    coord = await coordinator()
    raw, job, job_task, assign = await _job_with_raw_worker(coord, join_for(enrolled, "liar"))
    out = reference(small_image(5, (40, 40)), 1, 1)
    fields, payload = P.pack_array(out)
    await raw.send(_result(assign, payload, compute_ms=9e8, queue_ms=9e8, **fields))
    await raw.recv(P.TASK_ASSIGN)
    attempt = job.attempts[assign.header["attempt_id"]]
    assert attempt.compute_ms is not None
    assert attempt.compute_ms <= (attempt.result_t - attempt.sent_t) * 1000 + 1e-6
    job_task.cancel()
    with pytest.raises((asyncio.CancelledError, JobFailed)):
        await job_task
    await raw.close()
