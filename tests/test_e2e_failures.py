"""End-to-end failure recovery with real worker processes (spec Phase 5, G.3 row 5).

Each job must finish with output byte-identical to the reference, and no task
may be completed twice, despite crashes, drops, frozen and flaky workers.
Workers are slowed with emulated network delay so that failures land mid-job.
"""

from __future__ import annotations

import asyncio
import os
import signal
from collections import Counter
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from mdc.common.events import read_events
from mdc.coordinator.coordinator import Coordinator
from mdc.coordinator.registry import WorkerState
from tests.helpers.procs import REPO

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(180)]

FAST_LIVENESS = [
    "liveness.ping_interval_s=0.25",
    "liveness.suspect_after_s=1.0",
    "liveness.dead_after_s=2.0",
    "liveness.task_timeout_min_s=1.0",
    "liveness.task_timeout_factor=3.0",
    "liveness.task_timeout_slack_s=0.5",
]
JOB = [
    f"job.image_path={REPO / 'data/images/test_512.png'}",
    "job.blur_radius=1",
    "job.blur_iterations=2",
    "job.base_tile=128",
    "scheduler.policy=round_robin",
    "scheduler.allow_local=false",  # remote workers do the work; local is only the fallback
]
SLOW = ["--lab-mode", "--net-delay-ms", "400"]  # every result takes >= 0.4 s

Start = Callable[..., Awaitable[Coordinator]]


def events_of(coord: Coordinator) -> list[dict]:
    return read_events(coord.output_dir / "events.jsonl")


def assert_no_duplicate_done(coord: Coordinator) -> None:
    done = Counter(e["task_id"] for e in events_of(coord) if e["type"] == "task_done")
    assert done and max(done.values()) == 1, done


def state_changes(coord: Coordinator, worker_id: str) -> list[tuple[float, str]]:
    return [
        (e["t"], e["new"])
        for e in events_of(coord)
        if e["type"] == "worker_state" and e["worker_id"] == worker_id
    ]


async def start_workers(
    coord: Coordinator,
    enrolled: Callable[..., Path],
    worker_proc: Callable[..., object],
    flags: list[list[str]],
    prefix: str,
) -> list:  # type: ignore[type-arg]
    procs = []
    for i, extra in enumerate(flags):
        join = enrolled(f"{prefix}-{i}")
        procs.append(worker_proc(join, "--coordinator", f"127.0.0.1:{coord.port}", *extra))
    assert await coord.wait_for_workers(len(flags), 60) == len(flags), [p.log_text() for p in procs]
    return procs


async def wait_until(cond: Callable[[], bool], timeout: float) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not cond():
        assert asyncio.get_running_loop().time() < deadline, "condition not reached"
        await asyncio.sleep(0.05)


async def test_crashing_worker(coordinator: Start, enrolled, worker_proc) -> None:  # type: ignore[no-untyped-def]
    # 64 small tiles instead of 16: the job must outlast the crash (2.5 s) plus DEAD
    # detection (2 s) however fast this machine is (DECISIONS.md, Phase 7).
    coord = await coordinator(*FAST_LIVENESS, *JOB, "job.base_tile=64", local_worker=True)
    procs = await start_workers(
        coord, enrolled, worker_proc, [[*SLOW, "--crash-after-s", "2.5"], SLOW, SLOW], "crash"
    )
    report = await coord.run_job(timeout_s=120)
    assert report["output_correct"] is True
    assert procs[0].proc.wait(10) == 1  # the crash fault really fired
    crashed = coord.registry.get(next(r.worker_id for r in coord.registry if r.name == "crash-0"))
    assert crashed.state is WorkerState.DEAD
    assert report["attempts_lost"] + report["attempts_timeout"] >= 1  # it crashed mid-job
    assert_no_duplicate_done(coord)


async def test_dropping_worker_times_out(coordinator: Start, enrolled, worker_proc) -> None:  # type: ignore[no-untyped-def]
    coord = await coordinator(*FAST_LIVENESS, *JOB, local_worker=True)
    await start_workers(
        coord, enrolled, worker_proc, [[*SLOW, "--drop-prob", "1.0"], SLOW, SLOW], "drop"
    )
    report = await coord.run_job(timeout_s=120)
    assert report["output_correct"] is True
    assert report["attempts_timeout"] >= 1
    assert report["per_worker"]["drop-0"]["tasks_ok"] == 0
    assert any(e["type"] == "attempt_timeout" for e in events_of(coord))
    assert_no_duplicate_done(coord)


async def test_frozen_worker_sigstop_sigcont(coordinator: Start, enrolled, worker_proc) -> None:  # type: ignore[no-untyped-def]
    coord = await coordinator(*FAST_LIVENESS, *JOB, local_worker=True)
    procs = await start_workers(coord, enrolled, worker_proc, [SLOW, SLOW, SLOW], "frozen")
    victim = next(r for r in coord.registry if r.name == "frozen-0")
    job = asyncio.create_task(coord.run_job(timeout_s=120))
    await wait_until(lambda: bool(victim.inflight), 30)
    os.kill(procs[0].pid, signal.SIGSTOP)  # the phone freezes with work in hand
    try:
        await wait_until(lambda: victim.state is WorkerState.DEAD, 10)
    finally:
        os.kill(procs[0].pid, signal.SIGCONT)  # ...and wakes up again
    report = await job
    assert report["output_correct"] is True
    # The frozen worker's work came back either through DEAD detection ("lost") or its
    # per-attempt timeout, whichever fires first (that order depends on the predictions).
    assert report["attempts_lost"] + report["attempts_timeout"] >= 1
    states = [s for _, s in state_changes(coord, victim.worker_id)]
    assert "SUSPECT" in states and "DEAD" in states
    assert_no_duplicate_done(coord)
    # The resumed worker reconnects with a new session.
    await wait_until(lambda: victim.state is WorkerState.ACTIVE, 30)


async def test_flaky_worker(coordinator: Start, enrolled, worker_proc) -> None:  # type: ignore[no-untyped-def]
    coord = await coordinator(*FAST_LIVENESS, *JOB, local_worker=True)
    await start_workers(
        coord,
        enrolled,
        worker_proc,
        [[*SLOW, "--flaky-mean-s", "0.7", "--seed", "3"], SLOW, SLOW],
        "flaky",
    )
    report = await coord.run_job(timeout_s=120)
    assert report["output_correct"] is True
    flaky = next(r for r in coord.registry if r.name == "flaky-0")
    assert "DEAD" in [s for _, s in state_changes(coord, flaky.worker_id)]  # it dropped out
    assert_no_duplicate_done(coord)


async def test_all_remote_workers_fail_local_fallback(
    coordinator: Start,
    enrolled,
    worker_proc,  # type: ignore[no-untyped-def]
) -> None:
    coord = await coordinator(
        *FAST_LIVENESS, *JOB, "liveness.max_attempts_per_task=2", local_worker=True
    )
    await start_workers(
        coord, enrolled, worker_proc, [["--lab-mode", "--drop-prob", "1.0"]] * 2, "dead"
    )
    report = await coord.run_job(timeout_s=150)
    assert report["output_correct"] is True
    fallbacks = [e for e in events_of(coord) if e["type"] == "task_fallback_local"]
    assert len(fallbacks) == 20  # every task ended up on the trusted local worker
    assert report["per_worker"]["local"]["tasks_ok"] == 20
    assert_no_duplicate_done(coord)


async def test_hetero_faulty6_cluster(coordinator: Start, cluster) -> None:  # type: ignore[no-untyped-def]
    # Every worker of a local cluster connects from 127.0.0.1: raise the per-IP limit.
    coord = await coordinator(
        *FAST_LIVENESS, *JOB, "network.max_connections_per_ip=16", local_worker=True
    )
    launcher = cluster("configs/cluster_profiles/hetero_faulty6.yaml", coord.port)
    assert await coord.wait_for_workers(6, 60) == 6, launcher.log_text()
    report = await coord.run_job(timeout_s=150)
    assert report["output_correct"] is True
    assert report["n_workers_remote"] == 6
    assert_no_duplicate_done(coord)
