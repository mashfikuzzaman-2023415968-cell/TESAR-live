"""Network estimation and adaptive chunking against real workers (spec Phase 6)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from mdc.common import protocol as P
from mdc.common.credentials import read_join_file
from mdc.common.events import read_events
from mdc.coordinator.coordinator import Coordinator
from mdc.coordinator.registry import WorkerState
from mdc.worker.faults import FaultSettings
from mdc.worker.shaper import Shaper
from tests.helpers.procs import REPO
from tests.helpers.raw_worker import connect_raw
from tests.helpers.worker_harness import running_worker

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]

Start = Callable[..., Awaitable[Coordinator]]


def probe_events(coord: Coordinator) -> list[dict]:
    return [e for e in coord.events.recent() if e["type"] == "probe_done"]


async def wait_for(cond: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not cond():
        assert asyncio.get_running_loop().time() < deadline, "condition not reached"
        await asyncio.sleep(0.05)


async def test_probe_measures_bandwidth_then_activates(
    coordinator: Start, enrolled: Callable[..., Path]
) -> None:
    coord = await coordinator()
    code = read_join_file(enrolled("prober"))
    async with running_worker(code, coord.port):
        await wait_for(lambda: bool(probe_events(coord)))
        record = coord.registry.get(code.worker_id)
        assert record.state is WorkerState.ACTIVE
    event = probe_events(coord)[0]
    assert event["timed_out"] is False and event["sample_bytes_per_s"] > 0
    snap = coord.netest.snapshot(code.worker_id)
    assert snap["probe_samples"] == 1
    assert snap["rtt_s"] > 0  # seeded by the handshake round trip
    states = [
        e["new"]
        for e in coord.events.recent()
        if e["type"] == "worker_state" and e["worker_id"] == code.worker_id
    ]
    assert states[:2] == ["PROBING", "ACTIVE"]


async def test_unanswered_probe_times_out_to_active(
    coordinator: Start, enrolled: Callable[..., Path]
) -> None:
    coord = await coordinator()
    code = read_join_file(enrolled("silent-prober"))
    raw = await connect_raw(code, coord.port)  # never reads, so never acknowledges
    record = coord.registry.get(code.worker_id)
    assert record.state is WorkerState.PROBING
    await asyncio.sleep(1.0)
    assert record.state is WorkerState.PROBING  # not eligible for tasks meanwhile
    await wait_for(lambda: record.state is WorkerState.ACTIVE, 5)
    assert probe_events(coord)[-1]["timed_out"] is True
    assert coord.netest.bw(code.worker_id) == coord.cfg.netest.default_bw_bytes_per_s
    await raw.close()


async def test_stale_or_wrong_probe_ack_is_ignored(
    coordinator: Start, enrolled: Callable[..., Path]
) -> None:
    coord = await coordinator()
    code = read_join_file(enrolled("liar-prober"))
    raw = await connect_raw(code, coord.port)
    await raw.send(
        P.make_message(P.PROBE_ACK, probe_id="p_not_ours", t_c=0.0, received_bytes=262144)
    )
    await asyncio.sleep(0.3)
    assert coord.registry.get(code.worker_id).state is WorkerState.PROBING
    assert coord.netest.snapshot(code.worker_id)["probe_samples"] == 0
    await raw.close()


async def test_pings_feed_rtt(coordinator: Start, enrolled: Callable[..., Path]) -> None:
    coord = await coordinator("liveness.ping_interval_s=0.1")
    code = read_join_file(enrolled("pinger"))
    async with running_worker(code, coord.port):
        await wait_for(
            lambda: (
                coord.netest.links.get(code.worker_id) is not None
                and coord.netest.links[code.worker_id].rtt.count >= 5
            ),
            10,
        )
    assert 0 < coord.netest.rtt(code.worker_id) < 0.5


async def test_shaped_link_gets_a_lower_bandwidth_estimate(
    coordinator: Start, enrolled: Callable[..., Path]
) -> None:
    """In-process (coordinator and workers share one event loop, so timings are
    noisy): only the robust facts are asserted here. The chunk-size effect is
    checked with real processes in ``test_e2e_chunking.py``."""
    coord = await coordinator(
        "scheduler.policy=fastest",
        "scheduler.allow_local=false",
        f"job.image_path={REPO / 'data/images/test_1024.png'}",
        "job.blur_iterations=4",
        "job.base_tile=128",
    )
    fast_code = read_join_file(enrolled("fast-link"))
    slow_code = read_join_file(enrolled("slow-link"))
    async with (
        running_worker(fast_code, coord.port),
        running_worker(
            slow_code,
            coord.port,
            shaper=Shaper(delay_ms=20, bw_kbps=1500),
            faults=FaultSettings(),
            lab_mode=True,
        ),
    ):
        assert await coord.wait_for_workers(2, 20) == 2
        report = await coord.run_job(timeout_s=110)
    assert report["output_correct"] is True
    assert coord.netest.bw(slow_code.worker_id) < coord.netest.bw(fast_code.worker_id)
    assert (
        report["per_worker"]["slow-link"]["bw_kbps"] < report["per_worker"]["fast-link"]["bw_kbps"]
    )
    sent = [
        e for e in read_events(coord.output_dir / "events.jsonl") if e["type"] == "attempt_sent"
    ]
    assert sent and all(
        {"n_tiles", "predicted_s", "est_rtt_s", "est_bw_bytes_per_s"} <= set(e) for e in sent
    )
    assert all(e["predicted_s"] > 0 for e in sent)
