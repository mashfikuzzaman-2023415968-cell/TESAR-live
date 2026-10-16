"""Malicious workers with real processes (spec Phase 10 tests, S.12 row 10).

Workers with ``--malicious-prob`` return plausible, correctly hashed but wrong
results. Verification must catch them, quarantine them and repair the job; with
verification switched off the output must be wrong (the feature matters).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from mdc.common.credentials import read_join_file
from mdc.common.events import read_events
from mdc.coordinator.coordinator import Coordinator
from mdc.coordinator.registry import WorkerState
from tests.helpers.procs import REPO

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(180)]

JOB = [
    f"job.image_path={REPO / 'data/images/test_512.png'}",
    "job.blur_radius=1",
    "job.blur_iterations=2",
    "job.base_tile=64",
    "scheduler.policy=tesar",
    # No replicas: an honest replica could win every race, so a liar's results would
    # never be stored and the test would depend on timing.
    "scheduler.replication=false",
    "liveness.task_timeout_min_s=5",
]
EVIL = ["--lab-mode", "--malicious-prob", "1.0"]

Start = Callable[..., Awaitable[Coordinator]]


async def start(coord: Coordinator, enrolled, worker_proc, flags: list[list[str]], prefix: str):  # type: ignore[no-untyped-def]
    joins, procs = [], []
    for i, extra in enumerate(flags):
        join = enrolled(f"{prefix}-{i}")
        joins.append(join)
        procs.append(
            worker_proc(join, "--coordinator", f"127.0.0.1:{coord.port}", "--seed", str(i), *extra)
        )
    assert await coord.wait_for_workers(len(flags), 60) == len(flags), [p.log_text() for p in procs]
    return joins, procs


def state_of(coord: Coordinator, join: Path) -> WorkerState:
    record = coord.registry.get(read_join_file(join).worker_id)
    assert record is not None
    return record.state


async def test_malicious_worker_is_quarantined_and_output_correct(
    coordinator: Start, enrolled, worker_proc
) -> None:  # type: ignore[no-untyped-def]
    coord = await coordinator(*JOB, local_worker=True)
    joins, _ = await start(coord, enrolled, worker_proc, [EVIL, [], []], "mal")
    report = await coord.run_job(timeout_s=150)
    assert report["output_correct"] is True
    assert report["wrong_results_reached_final"] == 0
    assert state_of(coord, joins[0]) is WorkerState.QUARANTINED
    assert state_of(coord, joins[1]) is not WorkerState.QUARANTINED
    assert "mal-0" in report["quarantined"]
    assert report["verification_mismatches"] >= 2
    assert report["per_worker"]["mal-0"]["strikes"] >= 2
    assert [d["worker"] for d in report["detection_latency_s"]] == ["mal-0"]
    assert report["verifications"] >= 1 and report["verification_overhead_pct"] > 0
    events = read_events(coord.output_dir / "events.jsonl")
    assert any(e["type"] == "worker_quarantined" for e in events)


async def test_verify_everything_catches_an_occasional_liar(
    coordinator: Start, enrolled, worker_proc
) -> None:  # type: ignore[no-untyped-def]
    coord = await coordinator(*JOB, "verification.base_prob=1.0", local_worker=True)
    await start(
        coord, enrolled, worker_proc, [["--lab-mode", "--malicious-prob", "0.2"], [], []], "occ"
    )
    report = await coord.run_job(timeout_s=150)
    assert report["output_correct"] is True
    assert report["wrong_results_reached_final"] == 0


async def test_without_verification_the_output_is_wrong(
    coordinator: Start, enrolled, worker_proc
) -> None:  # type: ignore[no-untyped-def]
    """Negative control: the same attack with verification off corrupts the output."""
    coord = await coordinator(
        *JOB,
        "verification.enabled=false",
        "scheduler.allow_local=false",
        # The liar must certainly store results: under tesar an honest replica could
        # win every race and hide the attack.
        "scheduler.policy=round_robin",
        "scheduler.replication=false",
        local_worker=True,
    )
    await start(coord, enrolled, worker_proc, [EVIL, []], "off")
    report = await coord.run_job(timeout_s=150)
    assert report["output_correct"] is False
    assert report["wrong_results_reached_final"] >= 1
    assert report["verifications"] == 0 and report["quarantined"] == []


async def test_colluding_workers_with_identical_corruption_are_both_caught(
    coordinator: Start, enrolled, worker_proc
) -> None:  # type: ignore[no-untyped-def]
    """Two colluders verify each other and agree on the same wrong result: the collusion
    rule refuses that agreement, the trusted tiebreak catches both (S.12 row 10)."""
    coord = await coordinator(
        *JOB,
        "verification.prefer_local_verifier=false",  # colluder B verifies colluder A
        "scheduler.allow_local=false",
        "scheduler.policy=round_robin",
        local_worker=True,
    )
    collude = [*EVIL, "--collude-key", "gang"]
    joins, _ = await start(coord, enrolled, worker_proc, [collude, collude], "col")
    report = await coord.run_job(timeout_s=150)
    assert report["output_correct"] is True
    assert report["wrong_results_reached_final"] == 0
    assert state_of(coord, joins[0]) is WorkerState.QUARANTINED
    assert state_of(coord, joins[1]) is WorkerState.QUARANTINED
    assert report["tiebreaks"] >= 1  # their agreement was not accepted as a verification


async def test_quarantine_survives_reconnect(coordinator: Start, enrolled, worker_proc) -> None:  # type: ignore[no-untyped-def]
    coord = await coordinator(*JOB, local_worker=True)
    joins, procs = await start(coord, enrolled, worker_proc, [EVIL, []], "rec")
    await coord.run_job(timeout_s=150)
    bad_id = read_join_file(joins[0]).worker_id
    record = coord.registry.get(bad_id)
    assert record is not None and record.state is WorkerState.QUARANTINED
    old_session = record.session_id
    procs[0].stop()
    worker_proc(joins[0], "--coordinator", f"127.0.0.1:{coord.port}", *EVIL)  # comes back
    for _ in range(300):
        if record.session_id not in {None, old_session}:
            break
        await asyncio.sleep(0.1)
    assert record.session_id not in {None, old_session}, "the worker did not reconnect"
    assert record.state is WorkerState.QUARANTINED  # still quarantined on the new session
    second = await coord.run_job(timeout_s=150)
    assert second["output_correct"] is True
    events = read_events(coord.output_dir / "events.jsonl")
    starts = [i for i, e in enumerate(events) if e["type"] == "job_started"]
    assert len(starts) == 2
    after = events[starts[1] :]
    assert not [e for e in after if e["type"] == "attempt_sent" and e["worker_id"] == bad_id]
    assert any(e["type"] == "attempt_sent" for e in after)  # the job did run
