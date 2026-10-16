"""Chaos soak test (spec Phase 14, ``pytest -m soak``; run by the gate at Phase 14).

One coordinator runs many consecutive jobs while a cluster of honest, dropping,
flaky and malicious workers is randomly killed and restarted. Every job must
finish with byte-exact output, the liar must end up quarantined, and nothing
may accumulate between jobs (asyncio tasks, in-flight attempts, verification
checks, memory). ``MDC_SOAK_ROUNDS`` changes the length (default 40 jobs, about
one minute on the development laptop).
"""

from __future__ import annotations

import asyncio
import gc
import os
import random
import resource
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from mdc.common.credentials import read_join_file
from mdc.coordinator.coordinator import Coordinator
from mdc.coordinator.registry import WorkerState
from tests.helpers.procs import REPO

pytestmark = [pytest.mark.soak, pytest.mark.timeout(0)]

ROUNDS = int(os.environ.get("MDC_SOAK_ROUNDS", "40"))
MAX_RSS_GROWTH_MB = 150
WORKERS: dict[str, list[str]] = {
    "honest-a": [],
    "honest-b": [],
    "dropper": ["--lab-mode", "--drop-prob", "0.2"],
    "flaky": ["--lab-mode", "--flaky-mean-s", "4"],
    "liar": ["--lab-mode", "--malicious-prob", "0.5"],
}


async def test_chaos_soak(
    coordinator: Callable[..., Awaitable[Coordinator]],
    enrolled: Callable[..., Path],
    worker_proc: Callable[..., object],
) -> None:
    coord = await coordinator(
        f"job.image_path={REPO / 'data/images/test_512.png'}",
        "job.base_tile=64",
        "job.blur_iterations=2",
        "scheduler.policy=tesar",
        "verification.base_prob=1.0",  # every wrong result must be caught: output always correct
        "liveness.ping_interval_s=0.25",
        "liveness.suspect_after_s=1.0",
        "liveness.dead_after_s=2.0",
        "liveness.task_timeout_min_s=1.0",
        "network.max_connections_per_ip=16",
        local_worker=True,
    )
    joins = {name: enrolled(name) for name in WORKERS}

    def start(name: str) -> object:
        return worker_proc(joins[name], "--coordinator", f"127.0.0.1:{coord.port}", *WORKERS[name])

    procs = {name: start(name) for name in WORKERS}
    assert await coord.wait_for_workers(len(WORKERS) - 1, 60) >= len(WORKERS) - 1
    rng = random.Random(2026)  # noqa: S311 - test chaos, not security
    rss: list[int] = []
    for round_no in range(ROUNDS):
        if round_no and rng.random() < 0.6:  # chaos: kill a worker, sometimes bring it back
            victim = rng.choice(sorted(procs))
            procs[victim].kill()  # type: ignore[attr-defined]
            if rng.random() < 0.7:
                procs[victim] = start(victim)
        report = await coord.run_job(timeout_s=180)
        assert report["output_correct"] is True, (round_no, report)
        assert report["wrong_results_reached_final"] == 0
        # Nothing may be left over from the job.
        assert coord.dispatcher.verifier.outstanding() == 0
        assert all(not r.inflight for r in coord.registry)  # every attempt closed at job end
        gc.collect()
        rss.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        await asyncio.sleep(0.2)
    liar = coord.registry.get(read_join_file(joins["liar"]).worker_id)
    assert liar is not None and liar.state is WorkerState.QUARANTINED
    honest = coord.registry.get(read_join_file(joins["honest-a"]).worker_id)
    assert honest is not None and honest.state is not WorkerState.QUARANTINED
    tasks = [t for t in asyncio.all_tasks() if not t.done()]
    assert len(tasks) < 60, [t.get_coro() for t in tasks]  # bounded background tasks
    assert len(coord.events.recent()) <= 200
    growth_mb = (rss[-1] - rss[min(2, len(rss) - 1)]) / 1024  # ru_maxrss is KiB on Linux
    assert growth_mb < MAX_RSS_GROWTH_MB, rss
