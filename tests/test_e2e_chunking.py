"""Adaptive chunking with real worker processes (spec Phase 6 acceptance).

"Chunk sizes visibly differ per worker": a fast worker and a slow one (8x slower
compute, 1.5 Mbit/s shaped link) run an image job under the ``fastest`` policy.
``chunking.target_task_s`` is lowered so that per-tile costs on this fast
development machine fall in the chunking rule's operating range (with the 2 s
default every worker would earn the maximum chunk here). The blur iteration
count is calibrated from this machine's benchmark so that one 128 x 128 tile
(1024 x 1024 image: 64 tiles) costs about 10 ms on the fast worker whatever the
CPU speed: with a fixed count the test passed on a throttled laptop and failed
at full clock speed (every chunk was the maximum because tiles were too cheap).
The assertions are unchanged.
"""

from __future__ import annotations

from collections import defaultdict

import pytest

from mdc.common.credentials import read_join_file
from mdc.common.events import read_events
from mdc.tasks.base import img_tile_work_units
from mdc.tasks.benchmark import run_benchmark
from tests.helpers.procs import REPO

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(180)]

TILE = 128
TARGET_TILE_S = 0.01


def calibrated_iterations() -> int:
    """Blur iterations that make one 128 x 128 tile take ~10 ms (max 64, the task limit)."""
    per_iteration_s = run_benchmark() / 1000.0 * img_tile_work_units(TILE, TILE, 1, 1)
    return max(4, min(64, round(TARGET_TILE_S / per_iteration_s)))


async def test_fast_worker_gets_bigger_chunks(coordinator, enrolled, worker_proc) -> None:  # type: ignore[no-untyped-def]
    coord = await coordinator(
        "scheduler.policy=fastest",
        "scheduler.allow_local=false",
        "chunking.target_task_s=0.1",
        "chunking.max_chunk_tiles=8",
        f"job.image_path={REPO / 'data/images/test_1024.png'}",
        f"job.blur_iterations={calibrated_iterations()}",
        f"job.base_tile={TILE}",
    )
    fast = enrolled("fast-proc")
    slow = enrolled("slow-proc")
    worker_proc(fast, "--coordinator", f"127.0.0.1:{coord.port}")
    worker_proc(
        slow,
        "--coordinator",
        f"127.0.0.1:{coord.port}",
        "--lab-mode",
        "--slowdown",
        "8",
        "--net-bw-kbps",
        "1500",
        "--net-delay-ms",
        "20",
    )
    assert await coord.wait_for_workers(2, 60) == 2
    report = await coord.run_job(timeout_s=150)
    assert report["output_correct"] is True
    fast_id, slow_id = read_join_file(fast).worker_id, read_join_file(slow).worker_id
    tiles: dict[str, list[int]] = defaultdict(list)
    for e in read_events(coord.output_dir / "events.jsonl"):
        if (
            e["type"] == "attempt_sent"
            and e["task_ids"][0].startswith("s1_")  # image tiles only: rows are never chunked
            and e["purpose"] in {"primary", "replica"}  # verifications reuse the producer's chunk
        ):
            tiles[e["worker_id"]].append(e["n_tiles"])
    assert tiles[fast_id], "the fast worker got no image work"
    fast_avg = sum(tiles[fast_id]) / len(tiles[fast_id])
    slow_avg = sum(tiles[slow_id]) / len(tiles[slow_id]) if tiles[slow_id] else 0.0
    assert fast_avg > slow_avg, dict(tiles)
    assert max(tiles[fast_id]) > max(tiles[slow_id] or [0])
    per = report["per_worker"]
    assert per["slow-proc"]["ms_per_unit"]["img_tile"] > per["fast-proc"]["ms_per_unit"]["img_tile"]
