"""Adaptive chunk sizing (spec Phase 6 item 3, G.3 row 6)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mdc.common.config import load_config
from mdc.coordinator.chunking import choose_chunk_tiles, chunk_for
from mdc.coordinator.job import TaskStatus
from mdc.coordinator.job_builder import build_image_job_from_array, build_synthetic_job
from mdc.coordinator.policies.base import SchedulingContext, WorkerView

# 1 row x 12 tiles of 64x64, r=1, iters=2.
JOB_IMAGE = np.zeros((64, 12 * 64), dtype=np.uint8)


def ctx(*overrides: str, job=None) -> SchedulingContext:  # type: ignore[no-untyped-def]
    job = job or build_image_job_from_array(JOB_IMAGE, 64, 1, 2, 64)
    return SchedulingContext(
        job.ready_tasks(),
        [],
        job,
        load_config(overrides=["chunking.max_chunk_tiles=8", *overrides]),
        0.0,
    )


def worker(
    ms_per_unit: float = 10.0, bw: float = 1e7, rtt: float = 0.01, lam: float = 0.0
) -> WorkerView:
    return WorkerView(
        "w",
        "w",
        False,
        "phone",
        1,
        1,
        bench_ms_per_unit=ms_per_unit,
        rtt_s=rtt,
        bw_bytes_per_s=bw,
        failure_rate_per_s=lam,
    )


def per_tile_s(c: SchedulingContext, w: WorkerView) -> float:
    first = ("s1_r00_c00",)
    bin_, bout = c.wire_bytes(first)
    return c.compute_s(first, w) + (bin_ + bout) / w.bw_bytes_per_s


def test_time_target_formula() -> None:
    c = ctx("chunking.target_task_s=2.0")
    w = worker(ms_per_unit=15_000.0, rtt=0.1)  # ~0.37 s per tile
    expected = math.floor((2.0 - 0.1) / per_tile_s(c, w))
    assert 1 < expected < 8
    assert choose_chunk_tiles(w, c.job.tasks["s1_r00_c00"], c) == expected


def test_fast_stable_worker_gets_larger_chunk_than_slow_one() -> None:
    c = ctx()
    first = c.job.tasks["s1_r00_c00"]
    fast = choose_chunk_tiles(worker(ms_per_unit=10.0), first, c)
    slow = choose_chunk_tiles(worker(ms_per_unit=20_000.0), first, c)
    assert fast == 8 > slow  # fast hits the maximum
    assert slow >= 1


def test_low_bandwidth_means_smaller_chunks() -> None:
    c = ctx()
    first = c.job.tasks["s1_r00_c00"]
    assert choose_chunk_tiles(worker(bw=1e9), first, c) > choose_chunk_tiles(
        worker(bw=20_000), first, c
    )


def test_high_failure_rate_means_smaller_chunks() -> None:
    c = ctx("chunking.loss_tolerance=0.1")
    first = c.job.tasks["s1_r00_c00"]
    stable = choose_chunk_tiles(worker(ms_per_unit=4000.0, lam=0.0), first, c)  # ~0.1 s/tile
    flaky_w = worker(ms_per_unit=4000.0, lam=0.5)
    flaky = choose_chunk_tiles(flaky_w, first, c)
    expected_loss = math.floor(-math.log(0.9) / (0.5 * per_tile_s(c, flaky_w)))
    assert flaky == max(1, min(8, expected_loss)) < stable
    # The resulting chunk respects the loss tolerance (when not clamped to the minimum).
    if flaky > 1:
        assert 1 - math.exp(-0.5 * flaky * per_tile_s(c, flaky_w)) <= 0.1 + 1e-9


def test_min_and_max_respected() -> None:
    c = ctx("chunking.min_chunk_tiles=2", "chunking.max_chunk_tiles=3")
    first = c.job.tasks["s1_r00_c00"]
    assert choose_chunk_tiles(worker(ms_per_unit=1e-6), first, c) == 3
    assert choose_chunk_tiles(worker(ms_per_unit=1e9), first, c) == 2
    assert choose_chunk_tiles(worker(rtt=99.0), first, c) == 2  # rtt beyond the target


def test_row_boundary_and_runnable_tiles_cut_the_chunk() -> None:
    c = ctx()
    w = worker(ms_per_unit=1e-6)
    assert chunk_for(w, c.job.tasks["s1_r00_c09"], c, set()) == [
        "s1_r00_c09",
        "s1_r00_c10",
        "s1_r00_c11",
    ]  # row ends after 3 tiles
    c.job.tasks["s1_r00_c03"].status = TaskStatus.RUNNING
    assert chunk_for(w, c.job.tasks["s1_r00_c00"], c, set()) == [
        "s1_r00_c00",
        "s1_r00_c01",
        "s1_r00_c02",
    ]
    assert chunk_for(w, c.job.tasks["s1_r00_c00"], c, {"s1_r00_c01"}) == ["s1_r00_c00"]


def test_non_adaptive_returns_fixed() -> None:
    c = ctx("chunking.adaptive=false", "chunking.fixed_chunk_tiles=3")
    first = c.job.tasks["s1_r00_c00"]
    assert choose_chunk_tiles(worker(ms_per_unit=1e-6), first, c) == 3
    assert choose_chunk_tiles(worker(ms_per_unit=1e9, lam=5.0), first, c) == 3


def test_non_image_tasks_are_never_chunked() -> None:
    syn = build_synthetic_job(5, 100, 1)
    c = ctx(job=syn)
    assert choose_chunk_tiles(worker(), syn.tasks["syn_0000"], c) == 1


def test_zero_cost_tile_takes_the_maximum() -> None:
    c = ctx()
    w = worker(ms_per_unit=0.0, bw=1e300)
    assert choose_chunk_tiles(w, c.job.tasks["s1_r00_c00"], c) == 8


@pytest.mark.parametrize("n_tiles", [2, 5])
def test_chunk_bytes_share_the_halo(n_tiles: int) -> None:
    c = ctx()
    ids = tuple(f"s1_r00_c{i:02d}" for i in range(n_tiles))
    chunk_in, chunk_out = c.wire_bytes(ids)
    single_in, single_out = c.wire_bytes(ids[:1])
    assert chunk_out == n_tiles * single_out
    assert chunk_in < n_tiles * single_in  # one shared halo, not one per tile
