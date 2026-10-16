"""Standard benchmark that gives a device its speed score (spec 8.6).

Runs a fixed ``img_tile`` task (256 x 256 output core, r = 2, iters = 4) once
to warm up (first-call costs: page faults, CPU frequency ramp-up), then 5
timed times, and returns ``median_ms / work_units`` = **milliseconds per work
unit**. The median ignores one-off hiccups (another app waking up on a phone).

The worker reports this in HELLO as ``bench_ms_per_unit``; the coordinator
uses it as the initial speed estimate and replaces it with measured compute
times (EWMA per kind) as results arrive.
"""

from __future__ import annotations

import statistics
import time

import numpy as np

from mdc.tasks.base import execute, img_tile_work_units
from mdc.tasks.image_pipeline import halo

BENCH_CORE = 256
BENCH_R = 2
BENCH_ITERS = 4
BENCH_RUNS = 5
BENCH_SEED = 86


def benchmark_input() -> np.ndarray:
    """The fixed, deterministic benchmark input (core + halo on each side)."""
    side = BENCH_CORE + 2 * halo(BENCH_R, BENCH_ITERS)
    rng = np.random.default_rng(BENCH_SEED)
    return rng.integers(0, 256, size=(side, side), dtype=np.uint8)


def run_benchmark(runs: int = BENCH_RUNS) -> float:
    """Return this device's milliseconds per work unit."""
    if runs < 1:
        raise ValueError("runs must be >= 1")
    data = benchmark_input()
    params = {"r": BENCH_R, "iters": BENCH_ITERS}
    execute("img_tile", params, data)  # warm-up, not timed
    times_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        execute("img_tile", params, data)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    units = img_tile_work_units(BENCH_CORE, BENCH_CORE, BENCH_R, BENCH_ITERS)
    return statistics.median(times_ms) / units
