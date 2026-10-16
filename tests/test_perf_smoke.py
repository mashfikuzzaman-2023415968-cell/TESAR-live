"""Performance smoke tests (spec G.3 'Performance smoke', from Phase 9).

Budgets from the spec: one scheduling decision for 20 workers x 200 ready tasks
under 20 ms, and protocol encode/decode of a 10 MB array under 200 ms. Each is
the median of several runs so that one scheduler hiccup on a busy laptop does
not fail the gate.
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import pytest

from mdc.common import protocol as P
from mdc.common.config import load_config
from mdc.coordinator.energy import EnergyModel
from mdc.coordinator.job_builder import build_image_job_from_array
from mdc.coordinator.policies import POLICIES
from mdc.coordinator.policies.base import SchedulingContext, WorkerView

pytestmark = pytest.mark.perf

RUNS = 7
DECISION_BUDGET_S = 0.020
PROTOCOL_BUDGET_S = 0.200
TEN_MB = 10_000_000


def workers() -> list[WorkerView]:
    return [
        WorkerView(
            f"w{i:02d}",
            f"w{i}",
            i == 0,
            "phone" if i % 2 else "laptop",
            1,
            2,
            bench_ms_per_unit=1.0 + i,
            rep=0.5 + i / 50,
            battery={"pct": 80.0},
            rtt_s=0.01 * (1 + i % 3),
            bw_bytes_per_s=2e6 * (1 + i % 4),
        )
        for i in range(20)
    ]


@pytest.mark.parametrize("policy", ["tesar", "fastest", "most_reliable", "round_robin"])
def test_scheduler_decision_20_workers_200_tasks(policy: str) -> None:
    job = build_image_job_from_array(np.zeros((320, 640), dtype=np.uint8), 32, 1, 1, 64)
    ready = job.ready_tasks()
    assert len(ready) == 200
    cfg = load_config(overrides=[f"scheduler.policy={policy}"])
    times = []
    for _ in range(RUNS):
        ctx = SchedulingContext(ready, workers(), job, cfg, 0.0, energy=EnergyModel())
        decide = POLICIES[policy](cfg)
        t0 = time.perf_counter()
        out = decide.assign(ctx)
        times.append(time.perf_counter() - t0)
        assert out
    assert statistics.median(times) < DECISION_BUDGET_S, times


def test_protocol_encode_decode_10mb() -> None:
    """Framing, hashing and array (de)serialization of a 10 MB result. Compression is
    a separate, optional codec cost (zlib level 1: ~200 ms per 10 MB on the dev
    laptop; real per-chunk results are well under 1 MB) — DECISIONS.md, Phase 9."""
    arr = (np.arange(TEN_MB, dtype=np.uint64) % 251).astype(np.uint8).reshape(2500, 4000)
    limits = P.FrameLimits(max_payload_bytes=2 * TEN_MB, max_raw_array_bytes=2 * TEN_MB)
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fields, payload = P.pack_array(arr, P.COMPRESSION_NONE)
        msg = P.make_message(P.TASK_RESULT, payload, task_id="t", attempt_id="a", **fields)
        frame = P.encode_frame(msg, limits)
        back = P.decode_frame(frame, limits)
        out = P.unpack_array(back.header, back.payload, limits.max_raw_array_bytes)
        times.append(time.perf_counter() - t0)
    assert np.array_equal(out, arr)
    assert statistics.median(times) < PROTOCOL_BUDGET_S, times
