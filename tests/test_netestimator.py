"""Network estimation (spec Phase 6 item 1, G.3 row 6)."""

from __future__ import annotations

import math

import pytest

from mdc.common.config import load_config
from mdc.coordinator.netestimator import NetEstimator

CFG = load_config(
    overrides=[
        "netest.rtt_ewma_alpha=0.5",
        "netest.bw_ewma_alpha=0.5",
        "netest.default_bw_bytes_per_s=1000000",
        "netest.min_bw_bytes_per_s=10000",
    ]
).netest


def test_defaults_before_any_sample() -> None:
    net = NetEstimator(CFG)
    assert net.rtt("w") == 0.0 and net.rtt_min("w") == 0.0
    assert net.bw("w") == 1_000_000
    assert net.predict_transfer_s("w", 500_000, 500_000) == pytest.approx(1.0)
    assert net.snapshot("w")["passive_samples"] == 0


def test_rtt_ewma_and_minimum() -> None:
    net = NetEstimator(CFG)
    for sample in (0.10, 0.30, 0.02):
        assert net.on_rtt("w", sample)
    # EWMA alpha 0.5: 0.10 -> 0.20 -> 0.11
    assert net.rtt("w") == pytest.approx(0.11)
    assert net.rtt_min("w") == pytest.approx(0.02)


def test_probe_sample_subtracts_min_rtt() -> None:
    net = NetEstimator(CFG)
    net.on_rtt("w", 0.05)
    # 262144 bytes in 0.30 s elapsed, 0.05 s of it was latency -> 1 048 576 B/s
    assert net.on_probe_ack("w", 262_144, 0.30) == pytest.approx(262_144 / 0.25)
    assert net.bw("w") == pytest.approx(262_144 / 0.25)
    assert net.snapshot("w")["probe_samples"] == 1


def test_passive_sample_math() -> None:
    net = NetEstimator(CFG)
    net.on_rtt("w", 0.1)
    # total 1.5 s, compute 600 ms, queue 200 ms -> net 0.7 s, minus rtt 0.1 -> 0.6 s transfer
    sample = net.on_transfer("w", 1.5, 600.0, 200.0, 40_000, 20_000)
    assert sample == pytest.approx(60_000 / 0.6)
    net.on_transfer("w", 1.5, 600.0, 200.0, 40_000, 20_000)
    assert net.bw("w") == pytest.approx(60_000 / 0.6)
    assert net.predict_transfer_s("w", 30_000, 30_000) == pytest.approx(0.1 + 0.6)


@pytest.mark.parametrize(
    ("total", "compute", "queue", "bytes_in", "bytes_out"),
    [
        (1.0, 0.0, 0.0, 8_000, 8_000),  # under 16 KiB: too little data to mean anything
        (0.103, 100.0, 0.0, 100_000, 0),  # transfer time under 5 ms
        (0.5, 600.0, 0.0, 100_000, 0),  # compute longer than total -> clamped to 1 ms -> noisy
    ],
)
def test_noisy_passive_samples_ignored(
    total: float, compute: float, queue: float, bytes_in: int, bytes_out: int
) -> None:
    net = NetEstimator(CFG)
    assert net.on_transfer("w", total, compute, queue, bytes_in, bytes_out) is None
    assert net.bw("w") == 1_000_000
    assert net.snapshot("w")["ignored_samples"] == 1


@pytest.mark.parametrize("value", [math.nan, math.inf, -1.0, -math.inf, "fast", None, True])
def test_hostile_values_never_reach_the_estimates(value: object) -> None:
    net = NetEstimator(CFG)
    assert net.on_rtt("w", value) is False
    assert net.on_rtt("w", 10_000.0) is False  # absurdly large RTT
    assert net.on_probe_ack("w", 262_144, value) is None
    assert net.on_transfer("w", value, 1.0, 1.0, 100_000, 0) is None
    assert net.on_transfer("w", 2.0, value, 1.0, 100_000, 0) is None
    assert net.on_transfer("w", 2.0, 1.0, value, 100_000, 0) is None
    assert net.rtt("w") == 0.0 and net.bw("w") == 1_000_000


def test_bandwidth_is_clamped() -> None:
    net = NetEstimator(CFG)
    assert net.on_transfer("w", 1000.0, 0.0, 0.0, 20_000, 0) == 10_000  # floor
    net2 = NetEstimator(CFG)
    assert net2.on_probe_ack("w", 10**12, 0.001) == 1e9  # ceiling
    assert net2.on_probe_ack("w", 0, 1.0) is None


def test_compression_ratios() -> None:
    net = NetEstimator(CFG)
    assert net.compression_ratio("img_tile", "in") == 1.0
    net.on_compression("img_tile", "in", 1000, 400)
    net.on_compression("img_tile", "out", 1000, 900)
    net.on_compression("img_tile", "in", 1000, 600)
    assert net.compression_ratio("img_tile", "in") == pytest.approx(0.3 * 0.6 + 0.7 * 0.4)
    assert net.compression_table() == {"img_tile": (pytest.approx(0.46), 0.9)}
    net.on_compression("img_tile", "in", 0, 10)  # ignored
    net.on_compression("img_tile", "out", 10, 10**9)  # capped at 2
    assert net.compression_ratio("img_tile", "out") <= 2.0
