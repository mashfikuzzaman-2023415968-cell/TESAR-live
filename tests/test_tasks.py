"""Tests for the task registry, row statistics, cost model and benchmark."""

from __future__ import annotations

import numpy as np
import pytest

from mdc.common import validation as V
from mdc.tasks import base
from mdc.tasks.benchmark import BENCH_CORE, benchmark_input, run_benchmark
from mdc.tasks.image_pipeline import halo, pipeline_valid
from mdc.tasks.row_stats import row_stats
from mdc.tasks.synthetic import hash_chain


def test_registry_matches_validation_whitelist() -> None:
    assert set(base.TASK_KINDS) == V.TASK_KINDS
    for name, kind in base.TASK_KINDS.items():
        assert kind.name == name
        assert kind.needs_input == (name in V.KINDS_WITH_INPUT)


def test_execute_img_tile() -> None:
    data = np.random.default_rng(1).integers(0, 256, size=(30, 40), dtype=np.uint8)
    out = base.execute("img_tile", {"r": 2, "iters": 2}, data)
    assert out.json_result is None
    assert out.array is not None
    np.testing.assert_array_equal(out.array, pipeline_valid(data, 2, 2))
    assert out.array.shape == (30 - 2 * halo(2, 2), 40 - 2 * halo(2, 2))


def test_execute_row_stats_and_synthetic() -> None:
    strip = np.array([[0, 10, 200], [255, 64, 63]], dtype=np.uint8)
    out = base.execute("row_stats", {"edge_threshold": 64}, strip)
    assert out.array is None
    assert out.json_result == row_stats(strip, 64)
    assert out.json_result["edge_pixels"] == 3
    syn = base.execute("synthetic", {"seed": 3, "chain_len": 7})
    assert syn.json_result == {"digest": hash_chain(3, 7)}


def test_row_stats_values() -> None:
    strip = np.array([[0, 0, 5], [5, 5, 255]], dtype=np.uint8)
    stats = row_stats(strip, 5)
    assert stats["hist"][0] == 2 and stats["hist"][5] == 3 and stats["hist"][255] == 1
    assert stats["edge_pixels"] == 4
    assert stats["pixels"] == 6
    assert all(type(v) is int for v in stats["hist"])  # plain JSON ints
    assert row_stats(strip, 0)["edge_pixels"] == 6
    assert row_stats(strip, 255)["edge_pixels"] == 1


@pytest.mark.parametrize(
    ("strip", "threshold"),
    [
        (np.zeros((2, 2), dtype=np.int32), 1),
        (np.zeros((2,), dtype=np.uint8), 1),
        ([[1]], 1),
        (np.zeros((2, 2), dtype=np.uint8), 256),
        (np.zeros((2, 2), dtype=np.uint8), -1),
        (np.zeros((2, 2), dtype=np.uint8), True),
    ],
)
def test_row_stats_rejects_bad_input(strip: object, threshold: object) -> None:
    with pytest.raises(V.ValidationError):
        row_stats(strip, threshold)  # type: ignore[arg-type]


def test_cost_model() -> None:
    assert base.img_tile_work_units(1000, 1000, 2, 8) == pytest.approx(40.0)
    assert base.img_tile_work_units(128, 128, 1, 1) == pytest.approx(128 * 128 / 1e6 * 3)
    assert base.row_stats_work_units(10_000_000) == pytest.approx(1.0)
    assert base.synthetic_work_units(200_000) == pytest.approx(200.0)
    assert base.compute_seconds(40.0, 25.0) == pytest.approx(1.0)


def test_benchmark_is_positive_and_deterministic_input() -> None:
    a, b = benchmark_input(), benchmark_input()
    np.testing.assert_array_equal(a, b)
    assert a.shape == (BENCH_CORE + 2 * halo(2, 4),) * 2
    score = run_benchmark(runs=1)
    assert 0.001 < score < 10_000
    with pytest.raises(ValueError):
        run_benchmark(runs=0)
