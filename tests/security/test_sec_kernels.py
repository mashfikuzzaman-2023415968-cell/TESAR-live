"""Worker-side task safety (spec S.5.5, S.6.5, S.7.1, S.12 row 2).

The coordinator may be malicious from a phone's point of view (adversary A4).
These tests call the registry exactly as a worker will, with hostile input,
and check that only whitelisted kernels run, within the worker's own limits.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from mdc.common.validation import TaskLimits, ValidationError
from mdc.tasks import base
from mdc.tasks.errors import TaskTimeLimitError

pytestmark = pytest.mark.security

TILE = np.zeros((40, 40), dtype=np.uint8)


@pytest.mark.parametrize(
    "kind",
    ["os.system", "eval", "__import__", "pickle", "img_tile ", "IMG_TILE", "", "final"],
)
def test_only_whitelisted_kinds_execute(kind: str) -> None:
    with pytest.raises(ValidationError, match="unknown task kind"):
        base.execute(kind, {}, TILE)


@pytest.mark.parametrize(
    ("kind", "params", "data"),
    [
        ("img_tile", {"r": 6, "iters": 1}, TILE),  # radius above limit
        ("img_tile", {"r": 1, "iters": 65}, TILE),  # iterations above limit
        ("img_tile", {"r": 1, "iters": 1, "code": "x"}, TILE),  # extra key
        ("img_tile", {"r": "1", "iters": 1}, TILE),
        ("img_tile", "r=1", TILE),
        ("img_tile", {"r": 2, "iters": 10}, TILE),  # input smaller than 2 * halo
        ("img_tile", {"r": 1, "iters": 1}, None),  # missing input
        ("img_tile", {"r": 1, "iters": 1}, TILE.astype(np.int32)),
        ("img_tile", {"r": 1, "iters": 1}, np.zeros((10, 10, 3), dtype=np.uint8)),
        ("img_tile", {"r": 1, "iters": 1}, [[0] * 10] * 10),
        ("row_stats", {"edge_threshold": 300}, TILE),
        ("synthetic", {"seed": 1, "chain_len": 5_000_001}, None),
        ("synthetic", {"seed": 1, "chain_len": 10}, TILE),  # unexpected input
    ],
)
def test_hostile_tasks_rejected(kind: str, params: object, data: object) -> None:
    with pytest.raises(ValidationError):
        base.execute(kind, params, data)  # type: ignore[arg-type]


def test_worker_limits_override_defaults() -> None:
    # A phone owner can run with stricter limits than the coordinator asks for.
    strict = TaskLimits(blur_radius=1, blur_iterations=2, chain_len=100, max_tile_pixels=1000)
    with pytest.raises(ValidationError):
        base.execute("img_tile", {"r": 2, "iters": 1}, TILE, strict)
    with pytest.raises(ValidationError, match="max_tile_pixels"):
        base.execute("img_tile", {"r": 1, "iters": 1}, TILE, strict)  # 1600 px > 1000
    with pytest.raises(ValidationError):
        base.execute("synthetic", {"seed": 1, "chain_len": 101}, None, strict)
    base.execute("img_tile", {"r": 1, "iters": 2}, np.zeros((30, 30), dtype=np.uint8), strict)


def test_kernels_stop_at_deadline() -> None:
    past = time.monotonic() - 1
    with pytest.raises(TaskTimeLimitError):
        base.execute("img_tile", {"r": 1, "iters": 3}, TILE, deadline=past)
    with pytest.raises(TaskTimeLimitError):
        base.execute("synthetic", {"seed": 1, "chain_len": 20_000}, deadline=past)


def test_outputs_are_validated_results() -> None:
    out = base.execute("row_stats", {"edge_threshold": 10}, TILE)
    assert out.json_result is not None and sum(out.json_result["hist"]) == TILE.size
    syn = base.execute("synthetic", {"seed": 0, "chain_len": 1})
    assert syn.json_result is not None and len(syn.json_result["digest"]) == 64
