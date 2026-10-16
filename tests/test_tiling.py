"""Tiling exactness (spec 8.2): tiled + assembled output == whole-image reference."""

from __future__ import annotations

import numpy as np
import pytest

from mdc.common.validation import ValidationError
from mdc.tasks.image_pipeline import halo, pipeline_valid, reference
from mdc.tasks.tiling import Rect, TileGrid, chunk_runs, describe


def tiled_output(img: np.ndarray, tile: int, r: int, iters: int, sizes: list[int]) -> np.ndarray:
    """Run the pipeline chunk by chunk exactly as workers would, then assemble."""
    grid = TileGrid(img.shape[0], img.shape[1], tile, halo(r, iters))
    padded = grid.pad(img)
    results = {}
    for row in range(grid.rows):
        for c0, c1 in chunk_runs(grid.cols, sizes):
            out = pipeline_valid(grid.chunk_input(padded, row, c0, c1), r, iters)
            results.update(grid.split_chunk_output(out, row, c0, c1))
    return grid.assemble(results)


def _random_case(seed: int) -> tuple[np.ndarray, int, int, int, list[int]]:
    rng = np.random.default_rng(seed)
    h, w = int(rng.integers(1, 90)), int(rng.integers(1, 90))
    tile = int(rng.integers(4, 40))
    r, iters = int(rng.integers(1, 4)), int(rng.integers(1, 5))
    sizes = [int(s) for s in rng.integers(1, 6, size=int(rng.integers(1, 4)))]
    img = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
    return img, tile, r, iters, sizes


@pytest.mark.parametrize("seed", range(40))
def test_tiling_exact_random_cases(seed: int) -> None:
    img, tile, r, iters, sizes = _random_case(seed)
    out = tiled_output(img, tile, r, iters, sizes)
    assert out.tobytes() == reference(img, r, iters).tobytes()


def test_tiling_exact_non_multiple_image_with_full_rows() -> None:
    img = np.random.default_rng(9).integers(0, 256, size=(130, 70), dtype=np.uint8)
    for sizes in ([1], [2], [100], [3, 1]):
        np.testing.assert_array_equal(tiled_output(img, 32, 2, 3, sizes), reference(img, 2, 3))


def test_grid_geometry() -> None:
    grid = TileGrid(height=300, width=260, tile=128, halo=17)
    assert (grid.rows, grid.cols) == (3, 3)
    assert grid.tile_rect(0, 0) == Rect(0, 128, 0, 128)
    assert grid.tile_rect(2, 2) == Rect(256, 300, 256, 260)  # smaller edge tile
    assert grid.tile_rect(2, 2).shape == (44, 4)
    assert grid.chunk_rect(1, 0, 2) == Rect(128, 256, 0, 260)
    assert grid.chunk_rect(1, 0, 2).pixels == 128 * 260
    assert describe(grid) == {
        "height": 300,
        "width": 260,
        "base_tile": 128,
        "halo": 17,
        "rows": 3,
        "cols": 3,
    }


def test_chunk_input_shape_includes_halo() -> None:
    grid = TileGrid(100, 100, 32, 5)
    padded = grid.pad(np.zeros((100, 100), dtype=np.uint8))
    assert padded.shape == (110, 110)
    chunk = grid.chunk_input(padded, 3, 1, 3)  # last row (4 px high), cols 32..100
    assert chunk.shape == (4 + 10, 68 + 10)
    assert chunk.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize(
    ("row", "c0", "c1"), [(-1, 0, 0), (4, 0, 0), (0, 2, 1), (0, 0, 4), (0, -1, 0)]
)
def test_invalid_chunk_rejected(row: int, c0: int, c1: int) -> None:
    grid = TileGrid(100, 100, 32, 1)
    with pytest.raises(ValueError):
        grid.chunk_rect(row, c0, c1)


@pytest.mark.parametrize(
    "kwargs",
    [{"height": 0}, {"width": -1}, {"tile": 0}, {"halo": -1}, {"tile": 2.5}, {"height": True}],
)
def test_invalid_grid_rejected(kwargs: dict) -> None:
    args = {"height": 10, "width": 10, "tile": 4, "halo": 1, **kwargs}
    with pytest.raises(ValueError):
        TileGrid(**args)


def test_pad_and_chunk_input_check_shapes() -> None:
    grid = TileGrid(10, 10, 4, 2)
    with pytest.raises(ValueError):
        grid.pad(np.zeros((10, 11), dtype=np.uint8))
    with pytest.raises(ValueError):
        grid.chunk_input(np.zeros((10, 10), dtype=np.uint8), 0, 0, 0)


def test_split_chunk_output_checks_shape() -> None:
    grid = TileGrid(10, 10, 4, 1)
    with pytest.raises(ValidationError):
        grid.split_chunk_output(np.zeros((4, 7), dtype=np.uint8), 0, 0, 1)
    tiles = grid.split_chunk_output(np.arange(32, dtype=np.uint8).reshape(4, 8), 0, 0, 1)
    assert sorted(tiles) == [(0, 0), (0, 1)]
    assert tiles[(0, 1)].tolist()[0] == [4, 5, 6, 7]


def test_assemble_and_row_strip_detect_missing_or_bad_tiles() -> None:
    grid = TileGrid(8, 8, 4, 1)
    tiles = {(r, c): np.full((4, 4), r * 2 + c, dtype=np.uint8) for r in range(2) for c in range(2)}
    out = grid.assemble(tiles)
    assert out[0, 0] == 0 and out[0, 7] == 1 and out[7, 0] == 2 and out[7, 7] == 3
    assert grid.row_strip(tiles, 1).shape == (4, 8)
    missing = dict(tiles)
    del missing[(1, 1)]
    with pytest.raises(KeyError):
        grid.assemble(missing)
    with pytest.raises(KeyError):
        grid.row_strip(missing, 1)
    bad = dict(tiles)
    bad[(0, 1)] = np.zeros((3, 4), dtype=np.uint8)
    with pytest.raises(ValidationError):
        grid.assemble(bad)
    with pytest.raises(ValidationError):
        grid.row_strip(bad, 0)
    with pytest.raises(ValueError):
        grid.row_strip(tiles, 2)


def test_chunk_runs() -> None:
    assert chunk_runs(7, [3]) == [(0, 2), (3, 5), (6, 6)]
    assert chunk_runs(5, [1, 2]) == [(0, 0), (1, 2), (3, 3), (4, 4)]
    assert chunk_runs(2, [10]) == [(0, 1)]
    for bad in ([], [0], [2, -1]):
        with pytest.raises(ValueError):
            chunk_runs(4, bad)
