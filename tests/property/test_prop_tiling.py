"""Property-based tiling exactness (spec G.3 Phase 2: >= 200 hypothesis examples).

For random images (including sizes that are not multiples of the tile),
random r in {1,2,3}, iters in {1..4} and random chunkings, the assembled
tiled output must equal the whole-image reference byte for byte.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mdc.tasks.image_pipeline import halo, pipeline_valid, reference
from mdc.tasks.row_stats import row_stats
from mdc.tasks.tiling import TileGrid, chunk_runs

pytestmark = pytest.mark.property


@settings(max_examples=250, deadline=None)
@given(
    height=st.integers(1, 64),
    width=st.integers(1, 64),
    tile=st.integers(3, 33),
    r=st.integers(1, 3),
    iters=st.integers(1, 4),
    sizes=st.lists(st.integers(1, 5), min_size=1, max_size=4),
    seed=st.integers(0, 2**32 - 1),
)
def test_tiled_equals_reference(
    height: int, width: int, tile: int, r: int, iters: int, sizes: list[int], seed: int
) -> None:
    img = np.random.default_rng(seed).integers(0, 256, size=(height, width), dtype=np.uint8)
    grid = TileGrid(height, width, tile, halo(r, iters))
    padded = grid.pad(img)
    results = {}
    for row in range(grid.rows):
        for c0, c1 in chunk_runs(grid.cols, sizes):
            out = pipeline_valid(grid.chunk_input(padded, row, c0, c1), r, iters)
            results.update(grid.split_chunk_output(out, row, c0, c1))
    assert grid.assemble(results).tobytes() == reference(img, r, iters).tobytes()


@settings(max_examples=200, deadline=None)
@given(
    height=st.integers(1, 40),
    width=st.integers(1, 40),
    threshold=st.integers(0, 255),
    seed=st.integers(0, 2**32 - 1),
)
def test_row_stats_invariants(height: int, width: int, threshold: int, seed: int) -> None:
    strip = np.random.default_rng(seed).integers(0, 256, size=(height, width), dtype=np.uint8)
    stats = row_stats(strip, threshold)
    assert len(stats["hist"]) == 256
    assert sum(stats["hist"]) == stats["pixels"] == height * width
    assert stats["edge_pixels"] == sum(stats["hist"][threshold:])
    assert 0 <= stats["edge_pixels"] <= stats["pixels"]
