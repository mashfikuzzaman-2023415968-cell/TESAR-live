"""Tile geometry: base grid, halo extraction, chunks, split and assembly (spec 8.2).

The H x W image is cut into a grid of base tiles of edge ``T`` (edge tiles
may be smaller)::

    rows = ceil(H / T), cols = ceil(W / T)

A **chunk** is a horizontal run of contiguous base tiles in one row
(``row``, columns ``c0..c1`` inclusive) — one rectangle, so one kernel call.

Why the halo makes tiling exact: the pipeline uses valid convolution and
shrinks its input by ``halo`` pixels per side. The coordinator pads the whole
image once with edge replication (``padded = np.pad(image, halo, "edge")``).
For a core region ``[y0:y1, x0:x1]`` of the original image, the input slice
``padded[y0 : y1 + 2*halo, x0 : x1 + 2*halo]`` is exactly the data the
whole-image reference used for those output pixels (the ``+halo`` shift of
padded coordinates cancels out), so the chunk's output equals the matching
part of the reference byte for byte.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from mdc.common.validation import ValidationError

U8Array = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class Rect:
    """Core rectangle ``[y0:y1, x0:x1]`` in original-image coordinates."""

    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def shape(self) -> tuple[int, int]:
        """``(height, width)``."""
        return (self.y1 - self.y0, self.x1 - self.x0)

    @property
    def pixels(self) -> int:
        """Number of pixels in the rectangle."""
        return (self.y1 - self.y0) * (self.x1 - self.x0)


@dataclass(frozen=True)
class TileGrid:
    """Base-tile grid of an ``height`` x ``width`` image with tile edge ``tile``."""

    height: int
    width: int
    tile: int
    halo: int

    def __post_init__(self) -> None:
        for name in ("height", "width", "tile"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive int")
        if type(self.halo) is not int or self.halo < 0:
            raise ValueError("halo must be a non-negative int")

    @property
    def rows(self) -> int:
        """Number of tile rows, ``ceil(H / T)``."""
        return math.ceil(self.height / self.tile)

    @property
    def cols(self) -> int:
        """Number of tile columns, ``ceil(W / T)``."""
        return math.ceil(self.width / self.tile)

    def _check_row(self, row: int) -> None:
        if not 0 <= row < self.rows:
            raise ValueError(f"tile row {row} outside 0..{self.rows - 1}")

    def chunk_rect(self, row: int, c0: int, c1: int) -> Rect:
        """Core rectangle of the chunk ``row``, columns ``c0..c1`` (inclusive)."""
        self._check_row(row)
        if not 0 <= c0 <= c1 < self.cols:
            raise ValueError(f"invalid chunk columns {c0}..{c1} (cols={self.cols})")
        t = self.tile
        return Rect(
            y0=row * t,
            y1=min((row + 1) * t, self.height),
            x0=c0 * t,
            x1=min((c1 + 1) * t, self.width),
        )

    def tile_rect(self, row: int, col: int) -> Rect:
        """Core rectangle of one base tile."""
        return self.chunk_rect(row, col, col)

    def pad(self, image: U8Array) -> U8Array:
        """Edge-replicate ``image`` by the halo (done once per job)."""
        if image.shape != (self.height, self.width):
            raise ValueError(f"image shape {image.shape} != grid {(self.height, self.width)}")
        padded: U8Array = np.pad(image, self.halo, mode="edge")
        return padded

    def chunk_input(self, padded: U8Array, row: int, c0: int, c1: int) -> U8Array:
        """Input slice (core + halo on all sides) for a chunk, from the padded image."""
        expected = (self.height + 2 * self.halo, self.width + 2 * self.halo)
        if padded.shape != expected:
            raise ValueError(f"padded image shape {padded.shape} != {expected}")
        r = self.chunk_rect(row, c0, c1)
        h2 = 2 * self.halo
        # Contiguous copy: it is serialized and sent, and must not keep the
        # whole padded image alive through a view.
        out: U8Array = np.ascontiguousarray(padded[r.y0 : r.y1 + h2, r.x0 : r.x1 + h2])
        return out

    def split_chunk_output(
        self, chunk_output: U8Array, row: int, c0: int, c1: int
    ) -> dict[tuple[int, int], U8Array]:
        """Slice a chunk's output back into per-base-tile arrays ``{(row, col): tile}``."""
        rect = self.chunk_rect(row, c0, c1)
        if chunk_output.shape != rect.shape:
            raise ValidationError(f"chunk output shape {chunk_output.shape} != {rect.shape}")
        tiles: dict[tuple[int, int], U8Array] = {}
        for col in range(c0, c1 + 1):
            t = self.tile_rect(row, col)
            tiles[(row, col)] = np.ascontiguousarray(
                chunk_output[:, t.x0 - rect.x0 : t.x1 - rect.x0]
            )
        return tiles

    def row_strip(self, results: dict[tuple[int, int], U8Array], row: int) -> U8Array:
        """Stitch the outputs of one tile row (stage-2 input)."""
        self._check_row(row)
        parts = []
        for col in range(self.cols):
            tile = results.get((row, col))
            if tile is None:
                raise KeyError(f"missing result for tile ({row}, {col})")
            if tile.shape != self.tile_rect(row, col).shape:
                raise ValidationError(f"tile ({row}, {col}) has shape {tile.shape}")
            parts.append(tile)
        strip: U8Array = np.concatenate(parts, axis=1)
        return strip

    def assemble(self, results: dict[tuple[int, int], U8Array]) -> U8Array:
        """Place every tile output into the full ``H x W`` output image."""
        out = np.zeros((self.height, self.width), dtype=np.uint8)
        for row in range(self.rows):
            for col in range(self.cols):
                tile = results.get((row, col))
                if tile is None:
                    raise KeyError(f"missing result for tile ({row}, {col})")
                r = self.tile_rect(row, col)
                if tile.shape != r.shape:
                    raise ValidationError(f"tile ({row}, {col}) has shape {tile.shape}")
                out[r.y0 : r.y1, r.x0 : r.x1] = tile
        return out


def chunk_runs(cols: int, sizes: list[int]) -> list[tuple[int, int]]:
    """Split ``cols`` columns into consecutive ``(c0, c1)`` runs of the given sizes.

    Sizes are used cyclically; the last run is truncated at the row end.
    (Used by tests and the in-process job runner.)
    """
    if not sizes or any(s < 1 for s in sizes):
        raise ValueError("chunk sizes must be positive")
    runs: list[tuple[int, int]] = []
    c0, i = 0, 0
    while c0 < cols:
        c1 = min(cols, c0 + sizes[i % len(sizes)]) - 1
        runs.append((c0, c1))
        c0, i = c1 + 1, i + 1
    return runs


def describe(grid: TileGrid) -> dict[str, Any]:
    """Geometry summary (for events, snapshots and the dashboard)."""
    return {
        "height": grid.height,
        "width": grid.width,
        "base_tile": grid.tile,
        "halo": grid.halo,
        "rows": grid.rows,
        "cols": grid.cols,
    }
