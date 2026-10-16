"""Stage-2 task: histogram and edge count of one row of output tiles (spec 8.3).

Input: the row strip (stage-1 outputs of one tile row, stitched together).
Output: ``{"hist": [256 ints], "edge_pixels": int, "pixels": int}`` where
``edge_pixels`` counts pixels ``>= edge_threshold``.

The task is cheap; it exists to create **partial dependencies** in the job
DAG: each ``s2_rXX`` depends only on the tiles of row XX. If a stage-1 tile is
later found to be wrong, only its row's statistics must be recomputed — this
is what taint propagation (Phase 10) repairs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from mdc.common.validation import ValidationError

HIST_BINS = 256
MAX_PIXEL = 255


def row_stats(strip: npt.NDArray[np.uint8], edge_threshold: int) -> dict[str, Any]:
    """Compute the row statistics of a uint8 strip."""
    if not isinstance(strip, np.ndarray) or strip.dtype != np.uint8 or strip.ndim != 2:  # noqa: PLR2004
        raise ValidationError("row strip must be a 2-D uint8 numpy array")
    if type(edge_threshold) is not int or not 0 <= edge_threshold <= MAX_PIXEL:
        raise ValidationError("edge_threshold must be an int in 0..255")
    hist = np.bincount(strip.ravel(), minlength=HIST_BINS)
    # Python ints (not numpy ints) so the result is plain JSON.
    return {
        "hist": [int(v) for v in hist],
        "edge_pixels": int(np.count_nonzero(strip >= edge_threshold)),
        "pixels": int(strip.size),
    }
