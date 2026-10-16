"""Integer image pipeline: iterated box blur + Sobel edge magnitude (spec 8.1).

Everything is computed with integers (``int64`` sums, ``//`` floor division,
``uint8`` results), so results are bit-for-bit identical on every CPU — a
floating-point implementation could differ in the last bit between x86 and
ARM, and then a correct phone would look "malicious" to the verifier.

Only **valid** convolution is used: an output pixel is computed only where
the whole window lies inside the input, so the output shrinks by the window
radius on every side. That is what makes tiling exact: give each tile enough
surrounding input (a "halo") and its output is exactly the matching part of
the whole-image result, with no seams.

* ``box_blur_valid(img, r)``: mean over a (2r+1)x(2r+1) window, shrinks by r.
* ``sobel_valid(img)``: L1 gradient magnitude, shrinks by 1.
* ``pipeline_valid(img, r, iters)``: ``iters`` blurs then Sobel; total shrink
  per side is ``halo(r, iters) = r * iters + 1``.
* ``reference(img, r, iters)``: whole-image ground truth with edge-replicated
  padding, same shape as the input.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from mdc.common.validation import DEFAULT_TASK_LIMITS, TaskLimits, ValidationError
from mdc.tasks.errors import check_deadline

U8Array = npt.NDArray[np.uint8]
MAX_PIXEL = 255


def halo(r: int, iters: int) -> int:
    """Pixels the pipeline removes from each side: ``r * iters + 1``."""
    return r * iters + 1


def _check_image(img: npt.NDArray[Any], min_side: int) -> None:
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim != 2:  # noqa: PLR2004
        raise ValidationError("image must be a 2-D uint8 numpy array")
    if img.shape[0] < min_side or img.shape[1] < min_side:
        raise ValidationError(f"image {img.shape} too small: each side must be >= {min_side}")


def _check_params(r: int, iters: int, limits: TaskLimits) -> None:
    # type() checks reject bools and numpy floats; limits are enforced here so
    # that a kernel called directly (not through validation) is still safe.
    if type(r) is not int or not 1 <= r <= limits.blur_radius:
        raise ValidationError(f"blur radius must be an int in 1..{limits.blur_radius}")
    if type(iters) is not int or not 1 <= iters <= limits.blur_iterations:
        raise ValidationError(f"blur iterations must be an int in 1..{limits.blur_iterations}")


def box_blur_valid(img: U8Array, r: int, limits: TaskLimits = DEFAULT_TASK_LIMITS) -> U8Array:
    """Box blur with valid convolution, via a summed-area table.

    ``out[y, x] = floor(sum(img[y : y+k, x : x+k]) / k**2)`` with ``k = 2r + 1``,
    i.e. the window centred at input pixel ``(y + r, x + r)``. Output shape
    ``(H - 2r, W - 2r)``.

    Summed-area table: ``s[i, j]`` = sum of all pixels above and left of
    ``(i, j)``. Any rectangle sum is then 4 lookups, so the cost does not
    depend on ``r``: O(H*W) instead of O(H*W*k*k).
    """
    _check_params(r, 1, limits)
    k = 2 * r + 1
    _check_image(img, k)
    height, width = img.shape
    table = np.zeros((height + 1, width + 1), dtype=np.int64)
    table[1:, 1:] = img.astype(np.int64).cumsum(axis=0).cumsum(axis=1)
    window_sum = table[k:, k:] - table[:-k, k:] - table[k:, :-k] + table[:-k, :-k]
    out: U8Array = (window_sum // (k * k)).astype(np.uint8)
    return out


def sobel_valid(img: U8Array) -> U8Array:
    """Sobel edge magnitude ``min(255, |gx| + |gy|)``, valid region ``(H-2, W-2)``.

    ``gx = [[-1,0,1],[-2,0,2],[-1,0,1]]``, ``gy = [[-1,-2,-1],[0,0,0],[1,2,1]]``
    applied as correlation with numpy slicing (no scipy). The L1 magnitude
    keeps the arithmetic integer (an L2 magnitude would need a square root).
    """
    _check_image(img, 3)
    x = img.astype(np.int32)
    h, w = x.shape
    top, mid, bot = x[0 : h - 2], x[1 : h - 1], x[2:h]
    gx = (top[:, 2:w] + 2 * mid[:, 2:w] + bot[:, 2:w]) - (
        top[:, 0 : w - 2] + 2 * mid[:, 0 : w - 2] + bot[:, 0 : w - 2]
    )
    gy = (bot[:, 0 : w - 2] + 2 * bot[:, 1 : w - 1] + bot[:, 2:w]) - (
        top[:, 0 : w - 2] + 2 * top[:, 1 : w - 1] + top[:, 2:w]
    )
    out: U8Array = np.minimum(np.abs(gx) + np.abs(gy), MAX_PIXEL).astype(np.uint8)
    return out


def pipeline_valid(
    img: U8Array,
    r: int,
    iters: int,
    limits: TaskLimits = DEFAULT_TASK_LIMITS,
    deadline: float | None = None,
) -> U8Array:
    """``iters`` box blurs followed by Sobel; shrinks by ``halo(r, iters)`` per side.

    ``deadline`` (``time.monotonic()`` value) is checked between iterations.
    """
    _check_params(r, iters, limits)
    _check_image(img, 2 * halo(r, iters) + 1)
    x = img
    for _ in range(iters):
        check_deadline(deadline)
        x = box_blur_valid(x, r, limits)
    check_deadline(deadline)
    return sobel_valid(x)


def reference(
    img: U8Array, r: int, iters: int, limits: TaskLimits = DEFAULT_TASK_LIMITS
) -> U8Array:
    """Whole-image ground truth: edge-replicate pad by the halo, then run the pipeline.

    The output has the same shape as ``img``. Tiled results must equal this
    byte for byte.
    """
    _check_params(r, iters, limits)
    _check_image(img, 1)
    pad = halo(r, iters)
    padded: U8Array = np.pad(img, pad, mode="edge")
    return pipeline_valid(padded, r, iters, limits)
