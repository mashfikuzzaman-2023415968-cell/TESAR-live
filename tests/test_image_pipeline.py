"""Tests for the integer image pipeline against a naive, obviously-correct reference."""

from __future__ import annotations

import hashlib
import time

import numpy as np
import pytest

from mdc.common.validation import TaskLimits, ValidationError
from mdc.tasks import image_pipeline as IP
from mdc.tasks.errors import TaskTimeLimitError


def naive_blur(img: np.ndarray, r: int) -> np.ndarray:
    """Double-loop box blur: out[y, x] = floor(mean of the k x k window at (y, x))."""
    k = 2 * r + 1
    h, w = img.shape
    out = np.zeros((h - 2 * r, w - 2 * r), dtype=np.uint8)
    for y in range(h - 2 * r):
        for x in range(w - 2 * r):
            total = 0
            for dy in range(k):
                for dx in range(k):
                    total += int(img[y + dy, x + dx])
            out[y, x] = total // (k * k)
    return out


GX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
GY = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]


def naive_sobel(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    out = np.zeros((h - 2, w - 2), dtype=np.uint8)
    for y in range(h - 2):
        for x in range(w - 2):
            gx = sum(GX[i][j] * int(img[y + i, x + j]) for i in range(3) for j in range(3))
            gy = sum(GY[i][j] * int(img[y + i, x + j]) for i in range(3) for j in range(3))
            out[y, x] = min(255, abs(gx) + abs(gy))
    return out


def rand_img(seed: int, shape: tuple[int, int] = (20, 20)) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=shape, dtype=np.uint8)


@pytest.mark.parametrize("r", [1, 2, 3])
@pytest.mark.parametrize("seed", range(4))
def test_blur_matches_naive(r: int, seed: int) -> None:
    img = rand_img(seed)
    np.testing.assert_array_equal(IP.box_blur_valid(img, r), naive_blur(img, r))


@pytest.mark.parametrize("seed", range(6))
def test_sobel_matches_naive(seed: int) -> None:
    img = rand_img(seed, (20, 23))
    np.testing.assert_array_equal(IP.sobel_valid(img), naive_sobel(img))


@pytest.mark.parametrize(("r", "iters"), [(1, 1), (1, 3), (2, 2)])
def test_pipeline_matches_naive_composition(r: int, iters: int) -> None:
    img = rand_img(7, (24, 21))
    expected = img
    for _ in range(iters):
        expected = naive_blur(expected, r)
    expected = naive_sobel(expected)
    np.testing.assert_array_equal(IP.pipeline_valid(img, r, iters), expected)


def test_blur_extreme_values_no_overflow() -> None:
    img = np.full((40, 40), 255, dtype=np.uint8)
    np.testing.assert_array_equal(IP.box_blur_valid(img, 5), 255)
    img[::2, ::2] = 0
    np.testing.assert_array_equal(IP.box_blur_valid(img, 1), naive_blur(img, 1))


@pytest.mark.parametrize("value", [0, 1, 128, 255])
def test_blur_constant_image_stays_constant(value: int) -> None:
    img = np.full((15, 17), value, dtype=np.uint8)
    out = IP.box_blur_valid(img, 2)
    assert out.shape == (11, 13)
    assert np.all(out == value)


@pytest.mark.parametrize(("shape", "r"), [((5, 5), 2), ((9, 30), 4), ((3, 3), 1)])
def test_blur_output_shape(shape: tuple[int, int], r: int) -> None:
    out = IP.box_blur_valid(np.zeros(shape, dtype=np.uint8), r)
    assert out.shape == (shape[0] - 2 * r, shape[1] - 2 * r)
    assert out.dtype == np.uint8


@pytest.mark.parametrize("value", [0, 77, 255])
def test_sobel_constant_image_is_zero(value: int) -> None:
    out = IP.sobel_valid(np.full((10, 12), value, dtype=np.uint8))
    assert out.shape == (8, 10)
    assert not out.any()


def test_sobel_saturates_at_255() -> None:
    img = np.zeros((5, 5), dtype=np.uint8)
    img[:, 3:] = 255  # a hard vertical edge: |gx| = 4 * 255
    assert IP.sobel_valid(img).max() == 255


def test_sobel_detects_vertical_and_horizontal_edges() -> None:
    vertical = np.zeros((6, 6), dtype=np.uint8)
    vertical[:, 3:] = 10
    horizontal = vertical.T.copy()
    assert IP.sobel_valid(vertical)[:, 1].tolist() == [40, 40, 40, 40]
    assert IP.sobel_valid(horizontal)[1, :].tolist() == [40, 40, 40, 40]


def test_halo_formula() -> None:
    assert IP.halo(2, 8) == 17
    assert IP.halo(1, 1) == 2


@pytest.mark.parametrize(("r", "iters"), [(1, 1), (2, 3), (3, 2)])
def test_pipeline_shape_shrinks_by_halo(r: int, iters: int) -> None:
    img = rand_img(1, (40, 50))
    h = IP.halo(r, iters)
    assert IP.pipeline_valid(img, r, iters).shape == (40 - 2 * h, 50 - 2 * h)


def test_reference_same_shape_and_deterministic() -> None:
    img = rand_img(3, (33, 47))
    a = IP.reference(img, 2, 3)
    b = IP.reference(img.copy(), 2, 3)
    assert a.shape == img.shape
    assert a.tobytes() == b.tobytes()


def test_reference_constant_image_has_no_edges() -> None:
    # Edge replication means the border creates no artificial edges.
    assert not IP.reference(np.full((20, 20), 90, dtype=np.uint8), 2, 2).any()


def test_reference_works_on_tiny_images() -> None:
    assert IP.reference(np.array([[5]], dtype=np.uint8), 1, 1).shape == (1, 1)


def test_reference_known_pinned_digest() -> None:
    """Regression pin: any change to the arithmetic (rounding, kernel orientation,
    padding mode) changes this hash. The value was recorded once on x86-64; the
    integer pipeline must reproduce it on every platform, including ARM phones."""
    out = IP.reference(rand_img(42, (32, 32)), 2, 2)
    assert hashlib.sha256(out.tobytes()).hexdigest() == (
        "cb480dd6233f82e321be5b3d7ca7216bd7fbc96e915a3f1c75c3f0fd1d2c7b16"
    )


def test_pipeline_deadline_exceeded() -> None:
    img = rand_img(1, (60, 60))
    with pytest.raises(TaskTimeLimitError, match="time_limit"):
        IP.pipeline_valid(img, 1, 5, deadline=time.monotonic() - 1)
    IP.pipeline_valid(img, 1, 5, deadline=time.monotonic() + 60)


@pytest.mark.parametrize(
    ("r", "iters"), [(0, 1), (6, 1), (1, 0), (1, 65), (True, 1), (1.0, 1), (1, np.int64(2))]
)
def test_pipeline_rejects_out_of_range_params(r: object, iters: object) -> None:
    with pytest.raises(ValidationError):
        IP.pipeline_valid(rand_img(1, (60, 60)), r, iters)  # type: ignore[arg-type]


def test_custom_limits_enforced() -> None:
    limits = TaskLimits(blur_radius=1, blur_iterations=2)
    with pytest.raises(ValidationError):
        IP.pipeline_valid(rand_img(1, (30, 30)), 2, 1, limits)
    with pytest.raises(ValidationError):
        IP.box_blur_valid(rand_img(1, (30, 30)), 2, limits)
    with pytest.raises(ValidationError):
        IP.reference(rand_img(1, (30, 30)), 1, 3, limits)


@pytest.mark.parametrize(
    "img",
    [
        np.zeros((20, 20), dtype=np.int32),
        np.zeros((20, 20), dtype=np.float64),
        np.zeros((20, 20, 3), dtype=np.uint8),
        np.zeros((20,), dtype=np.uint8),
        [[1, 2], [3, 4]],
    ],
)
def test_kernels_reject_wrong_arrays(img: object) -> None:
    for call in (
        lambda: IP.box_blur_valid(img, 1),  # type: ignore[arg-type]
        lambda: IP.sobel_valid(img),  # type: ignore[arg-type]
        lambda: IP.pipeline_valid(img, 1, 1),  # type: ignore[arg-type]
    ):
        with pytest.raises(ValidationError):
            call()


def test_kernels_reject_too_small_inputs() -> None:
    with pytest.raises(ValidationError, match="too small"):
        IP.box_blur_valid(np.zeros((4, 10), dtype=np.uint8), 2)
    with pytest.raises(ValidationError, match="too small"):
        IP.sobel_valid(np.zeros((2, 10), dtype=np.uint8))
    with pytest.raises(ValidationError, match="too small"):
        IP.pipeline_valid(np.zeros((10, 10), dtype=np.uint8), 2, 2)  # needs > 2*halo = 10
    IP.pipeline_valid(np.zeros((11, 11), dtype=np.uint8), 2, 2)  # boundary


def test_input_not_modified() -> None:
    img = rand_img(5, (30, 30))
    before = img.copy()
    IP.pipeline_valid(img, 2, 2)
    IP.reference(img, 1, 1)
    np.testing.assert_array_equal(img, before)
