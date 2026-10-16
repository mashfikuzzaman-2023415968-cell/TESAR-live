"""Generate deterministic grayscale test images into ``data/images/``.

Images: ``test_512.png``, ``test_1024.png``, ``test_2048.png`` (square) and
``test_1000x1300.png`` (width 1000, height 1300 — deliberately *not* a
multiple of the 128-pixel base tile, so edge tiles are smaller).

Each image is a smooth gradient + random rectangles and discs + mild noise,
so the edge detector has plenty of real edges to find. A fixed seed per image
(``numpy.random.default_rng``) makes the pixel content identical on every
machine; this is experiment randomness, not security randomness.

Usage: ``python scripts/make_test_images.py [--out data/images] [--force]``
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# (file name, width, height, seed)
IMAGES: list[tuple[str, int, int, int]] = [
    ("test_512.png", 512, 512, 512),
    ("test_1024.png", 1024, 1024, 1024),
    ("test_2048.png", 2048, 2048, 2048),
    ("test_1000x1300.png", 1000, 1300, 1300),
]

RECTANGLE_SHARE = 0.5  # half the shapes are rectangles, half discs


def make_image(width: int, height: int, seed: int) -> np.ndarray:
    """Return a deterministic ``uint8`` array of shape ``(height, width)``."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    # Diagonal gradient as the background.
    img = 60.0 + 80.0 * (xx / max(1, width - 1)) + 40.0 * (yy / max(1, height - 1))

    # Shape count scales with area so every tile has some structure.
    n_shapes = max(8, (width * height) // 12000)
    for _ in range(n_shapes):
        value = float(rng.integers(0, 256))
        if rng.random() < RECTANGLE_SHARE:
            w = int(rng.integers(8, max(9, width // 6)))
            h = int(rng.integers(8, max(9, height // 6)))
            x0 = int(rng.integers(0, max(1, width - w)))
            y0 = int(rng.integers(0, max(1, height - h)))
            img[y0 : y0 + h, x0 : x0 + w] = value
        else:
            r = float(rng.integers(6, max(7, min(width, height) // 10)))
            cx = float(rng.integers(0, width))
            cy = float(rng.integers(0, height))
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
            img[mask] = value

    img += rng.normal(0.0, 6.0, size=img.shape)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def main(argv: list[str] | None = None) -> int:
    """Generate all images; print one line per image with its pixel hash."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data/images", help="output directory")
    parser.add_argument("--force", action="store_true", help="overwrite existing images")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, width, height, seed in IMAGES:
        path = out_dir / name
        if path.exists() and not args.force:
            print(f"exists   {path}")
            continue
        pixels = make_image(width, height, seed)
        Image.fromarray(pixels).save(path, optimize=True)
        digest = hashlib.sha256(pixels.tobytes()).hexdigest()[:16]
        print(f"wrote    {path}  {width}x{height}  pixels-sha256={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
