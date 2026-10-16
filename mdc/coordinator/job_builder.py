"""Build image and synthetic jobs from the configuration (spec Phase 2).

Image job: load the image (Pillow, converted to 8-bit grayscale "L"), compute
the halo, pad once with edge replication, create the base grid and the DAG:

* stage 1: one ``img_tile`` task per base tile (no deps),
* stage 2: one ``row_stats`` task per tile row (deps = that row's tiles),
* stage 3: ``final`` (deps = all stage-2 tasks), run in-process.

Image loading is defensive (spec S.9.8): Pillow's decompression-bomb limit is
set to 50 MP and anything larger than 8192 x 8192 is refused.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from mdc.common.config import Config
from mdc.common.ids import new_id
from mdc.coordinator.job import FINAL, Job, TaskNode
from mdc.tasks.base import img_tile_work_units, row_stats_work_units, synthetic_work_units
from mdc.tasks.image_pipeline import halo as pipeline_halo
from mdc.tasks.tiling import TileGrid

U8Array = npt.NDArray[np.uint8]
MAX_IMAGE_SIDE = 8192
Image.MAX_IMAGE_PIXELS = 50_000_000  # Pillow raises DecompressionBombError above 2x this
ROW_STATS_OUTPUT_BYTES = 2048  # JSON histogram, roughly
SYNTHETIC_OUTPUT_BYTES = 128
SYNTHETIC_SEED_STRIDE = 1_000_003  # spreads seeds of different experiment seeds apart


class ImageLoadError(ValueError):
    """The image could not be loaded safely."""


def load_image(path: str | Path) -> U8Array:
    """Load an image as a 2-D uint8 grayscale array, with size limits."""
    try:
        with Image.open(path) as img:
            if img.width > MAX_IMAGE_SIDE or img.height > MAX_IMAGE_SIDE:
                raise ImageLoadError(
                    f"image {img.width}x{img.height} exceeds {MAX_IMAGE_SIDE}x{MAX_IMAGE_SIDE}"
                )
            gray = img.convert("L")
            gray.load()
    except Image.DecompressionBombError as exc:
        raise ImageLoadError(f"image rejected as a decompression bomb: {exc}") from None
    except (OSError, SyntaxError) as exc:  # Pillow raises these for bad/unknown files
        raise ImageLoadError(f"cannot read image {path}: {exc}") from None
    pixels: U8Array = np.array(gray, dtype=np.uint8)
    return pixels


def save_image(path: str | Path, array: U8Array) -> Path:
    """Save a 2-D uint8 array as PNG (parent directories created)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(array, dtype=np.uint8)).save(target, format="PNG")
    return target


def build_image_job_from_array(
    image: U8Array,
    base_tile: int,
    r: int,
    iters: int,
    edge_threshold: int,
    job_id: str | None = None,
    image_path: str | None = None,
) -> Job:
    """Build an image job from an in-memory image (used by tests and the CLI)."""
    if image.dtype != np.uint8 or image.ndim != 2:  # noqa: PLR2004
        raise ValueError("image must be a 2-D uint8 array")
    h = pipeline_halo(r, iters)
    grid = TileGrid(int(image.shape[0]), int(image.shape[1]), base_tile, h)
    tasks: dict[str, TaskNode] = {}
    tile_params = {"r": r, "iters": iters}
    for row in range(grid.rows):
        row_ids = []
        for col in range(grid.cols):
            rect = grid.tile_rect(row, col)
            core_h, core_w = rect.shape
            tid = f"s1_r{row:02d}_c{col:02d}"
            tasks[tid] = TaskNode(
                task_id=tid,
                stage=1,
                kind="img_tile",
                deps=[],
                work_units=img_tile_work_units(core_h, core_w, r, iters),
                input_bytes_est=(core_h + 2 * h) * (core_w + 2 * h),
                output_bytes_est=core_h * core_w,
                row=row,
                col=col,
                params=dict(tile_params),
            )
            row_ids.append(tid)
        strip_pixels = grid.chunk_rect(row, 0, grid.cols - 1).pixels
        sid = f"s2_r{row:02d}"
        tasks[sid] = TaskNode(
            task_id=sid,
            stage=2,
            kind="row_stats",
            deps=row_ids,
            work_units=row_stats_work_units(strip_pixels),
            input_bytes_est=strip_pixels,
            output_bytes_est=ROW_STATS_OUTPUT_BYTES,
            row=row,
            params={"edge_threshold": edge_threshold},
        )
    stage2 = [t for t in tasks if t.startswith("s2_")]
    tasks[FINAL] = TaskNode(task_id=FINAL, stage=3, kind="final", deps=stage2)
    return Job(
        job_id=job_id or new_id("job"),
        job_type="image",
        tasks=tasks,
        params={
            "base_tile": base_tile,
            "blur_radius": r,
            "blur_iterations": iters,
            "edge_threshold": edge_threshold,
            "halo": h,
        },
        grid=grid,
        padded=grid.pad(image),
        image_path=image_path,
    )


def build_image_job(cfg: Config, job_id: str | None = None) -> Job:
    """Build the image job described by ``cfg.job``."""
    j = cfg.job
    image = load_image(j.image_path)
    return build_image_job_from_array(
        image,
        j.base_tile,
        j.blur_radius,
        j.blur_iterations,
        j.edge_threshold,
        job_id,
        image_path=j.image_path,
    )


def build_synthetic_job(n_tasks: int, chain_len: int, seed: int, job_id: str | None = None) -> Job:
    """Independent hash-chain tasks plus a final XOR aggregation (no stage 2)."""
    if n_tasks < 1:
        raise ValueError("a synthetic job needs at least one task")
    tasks: dict[str, TaskNode] = {}
    for i in range(n_tasks):
        tid = f"syn_{i:04d}"
        tasks[tid] = TaskNode(
            task_id=tid,
            stage=1,
            kind="synthetic",
            deps=[],
            work_units=synthetic_work_units(chain_len),
            input_bytes_est=0,
            output_bytes_est=SYNTHETIC_OUTPUT_BYTES,
            col=i,
            params={"seed": seed * SYNTHETIC_SEED_STRIDE + i, "chain_len": chain_len},
        )
    tasks[FINAL] = TaskNode(task_id=FINAL, stage=3, kind="final", deps=list(tasks))
    params: dict[str, Any] = {"synthetic_tasks": n_tasks, "chain_len": chain_len, "seed": seed}
    return Job(job_id=job_id or new_id("job"), job_type="synthetic", tasks=tasks, params=params)


def build_synthetic_job_from_config(cfg: Config, job_id: str | None = None) -> Job:
    """Build the synthetic job described by ``cfg.job``."""
    return build_synthetic_job(
        cfg.job.synthetic_tasks, cfg.job.synthetic_chain_len, cfg.experiment.seed, job_id
    )


def build_job(cfg: Config, job_id: str | None = None) -> Job:
    """Build the job of type ``cfg.job.type``."""
    if cfg.job.type == "image":
        return build_image_job(cfg, job_id)
    return build_synthetic_job_from_config(cfg, job_id)
