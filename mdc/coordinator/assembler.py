"""Output assembly and the correctness check (spec Phase 4 item 8, 11.1).

At job start the **reference** result is computed in a background thread
(whole-image pipeline for image jobs; all hash chains for synthetic jobs), so
it is ready when the distributed job ends. Image references are cached in
``runs/.cache/`` keyed by a hash of the image pixels and the parameters, so
repeated experiment runs do not recompute them.

When every stage-2 task has a result, the coordinator runs stage 3
(``job.run_final()``) in a worker thread, writes ``output.png`` and
``reference.png``, and compares byte for byte: ``output_correct`` and
``differing_pixels`` go into ``report.json``.

Large-array work (assembly, PNG encoding, reference) never runs on the event
loop (spec Section 12).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from mdc.coordinator.job import Job
from mdc.coordinator.job_builder import load_image, save_image
from mdc.tasks.image_pipeline import reference as image_reference
from mdc.tasks.synthetic import hash_chain, xor_digests

log = logging.getLogger(__name__)

U8Array = npt.NDArray[np.uint8]


def reference_cache_key(image: U8Array, r: int, iters: int) -> str:
    """Stable key for the reference of ``image`` with ``(r, iters)``."""
    h = hashlib.sha256()
    h.update(f"{image.shape}|{r}|{iters}|".encode())
    h.update(np.ascontiguousarray(image).tobytes())
    return h.hexdigest()[:24]


def original_image(job: Job) -> U8Array:
    """The job's input image, recovered from the padded copy."""
    if job.padded is None or job.grid is None:
        raise ValueError("not an image job")
    h = job.grid.halo
    out: U8Array = job.padded[h : h + job.grid.height, h : h + job.grid.width]
    return out


def compute_reference(job: Job, cache_dir: Path | None) -> Any:
    """Ground truth for ``job`` (blocking; run it in a thread)."""
    if job.job_type == "synthetic":
        digests = [
            hash_chain(int(t.params["seed"]), int(t.params["chain_len"]))
            for t in job.tasks.values()
            if t.kind == "synthetic"
        ]
        return {"digest": xor_digests(digests)}
    image = original_image(job)
    r, iters = int(job.params["blur_radius"]), int(job.params["blur_iterations"])
    path = None
    if cache_dir is not None:
        path = cache_dir / f"ref_{reference_cache_key(image, r, iters)}.png"
        if path.exists():
            try:
                cached = load_image(path)
                if cached.shape == image.shape:
                    return cached
            except ValueError:
                log.warning("ignoring unreadable cached reference %s", path)
    ref = image_reference(image, r, iters)
    if path is not None:
        save_image(path, ref)
    return ref


def wrong_tiles(job: Job, reference: Any) -> int:
    """Stage-1 tiles whose result in the final output differs from the reference:
    wrong results that reached ``final`` without being repaired (spec Phase 10 item 7)."""
    if job.grid is None:
        return 0
    count = 0
    for (row, col), tile in job.tile_results().items():
        rect = job.grid.chunk_rect(row, col, col)
        if not np.array_equal(tile, reference[rect.y0 : rect.y1, rect.x0 : rect.x1]):
            count += 1
    return count


class Assembler:
    """Computes the reference and finishes jobs."""

    def __init__(self, output_dir: Path, cache_dir: Path | None) -> None:
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self._reference: asyncio.Future[Any] | None = None

    def start_reference(self, job: Job) -> None:
        """Begin computing the reference in a background thread."""
        loop = asyncio.get_running_loop()
        self._reference = loop.run_in_executor(None, compute_reference, job, self.cache_dir)

    async def finish(self, job: Job) -> dict[str, Any]:
        """Run stage 3, write outputs, compare with the reference."""
        loop = asyncio.get_running_loop()
        summary = await loop.run_in_executor(None, job.run_final)
        reference = await self._reference if self._reference is not None else None
        result: dict[str, Any] = {"output_correct": None, "differing_pixels": None}
        if job.job_type == "image":
            output = summary["output"]
            await loop.run_in_executor(None, save_image, self.output_dir / "output.png", output)
            result["edge_density"] = summary["edge_density"]
            if reference is not None:
                await loop.run_in_executor(
                    None, save_image, self.output_dir / "reference.png", reference
                )
                differing = int(np.count_nonzero(output != reference))
                result.update(
                    output_correct=differing == 0,
                    differing_pixels=differing,
                    wrong_results_reached_final=wrong_tiles(job, reference),
                )
        else:
            result["digest"] = summary["digest"]
            if reference is not None:
                same = summary["digest"] == reference["digest"]
                result.update(output_correct=same, differing_pixels=0 if same else None)
        return result
