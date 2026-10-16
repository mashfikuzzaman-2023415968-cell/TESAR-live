"""``local`` policy: run everything on the trusted local worker (the baseline).

This is the "no offloading" reference every experiment compares against:
speedup = local makespan / offloaded makespan. Chunks use the fixed size
``chunking.fixed_chunk_tiles``.
"""

from __future__ import annotations

from mdc.common.config import Config
from mdc.coordinator.policies.base import Assignment, SchedulingContext


class LocalOnlyPolicy:
    """Assign every ready task to the local worker while it has capacity."""

    name = "local"

    def __init__(self, config: Config) -> None:
        self.chunk_tiles = config.chunking.fixed_chunk_tiles

    def assign(self, ctx: SchedulingContext) -> list[Assignment]:
        """Fill the local worker's free capacity, in ``ctx.ready`` order."""
        local = ctx.local_worker()
        if local is None:
            return []
        out: list[Assignment] = []
        taken: set[str] = set()
        free = local.free
        for task in ctx.ready:
            if free <= 0:
                break
            if task.task_id in taken:
                continue
            ids = ctx.contiguous_ready(task, self.chunk_tiles, taken)
            taken.update(ids)
            out.append(Assignment(tuple(ids), local.worker_id))
            free -= 1
        return out
