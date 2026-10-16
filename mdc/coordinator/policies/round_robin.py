"""``round_robin`` policy: cycle through eligible workers, one base tile each.

The simplest offloading strategy: it ignores speed, reliability and network
entirely. It is a baseline that shows what the smarter policies gain. It does
respect the device owners' limits (battery floor, energy budget; Phase 8): a
worker that cannot afford a task is skipped for it. The local worker takes part
only if ``scheduler.allow_local`` is true.

The rotation position persists between scheduling rounds, so load spreads
evenly even when only one worker frees up at a time.
"""

from __future__ import annotations

from mdc.common.config import Config
from mdc.coordinator.policies.base import Assignment, SchedulingContext


class RoundRobinPolicy:
    """Give each eligible worker one tile in turn."""

    name = "round_robin"

    def __init__(self, config: Config) -> None:
        self.allow_local = config.scheduler.allow_local
        self._next = 0  # index into the sorted worker list (persists across rounds)

    def assign(self, ctx: SchedulingContext) -> list[Assignment]:
        """Hand out ready tasks one by one to workers in rotation."""
        workers = sorted(
            (w for w in ctx.workers if self.allow_local or not w.trusted),
            key=lambda w: w.worker_id,
        )
        free = {w.worker_id: w.free for w in workers}
        out: list[Assignment] = []
        tasks = list(ctx.ready)
        while tasks and workers and any(f > 0 for f in free.values()):
            task = tasks.pop(0)
            ids = (task.task_id,)
            for _ in range(len(workers)):
                worker = workers[self._next % len(workers)]
                self._next += 1
                if free[worker.worker_id] > 0 and ctx.energy_feasible(ids, worker):
                    out.append(Assignment(ids, worker.worker_id))
                    free[worker.worker_id] -= 1
                    worker.energy_left_j -= ctx.energy_j(ids, worker)
                    break
        return out
