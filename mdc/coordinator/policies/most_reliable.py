"""``most_reliable`` policy: maximum predicted reliability (spec Phase 9 item 2).

Repeat while there are ready tasks and free capacity:

1. take the first ready task;
2. for every eligible worker ``d`` that can afford it (battery, energy budget):
   size a chunk for ``d`` (adaptive chunking, as ``fastest`` does) and predict
   ``R(chunk, d) = rep(d) * exp(-λ(d) * predicted_s)`` (Phase 7);
3. assign the chunk to the worker with the highest ``R``; ties are broken by
   the lower predicted time, then by worker id.

Speed is only a tie-breaker, so this policy happily waits for the trusted
local worker (``R = 0.9999``) when ``scheduler.allow_local`` is true. It is the
``alpha = 0`` end of the TESAR tradeoff.
"""

from __future__ import annotations

from mdc.common.config import Config
from mdc.coordinator.chunking import chunk_for
from mdc.coordinator.policies.base import Assignment, SchedulingContext, WorkerView


class MostReliablePolicy:
    """Send each chunk where it is most likely to succeed."""

    name = "most_reliable"

    def __init__(self, config: Config) -> None:
        self.allow_local = config.scheduler.allow_local

    def assign(self, ctx: SchedulingContext) -> list[Assignment]:
        """Greedy maximum-reliability assignment."""
        workers = [w for w in ctx.workers if self.allow_local or not w.trusted]
        free = {w.worker_id: w.free for w in workers}
        out: list[Assignment] = []
        taken: set[str] = set(ctx.blocked)
        for task in ctx.ready:
            if task.task_id in taken:
                continue
            options = [w for w in workers if free[w.worker_id] > 0]
            if not options:
                break
            best: tuple[float, float, str, list[str], WorkerView] | None = None
            for worker in options:
                ids = chunk_for(worker, task, ctx, taken)
                if not ctx.energy_feasible(ids, worker):
                    continue
                predicted = ctx.predicted_s(ids, worker)
                key = (-ctx.reliability(ids, worker), predicted, worker.worker_id, ids, worker)
                if best is None or key[:3] < best[:3]:
                    best = key
            if best is None:  # nobody can afford this task right now
                continue
            _, predicted, _, ids, worker = best
            out.append(Assignment(tuple(ids), worker.worker_id, "primary", predicted))
            taken.update(ids)
            free[worker.worker_id] -= 1
            worker.inflight_remaining_s += predicted
            worker.energy_left_j -= ctx.energy_j(ids, worker)
        return out
