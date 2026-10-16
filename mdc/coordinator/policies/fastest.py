"""``fastest`` policy: minimum predicted completion time (spec Phase 9 item 1).

Pulled forward to Phase 6 because it is the natural consumer of the network
estimates and adaptive chunking (see DECISIONS.md). Since Phase 8 a worker is
only considered for a chunk it can afford (battery floor, energy budget;
``SchedulingContext.energy_feasible``).

Repeat while there are ready tasks and free capacity:

1. take the first ready task (stage-2 first, then row-major tiles);
2. for every eligible worker ``d``: size a chunk for ``d`` starting there
   (``chunking.chunk_for`` — fast, stable workers get bigger chunks) and
   predict ``queue_wait + transfer + compute``;
3. assign the chunk to the worker with the smallest *per-tile* prediction
   (comparing whole-chunk times would always favour the worker offered the
   smallest chunk), ties broken by worker id;
4. add the prediction to that worker's queue, so later decisions in the same
   round see the extra load.

Reliability is ignored entirely — that is what ``most_reliable`` and ``tesar``
(Phase 9) add.
"""

from __future__ import annotations

from mdc.common.config import Config
from mdc.coordinator.chunking import chunk_for
from mdc.coordinator.policies.base import Assignment, SchedulingContext, WorkerView


class FastestPolicy:
    """Send each chunk where it is predicted to finish soonest."""

    name = "fastest"

    def __init__(self, config: Config) -> None:
        self.allow_local = config.scheduler.allow_local

    def candidates(self, ctx: SchedulingContext) -> list[WorkerView]:
        """Workers this policy may use (local only if ``allow_local``)."""
        return [w for w in ctx.workers if self.allow_local or not w.trusted]

    def assign(self, ctx: SchedulingContext) -> list[Assignment]:
        """Greedy minimum-predicted-time assignment."""
        workers = self.candidates(ctx)
        free = {w.worker_id: w.free for w in workers}
        out: list[Assignment] = []
        taken: set[str] = set(ctx.blocked)
        for task in ctx.ready:
            if task.task_id in taken:
                continue
            options = [w for w in workers if free[w.worker_id] > 0]
            if not options:
                break
            best: tuple[float, str, list[str], float, WorkerView] | None = None
            for worker in options:
                ids = chunk_for(worker, task, ctx, taken)
                if not ctx.energy_feasible(ids, worker):
                    continue
                predicted = ctx.predicted_s(ids, worker)
                key = (predicted / len(ids), worker.worker_id, ids, predicted, worker)
                if best is None or key[:2] < best[:2]:
                    best = key
            if best is None:  # nobody can afford this task right now
                continue
            _, _, ids, predicted, worker = best
            out.append(Assignment(tuple(ids), worker.worker_id, "primary", predicted))
            taken.update(ids)
            free[worker.worker_id] -= 1
            worker.inflight_remaining_s += predicted
            worker.energy_left_j -= ctx.energy_j(ids, worker)
        return out
