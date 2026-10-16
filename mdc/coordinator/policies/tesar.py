"""``tesar`` policy: the speedup-reliability tradeoff (spec Phase 9 item 3).

This is the project's main contribution, inspired by the TESAR formulation
(see ``docs/TESAR_ALIGNMENT.md``). For each ready task we weigh *how fast* a
device would finish it against *how likely* it is to finish it at all, with
one knob, ``scheduler.alpha``.

Algorithm (greedy, online — it runs every scheduling round)::

    for the first READY task t (stage-2 tasks first, then row-major tiles):
      T_loc(t) = predicted compute time of t on the trusted local worker
      for each eligible, energy-feasible device d (local only if allow_local):
          chunk = t plus following tiles (adaptive chunking for d), n = |chunk|
          Ts = predicted_s(chunk, d)        # queue wait + rtt + bytes/bw + compute
          Rd = R(chunk, d)                  # rep(d) * exp(-λ(d) * Ts)
          speed_cost = (Ts / n) / T_loc(t)  # time per tile; 1.0 = as slow as local
          rel_cost   = -ln(max(Rd, 1e-6)) / n
          cost(d)    = alpha * speed_cost + (1 - alpha) * kappa * rel_cost
      d* = argmin cost (ties: worker id); assign chunk to d* (primary)
      if replication and R(chunk, d*) < reliability_target:
          add replicas of the SAME chunk on the next-cheapest other devices
          while 1 - Π(1 - R_i) < target and replicas < max_replicas

**Why ``-ln R``?** The job succeeds only if every task does, so its reliability
is the *product* of the task reliabilities. Maximizing a product is the same as
minimizing ``Σ -ln R``: the logarithm turns the reliability term into a sum,
exactly like time. That makes the two terms commensurable in one additive cost
(``kappa`` sets the exchange rate), keeps the greedy choice per task
consistent with the whole-job objective, and is what makes the offline model
in ``milp_offline.py`` linear. Dividing by ``n`` puts both terms on a per-tile
basis, so a device is not favoured merely because it was offered a smaller
chunk.

``alpha = 1`` reduces to ``fastest`` (same argmin); ``alpha = 0`` to a
per-tile version of ``most_reliable``. Sweeping alpha is experiment E2.

**Replicas.** A replica is a second copy of the same chunk on another device;
the first accepted result wins and the dispatcher cancels the others. Two
copies fail together only if both fail, so the chunk's reliability becomes
``1 - (1 - R1)(1 - R2)``. ``scheduler.max_replicas`` bounds the number of
*extra* copies. Replicas never go to the device holding the primary (the
scheduler enforces this too) and are not added for the trusted local worker,
whose ``R = 0.9999`` already meets any sensible target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from mdc.common.config import Config
from mdc.coordinator.chunking import chunk_for
from mdc.coordinator.job import TaskNode
from mdc.coordinator.policies.base import Assignment, SchedulingContext, WorkerView

MIN_RELIABILITY = 1e-6


@dataclass
class Option:
    """One candidate device for one chunk."""

    worker: WorkerView
    ids: list[str]
    predicted_s: float
    reliability: float
    cost: float


class TesarPolicy:
    """Minimize ``alpha * speed_cost + (1 - alpha) * kappa * rel_cost`` per chunk."""

    name = "tesar"

    def __init__(self, config: Config) -> None:
        sc = config.scheduler
        self.alpha = sc.alpha
        self.kappa = sc.kappa
        self.allow_local = sc.allow_local
        self.replication = sc.replication
        self.target = sc.reliability_target
        self.max_replicas = sc.max_replicas

    # ------------------------------------------------------------------ cost

    def cost(self, predicted_s: float, reliability: float, n: int, t_loc: float) -> float:
        """The TESAR cost of a chunk of ``n`` tiles (see the module docstring)."""
        speed_cost = (predicted_s / n) / max(t_loc, 1e-9)
        rel_cost = -math.log(max(reliability, MIN_RELIABILITY)) / n
        return self.alpha * speed_cost + (1.0 - self.alpha) * self.kappa * rel_cost

    def option(
        self, ctx: SchedulingContext, worker: WorkerView, ids: list[str], t_loc: float
    ) -> Option | None:
        """Cost a chunk on ``worker``; ``None`` if the worker cannot afford it."""
        if not ctx.energy_feasible(ids, worker):
            return None
        predicted = ctx.predicted_s(ids, worker)
        reliability = ctx.reliability(ids, worker)
        cost = self.cost(predicted, reliability, len(ids), t_loc)
        return Option(worker, ids, predicted, reliability, cost)

    # ---------------------------------------------------------------- assign

    def assign(self, ctx: SchedulingContext) -> list[Assignment]:
        """Greedy minimum-cost assignment with optional replication."""
        workers = [w for w in ctx.workers if self.allow_local or not w.trusted]
        free = {w.worker_id: w.free for w in workers}
        out: list[Assignment] = []
        taken: set[str] = set(ctx.blocked)
        for task in ctx.ready:
            if task.task_id in taken:
                continue
            if not any(free[w.worker_id] > 0 for w in workers):
                break
            best = self._best(ctx, task, workers, free, taken)
            if best is None:  # nobody can afford this task right now
                continue
            self._take(ctx, best, free, out, "primary")
            taken.update(best.ids)
            if self.replication and not best.worker.trusted and best.reliability < self.target:
                self._replicate(ctx, best, workers, free, out)
        return out

    def _best(
        self,
        ctx: SchedulingContext,
        task: TaskNode,
        workers: list[WorkerView],
        free: dict[str, int],
        taken: set[str],
    ) -> Option | None:
        t_loc = ctx.local_compute_s((task.task_id,))
        best: Option | None = None
        for worker in workers:
            if free[worker.worker_id] <= 0:
                continue
            option = self.option(ctx, worker, chunk_for(worker, task, ctx, taken), t_loc)
            if option is None:
                continue
            if best is None or (option.cost, worker.worker_id) < (
                best.cost,
                best.worker.worker_id,
            ):
                best = option
        return best

    def _replicate(
        self,
        ctx: SchedulingContext,
        primary: Option,
        workers: list[WorkerView],
        free: dict[str, int],
        out: list[Assignment],
    ) -> None:
        """Add replicas of ``primary.ids`` until the combined R reaches the target."""
        t_loc = ctx.local_compute_s(primary.ids[:1])
        options = [
            option
            for w in workers
            if w.worker_id != primary.worker.worker_id and free[w.worker_id] > 0
            if (option := self.option(ctx, w, primary.ids, t_loc)) is not None
        ]
        options.sort(key=lambda o: (o.cost, o.worker.worker_id))
        failure = 1.0 - primary.reliability
        for option in options[: self.max_replicas]:
            if 1.0 - failure >= self.target:
                break
            self._take(ctx, option, free, out, "replica")
            failure *= 1.0 - option.reliability

    @staticmethod
    def _take(
        ctx: SchedulingContext,
        option: Option,
        free: dict[str, int],
        out: list[Assignment],
        purpose: str,
    ) -> None:
        worker = option.worker
        out.append(Assignment(tuple(option.ids), worker.worker_id, purpose, option.predicted_s))
        free[worker.worker_id] -= 1
        worker.inflight_remaining_s += option.predicted_s
        worker.energy_left_j -= ctx.energy_j(option.ids, worker)
