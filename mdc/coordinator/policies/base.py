"""Scheduling context and the policy interface (spec Phase 4 items 4-5).

A policy answers one question: *given the READY tasks and the workers with
free capacity right now, which tasks go to which worker?* It returns a list
of :class:`Assignment` and performs **no I/O** — no sockets, no clocks of its
own, no randomness without a seed. That makes every policy a pure function
of its :class:`SchedulingContext`, so it can be unit-tested with hand-built
contexts and compared fairly in experiments.

The scheduler (``scheduler.py``) builds the context, calls ``assign``, checks
each assignment again (the worker may have disconnected meanwhile) and hands
them to the dispatcher.

Task order in ``ctx.ready``: stage-2 tasks first, then stage-1 tiles in
row-major order — finishing rows early keeps the DAG flowing (a row's
statistics can start while other rows are still computing).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from mdc.common.config import Config
from mdc.coordinator.energy import EnergyModel
from mdc.coordinator.job import RUNNABLE, Job, TaskNode
from mdc.coordinator.reputation import task_reliability
from mdc.tasks.base import compute_seconds


@dataclass(frozen=True)
class Assignment:
    """Send ``task_ids`` (one chunk) to ``worker_id``."""

    task_ids: tuple[str, ...]
    worker_id: str
    purpose: str = "primary"
    predicted_s: float | None = None


@dataclass
class WorkerView:
    """What a policy may know about one eligible worker (read-only by convention)."""

    worker_id: str
    name: str
    trusted: bool
    device_class: str
    slots: int
    free: int  # how many more attempts it can take now (slots + prefetch - inflight)
    ms_per_unit: dict[str, float] = field(default_factory=dict)
    bench_ms_per_unit: float = 1.0
    battery: dict[str, Any] = field(default_factory=dict)
    # Network and load estimates (Phase 6):
    rtt_s: float = 0.0
    bw_bytes_per_s: float = 1_000_000.0
    failure_rate_per_s: float = 0.001  # lambda; measured from Phase 7
    rep: float = 0.5  # Beta reputation mean (Phase 7)
    eligible: bool = True  # False: battery below the floor and not charging
    energy_left_j: float = math.inf  # job energy budget not yet spent or committed (Phase 8)
    inflight_remaining_s: float = 0.0  # sum of remaining predicted time of in-flight attempts

    def speed(self, kind: str) -> float:
        """ms per work unit for ``kind`` (measured, else benchmark)."""
        return self.ms_per_unit.get(kind, self.bench_ms_per_unit)


@dataclass
class SchedulingContext:
    """Everything a policy may read when deciding (spec Phase 4 item 5)."""

    ready: list[TaskNode]
    workers: list[WorkerView]
    job: Job
    config: Config
    now: float
    # Tasks reserved by the scheduler (e.g. trusted-local fallback): never put them
    # into another assignment's chunk.
    blocked: set[str] = field(default_factory=set)
    # kind -> (wire/raw ratio of inputs, of outputs), observed by the NetEstimator.
    compression: dict[str, tuple[float, float]] = field(default_factory=dict)
    # Energy formulas (Phase 8); None: energy is not considered.
    energy: EnergyModel | None = None
    # Speed of the trusted local worker (ms per work unit by kind, and its benchmark),
    # even when it is busy: the TESAR cost normalizes times by "as slow as local".
    local_ms_per_unit: dict[str, float] = field(default_factory=dict)
    local_bench_ms_per_unit: float | None = None
    _wire_cache: dict[tuple[str, ...], tuple[float, float]] = field(
        default_factory=dict, init=False, repr=False
    )

    def worker(self, worker_id: str) -> WorkerView:
        """Look up an eligible worker by id."""
        for view in self.workers:
            if view.worker_id == worker_id:
                return view
        raise KeyError(worker_id)

    def local_worker(self) -> WorkerView | None:
        """The trusted local worker, if eligible."""
        return next((w for w in self.workers if w.trusted), None)

    def contiguous_ready(self, first: TaskNode, max_tiles: int, taken: set[str]) -> list[str]:
        """``first`` plus following runnable tiles of the same row, up to ``max_tiles``.

        Non-image tasks are never chunked. ``taken`` holds task ids already
        assigned in this round (so two assignments never share a tile).
        """
        ids = [first.task_id]
        grid = self.job.grid
        if first.kind != "img_tile" or grid is None or first.row is None or first.col is None:
            return ids
        col = first.col + 1
        while len(ids) < max_tiles and col < grid.cols:
            tid = self.job.tile_id(first.row, col)
            task = self.job.tasks[tid]
            if tid in taken or tid in self.blocked or task.status not in RUNNABLE:
                break
            ids.append(tid)
            col += 1
        return ids

    def compute_s(self, task_ids: list[str] | tuple[str, ...], worker: WorkerView) -> float:
        """Predicted compute time of a chunk on ``worker`` (spec 8.6)."""
        return chunk_compute_s(self.job, task_ids, worker.speed)

    def wire_bytes(self, task_ids: list[str] | tuple[str, ...]) -> tuple[float, float]:
        """Estimated (input, output) bytes on the wire for a chunk.

        Memoized for the round: the same chunk is costed on every worker, and the
        geometry does not depend on the worker (perf smoke: 20 workers x 200 tasks).
        """
        key = tuple(task_ids)
        cached = self._wire_cache.get(key)
        if cached is None:
            cached = chunk_wire_bytes(self.job, task_ids, self.compression)
            self._wire_cache[key] = cached
        return cached

    def transfer_s(self, task_ids: list[str] | tuple[str, ...], worker: WorkerView) -> float:
        """``rtt + (bytes_in + bytes_out) / bw`` (spec Phase 6 item 1)."""
        bytes_in, bytes_out = self.wire_bytes(task_ids)
        return worker.rtt_s + (bytes_in + bytes_out) / max(1.0, worker.bw_bytes_per_s)

    @staticmethod
    def queue_wait_s(worker: WorkerView) -> float:
        """Remaining predicted time of the worker's in-flight attempts / its slots."""
        return worker.inflight_remaining_s / max(1, worker.slots)

    def predicted_s(self, task_ids: list[str] | tuple[str, ...], worker: WorkerView) -> float:
        """``queue_wait + transfer + compute`` (spec Phase 6 item 2)."""
        return (
            self.queue_wait_s(worker)
            + self.transfer_s(task_ids, worker)
            + self.compute_s(task_ids, worker)
        )

    def local_compute_s(self, task_ids: list[str] | tuple[str, ...]) -> float:
        """Predicted compute time of a chunk on the trusted local worker (no transfer).

        Without a local worker the mean benchmark of the eligible workers stands in
        (only the *ratio* between devices matters for the normalized costs).
        """
        bench = self.local_bench_ms_per_unit
        if bench is None:
            benches = [w.bench_ms_per_unit for w in self.workers] or [1.0]
            bench = sum(benches) / len(benches)
        fallback = bench

        def speed(kind: str) -> float:
            return self.local_ms_per_unit.get(kind, fallback)

        return chunk_compute_s(self.job, task_ids, speed)

    def energy_j(self, task_ids: list[str] | tuple[str, ...], worker: WorkerView) -> float:
        """Predicted energy of a chunk on ``worker``: ``p_compute * compute + p_tx * transfer``."""
        if self.energy is None:
            return 0.0
        return self.energy.task_energy_j(
            worker.device_class,
            self.compute_s(task_ids, worker),
            self.transfer_s(task_ids, worker),
        )

    def energy_feasible(self, task_ids: list[str] | tuple[str, ...], worker: WorkerView) -> bool:
        """Battery floor and energy budget allow this chunk (the trusted local is exempt)."""
        if worker.trusted or self.energy is None or not self.energy.cfg.enabled:
            return True
        return worker.eligible and self.energy_j(task_ids, worker) <= worker.energy_left_j

    def reliability(self, task_ids: list[str] | tuple[str, ...], worker: WorkerView) -> float:
        """``R(task, d) = rep * exp(-λ * predicted_s)`` (spec Phase 7)."""
        return task_reliability(
            worker.rep,
            worker.failure_rate_per_s,
            self.predicted_s(task_ids, worker),
            worker.trusted,
            worker.eligible,
        )


def chunk_compute_s(
    job: Job, task_ids: list[str] | tuple[str, ...], speed: Callable[[str], float]
) -> float:
    """Sum of ``work_units * ms_per_unit / 1000`` over the chunk's tasks."""
    return sum(compute_seconds(job.tasks[t].work_units, speed(job.tasks[t].kind)) for t in task_ids)


def chunk_raw_bytes(job: Job, task_ids: list[str] | tuple[str, ...]) -> tuple[int, int]:
    """Uncompressed (input, output) bytes of a chunk.

    An image chunk's input is one rectangle (core + halo on every side) — the
    halo is shared between neighbouring tiles, so this is less than the sum of
    the tiles' individual inputs; that is part of why bigger chunks are cheaper.
    """
    first = job.tasks[task_ids[0]]
    if first.kind == "img_tile" and job.grid is not None and len(task_ids) > 1:
        row, c0, c1 = job.chunk_columns(list(task_ids))
        rect = job.grid.chunk_rect(row, c0, c1)
        halo2 = 2 * job.grid.halo
        return (rect.y1 - rect.y0 + halo2) * (rect.x1 - rect.x0 + halo2), rect.pixels
    return (
        sum(job.tasks[t].input_bytes_est for t in task_ids),
        sum(job.tasks[t].output_bytes_est for t in task_ids),
    )


def chunk_wire_bytes(
    job: Job,
    task_ids: list[str] | tuple[str, ...],
    compression: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    """Raw bytes times the observed compression ratio of the task kind."""
    raw_in, raw_out = chunk_raw_bytes(job, task_ids)
    ratio_in, ratio_out = compression.get(job.tasks[task_ids[0]].kind, (1.0, 1.0))
    return raw_in * ratio_in, raw_out * ratio_out


def predict_seconds(
    job: Job,
    task_ids: list[str] | tuple[str, ...],
    speed: Callable[[str], float],
    rtt_s: float,
    bw_bytes_per_s: float,
    queue_wait_s: float,
    compression: dict[str, tuple[float, float]],
) -> float:
    """The prediction of :meth:`SchedulingContext.predicted_s` from plain values
    (used by the dispatcher for assignments that come without a prediction)."""
    bytes_in, bytes_out = chunk_wire_bytes(job, task_ids, compression)
    transfer = rtt_s + (bytes_in + bytes_out) / max(1.0, bw_bytes_per_s)
    return queue_wait_s + transfer + chunk_compute_s(job, task_ids, speed)


class Policy(Protocol):
    """A scheduling policy."""

    name: str

    def assign(self, ctx: SchedulingContext) -> list[Assignment]:
        """Decide assignments for this scheduling round."""
        ...


PolicyFactory = Callable[[Config], Policy]
