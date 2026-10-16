"""Event-driven scheduling loop (spec Phase 4 item 4).

The loop sleeps until something could change the plan — a job starts, a task
becomes READY, an attempt finishes or fails, a worker becomes ACTIVE or
leaves — or at most 0.5 s (a safety net, so a missed trigger can only delay
scheduling, never stall it). Each round:

1. collect *eligible* workers: ACTIVE (not SUSPECT, not QUARANTINED) with
   ``len(inflight) < slots + prefetch`` — prefetch keeps one extra task queued
   on the worker to hide network latency between tasks;
2. build a read-only :class:`SchedulingContext` and ask the policy;
3. re-check every returned assignment (capacity, tasks still runnable, no
   task assigned twice in one round) and pass it to the dispatcher.

Policies decide; the scheduler enforces. A buggy policy can therefore never
overload a worker or send the same tile twice as primary.

**Trusted fallback** (spec Phase 5 item 3): a task that has already had
``liveness.max_attempts_per_task`` attempts is taken away from the policy and
sent only to the trusted local worker (``task_fallback_local`` event). Unless
the local worker itself is broken, this guarantees that no task — and hence no
job — can be retried forever on unreliable devices.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from mdc.common.config import Config
from mdc.common.timeutil import monotonic, wait_event
from mdc.coordinator.dispatcher import Dispatcher
from mdc.coordinator.estimators import Estimators
from mdc.coordinator.job import RUNNABLE, Job, TaskNode
from mdc.coordinator.policies.base import (
    Assignment,
    Policy,
    SchedulingContext,
    WorkerView,
    chunk_compute_s,
    chunk_wire_bytes,
)
from mdc.coordinator.registry import Registry, WorkerRecord
from mdc.coordinator.reputation import battery_eligible

log = logging.getLogger(__name__)

SAFETY_TICK_S = 0.5


class Scheduler:
    """Runs the policy whenever something changes."""

    def __init__(
        self,
        config: Config,
        registry: Registry,
        dispatcher: Dispatcher,
        policy: Policy,
        estimators: Estimators | None = None,
    ) -> None:
        self.cfg = config
        self.registry = registry
        self.dispatcher = dispatcher
        self.policy = policy
        shared = getattr(dispatcher, "estimators", None)  # one set of models per run
        self.estimators: Estimators = (
            estimators
            if estimators is not None
            else shared
            if isinstance(shared, Estimators)
            else Estimators.from_config(config)
        )
        self.netest = self.estimators.netest
        self.reputation = self.estimators.reputation
        self.energy = self.estimators.energy
        self.job: Job | None = None
        self.fallback_logged: set[str] = set()
        self._warned_no_local = False
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self.rounds = 0

    def trigger(self) -> None:
        """Ask for a scheduling round soon (cheap; safe to call often)."""
        self._wake.set()

    def start_job(self, job: Job) -> None:
        """Schedule ``job`` from now on."""
        self.job = job
        self.fallback_logged = set()
        self.trigger()

    @staticmethod
    def attempts_used(job: Job, task_id: str) -> int:
        """Primary/replica attempts a task has had so far (verification does not count)."""
        return sum(
            1
            for a in job.tasks[task_id].attempts
            if job.attempts[a].purpose in {"primary", "replica"}
        )

    def _fallback(
        self, job: Job, ready: list[TaskNode], workers: list[WorkerRecord]
    ) -> tuple[set[str], int]:
        """Send exhausted tasks to the trusted local worker; return (reserved ids, sent)."""
        limit = self.cfg.liveness.max_attempts_per_task
        exhausted = [t for t in ready if self.attempts_used(job, t.task_id) >= limit]
        if not exhausted:
            return set(), 0
        if not self.cfg.workers.local_worker:
            if not self._warned_no_local:
                log.warning(
                    "tasks exceeded %d attempts but there is no local worker to fall "
                    "back to; they stay with the policy",
                    limit,
                )
                self._warned_no_local = True
            return set(), 0
        reserved = {t.task_id for t in exhausted}
        local = next((r for r in workers if r.trusted), None)
        sent = 0
        for task in exhausted:
            if local is None or self.capacity(local) <= 0:
                break  # the local worker is busy or restarting: wait for it
            assignment = Assignment((task.task_id,), local.worker_id, "primary")
            if self.dispatcher.send(assignment) is not None:
                sent += 1
                if task.task_id not in self.fallback_logged:
                    self.fallback_logged.add(task.task_id)
                    self.dispatcher.events.emit(
                        "task_fallback_local",
                        task_id=task.task_id,
                        attempts=self.attempts_used(job, task.task_id) - 1,
                    )
        return reserved, sent

    def _local_takeover(self, job: Job, ready: list[TaskNode], workers: list[WorkerRecord]) -> int:
        local = next((r for r in workers if r.trusted), None)
        sent = 0
        for task in ready:
            if local is None or self.capacity(local) <= 0:
                break
            if self.dispatcher.send(Assignment((task.task_id,), local.worker_id)) is None:
                break
            sent += 1
            if task.task_id not in self.fallback_logged:
                self.fallback_logged.add(task.task_id)
                self.dispatcher.events.emit(
                    "task_fallback_local", task_id=task.task_id, reason="no_remote_workers"
                )
        return sent

    def stop_job(self) -> None:
        """Stop scheduling (the job finished or was aborted)."""
        self.job = None

    def capacity(self, record: WorkerRecord) -> int:
        """How many more attempts ``record`` may take right now."""
        return record.slots + self.cfg.workers.prefetch - len(record.inflight)

    @staticmethod
    def inflight_remaining_s(record: WorkerRecord, now: float) -> float:
        """Sum over in-flight attempts of ``max(0, predicted_s - elapsed)`` (spec Phase 6)."""
        return sum(
            max(0.0, a.predicted_s - (now - (a.sent_t or now))) for a in record.inflight.values()
        )

    def eligible(self) -> list[WorkerRecord]:
        """ACTIVE workers with free capacity that are not in an error cooldown."""
        now = monotonic()
        return [
            r for r in self.registry.active() if self.capacity(r) > 0 and now >= r.cooldown_until
        ]

    def schedule_once(self) -> int:
        """One scheduling round; returns the number of attempts sent."""
        job = self.job
        if job is None:
            return 0
        workers = self.eligible()
        if not workers:
            return 0
        # Verification checks first (Phase 10): they gate the job's end and are small.
        verifier = getattr(self.dispatcher, "verifier", None)
        checks_sent = verifier.dispatch(workers, self.capacity) if verifier is not None else 0
        workers = [r for r in workers if self.capacity(r) > 0]
        ready = job.ready_tasks()
        if not ready or not workers:
            return checks_sent
        reserved, sent = self._fallback(job, ready, workers)
        sent += checks_sent
        ready = [t for t in ready if t.task_id not in reserved]
        if not self.registry.remote_active():
            # Every remote worker is gone, dead or quarantined: the trusted local worker
            # finishes the job even if allow_local is false — a job must never hang.
            self.rounds += 1
            return sent + self._local_takeover(job, ready, workers)
        workers = [r for r in workers if self.capacity(r) > 0]
        if not ready or not workers:
            self.rounds += 1
            return sent
        workers = self._energy_gate(job, ready, workers)
        if not workers:
            self.rounds += 1
            return sent
        now = monotonic()
        views = [
            WorkerView(
                worker_id=r.worker_id,
                name=r.name,
                trusted=r.trusted,
                device_class=r.device_class,
                slots=r.slots,
                free=self.capacity(r),
                ms_per_unit={k: e.value for k, e in r.speed.items() if e.value is not None},
                bench_ms_per_unit=r.bench_ms_per_unit,
                battery=dict(r.battery),
                rtt_s=self.netest.rtt(r.worker_id),
                bw_bytes_per_s=self.netest.bw(r.worker_id),
                failure_rate_per_s=self.reputation.failure_rate(r.worker_id),
                rep=self.reputation.rep(r.worker_id),
                eligible=r.trusted or battery_eligible(r.battery, self.cfg.workers.min_battery_pct),
                energy_left_j=self.energy.ledger(r.worker_id).remaining_j,
                inflight_remaining_s=self.inflight_remaining_s(r, now),
            )
            for r in workers
        ]
        local = next((r for r in self.registry if r.trusted), None)  # even if busy
        local_speeds = (
            {k: e.value for k, e in local.speed.items() if e.value is not None} if local else {}
        )
        local_bench = local.bench_ms_per_unit if local is not None else None
        ctx = SchedulingContext(
            ready=ready,
            workers=views,
            job=job,
            config=self.cfg,
            now=monotonic(),
            blocked=reserved,
            compression=self.netest.compression_table(),
            energy=self.energy,
            local_ms_per_unit=local_speeds,
            local_bench_ms_per_unit=local_bench,
        )
        by_id = {v.worker_id: v for v in views}
        taken: set[str] = set(reserved)
        for assignment in self.policy.assign(ctx):
            record = self.registry.get(assignment.worker_id)
            if record is None or record not in workers or self.capacity(record) <= 0:
                continue
            ids = assignment.task_ids
            if not ids or any(t not in job.tasks for t in ids):
                continue
            if assignment.purpose == "primary" and (
                any(t in taken for t in ids)
                or any(job.tasks[t].status not in RUNNABLE for t in ids)
            ):
                continue
            if any(a.worker_id == record.worker_id for t in ids for a in job.inflight_attempts(t)):
                continue  # a replica must run on a different device
            view = by_id[record.worker_id]
            view.energy_left_j = self.energy.ledger(record.worker_id).remaining_j
            if not ctx.energy_feasible(ids, view):  # enforced whatever the policy did
                if view.eligible:
                    self._log_exhausted(record.worker_id)
                continue
            if self.dispatcher.send(assignment) is not None:
                taken.update(ids)
                sent += 1
        self.rounds += 1
        return sent

    def _energy_gate(
        self, job: Job, ready: list[TaskNode], workers: list[WorkerRecord]
    ) -> list[WorkerRecord]:
        """Drop remote workers below the battery floor (not charging) or out of energy
        budget for even the next ready task (compute + transfer, as the policies
        predict it); log each once (spec Phase 8 item 3).

        The trusted local worker is exempt. Budgets are fixed the first time a worker
        is considered during the job (job start for workers already connected).
        """
        if not self.energy.cfg.enabled:
            return workers
        floor = self.cfg.workers.min_battery_pct
        kept = []
        for record in workers:
            entry = self.energy.open_budget(
                record.worker_id, record.device_class, record.battery, record.trusted
            )
            if record.trusted:
                kept.append(record)
                continue
            if not battery_eligible(record.battery, floor):
                if not entry.battery_logged:
                    entry.battery_logged = True
                    self.dispatcher.events.emit(
                        "worker_ineligible_battery",
                        worker_id=record.worker_id,
                        battery_pct=record.battery.get("pct"),
                        charging=record.battery.get("charging"),
                        min_battery_pct=floor,
                    )
                continue
            entry.battery_logged = False  # log again if it drops below the floor later
            first = (ready[0].task_id,)
            bytes_in, bytes_out = chunk_wire_bytes(job, first, self.netest.compression_table())
            transfer_s = self.netest.rtt(record.worker_id) + (bytes_in + bytes_out) / max(
                1.0, self.netest.bw(record.worker_id)
            )
            next_j = self.energy.task_energy_j(
                record.device_class,
                chunk_compute_s(job, first, record.ms_per_unit),
                transfer_s,
            )
            if entry.remaining_j < next_j:
                self._log_exhausted(record.worker_id)
                continue
            kept.append(record)
        return kept

    def _log_exhausted(self, worker_id: str) -> None:
        """``energy_budget_exhausted``, once per worker and job."""
        entry = self.energy.ledger(worker_id)
        if not entry.exhausted_logged:
            entry.exhausted_logged = True
            self.dispatcher.events.emit(
                "energy_budget_exhausted",
                worker_id=worker_id,
                budget_j=round(entry.budget_j, 6),
                spent_j=round(entry.spent_j, 6),
                committed_j=round(entry.committed_j, 6),
            )

    async def run(self) -> None:
        """The loop (run as a background task)."""
        while not self._stopping:
            await wait_event(self._wake, SAFETY_TICK_S)
            self._wake.clear()
            if self._stopping:
                break
            try:
                self.schedule_once()
            except Exception:
                # A scheduling bug must be visible but must not kill the coordinator.
                log.exception("scheduling round failed")

    def start(self) -> None:
        """Start the background loop."""
        if self._task is None:
            self._stopping = False
            self._task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        """Stop the background loop (flag + wake-up + cancel: never relies on one alone)."""
        if self._task is not None:
            self._stopping = True
            self._wake.set()
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
