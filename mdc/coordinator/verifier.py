"""Result verification and malicious-worker detection (spec Phase 10, S.12 row 10).

Two different checks — kept apart on purpose:

* **Transport integrity** (Phase 1, ``protocol.py``): the payload's SHA-256
  matches its header. It detects corruption *in transit*; a failure is a
  transmission error, not dishonesty.
* **Computational correctness** (this module): does the result equal what an
  honest device computes? A worker can return a well-formed, correctly hashed,
  plausible but *wrong* result. Only re-execution can tell.

**What gets verified.** Every result accepted from an untrusted worker::

    worker's accepted results so far <= probation_tasks  -> verify
    otherwise verify with p = min(1, base_prob + (1 - rep) * low_rep_boost)   (seeded rng)
    results of the trusted local worker are never verified
    results confirmed by a matching replica are verified (collusion rule below)

**How.** A *check* re-executes the producing attempt's chunk with
``purpose="verify"`` — on the trusted local worker if
``prefer_local_verifier`` (else the eligible worker with the highest reputation
that did not produce it). The task becomes VERIFYING but **dependents may
proceed** with the unverified result (optimistic execution keeps the speed; a
wrong result is repaired below). Results are compared by ``result_sha256`` per
tile (array tasks) or canonical-JSON hash (JSON tasks):

* match → ``verified``; the producer's ``verified_ok`` grows;
* mismatch with a **trusted** verifier → the producer is wrong: reputation
  ``wrong_result``, ``wrong += 1``, ``strikes += 1``, the verifier's result
  replaces the stored one and every dependent is invalidated (taint
  propagation, ``Job.replace_result``);
* mismatch with an **untrusted** verifier → a ``tiebreak`` check on the trusted
  local worker; whoever disagrees with it gets the strike (possibly both).

**Quarantine.** ``strikes >= strikes_to_quarantine`` → QUARANTINED (sticky,
survives reconnects: the registry never leaves that state), in-flight attempts
cancelled and requeued, ``worker_quarantined`` logged, and **retroactive
repair**: every result it produced that is not verified is invalidated and
re-executed by others, cascading to dependents.

**Collusion rule (S.3).** Two matching results from different untrusted
workers count as a verification only if one producer is trusted, or has
``rep >= 0.9`` and at least 10 results verified by the trusted worker;
otherwise a trusted spot check still runs. Every worker that returned the same
wrong hash gets the strike, so two colluders returning identical corruption are
both caught.

The job only finishes when ``final`` can run **and** no check is outstanding.
"""

from __future__ import annotations

import hmac
import itertools
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from mdc.common.config import Config
from mdc.common.timeutil import monotonic
from mdc.coordinator.job import SATISFIED, Attempt, Job, TaskStatus, result_hash
from mdc.coordinator.policies.base import Assignment
from mdc.coordinator.registry import WorkerRecord, WorkerState

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mdc.coordinator.dispatcher import Dispatcher

COLLUSION_MIN_REP = 0.9
COLLUSION_MIN_TRUSTED_VERIFIED = 10
VERIFY = "verify"
TIEBREAK = "tiebreak"
PRODUCING = frozenset({"primary", "replica"})
ALIVE = frozenset(
    {WorkerState.CONNECTING, WorkerState.PROBING, WorkerState.ACTIVE, WorkerState.SUSPECT}
)


def same_hash(a: str | None, b: str | None) -> bool:
    """Compare two result hashes (constant time, as for every digest in this code base)."""
    return a is not None and b is not None and hmac.compare_digest(a, b)


class Capacity(Protocol):
    """How many more attempts a worker may take (the scheduler's rule)."""

    def __call__(self, record: WorkerRecord) -> int: ...


@dataclass
class Check:
    """One pending re-execution of a chunk and the results it will judge.

    ``claims`` maps each worker that returned a result for these tasks to
    ``{task_id: hash}``; ``holder`` is the worker whose result is stored in the
    job (the one a mismatch replaces).
    """

    check_id: int
    kind: str  # VERIFY | TIEBREAK
    task_ids: tuple[str, ...]
    holder: str
    claims: dict[str, dict[str, str]]
    accepted_t: float
    trusted_only: bool = False
    attempt_id: str | None = None

    def claimants(self) -> set[str]:
        """Workers whose results this check judges (they never verify themselves)."""
        return set(self.claims)


@dataclass
class VerificationStats:
    """Counters for ``report.json`` (spec Phase 10 item 7)."""

    checks_created: int = 0
    attempts_sent: int = 0
    matches: int = 0
    mismatches: int = 0
    tiebreaks: int = 0
    free_replica_matches: int = 0
    skipped_no_verifier: int = 0
    repaired_tasks: int = 0
    first_wrong_t: dict[str, float] = field(default_factory=dict)
    quarantined_t: dict[str, float] = field(default_factory=dict)

    def detection_latency_s(self) -> dict[str, float]:
        """Per quarantined worker: time from its first (caught) wrong result to quarantine."""
        return {
            w: round(max(0.0, t - self.first_wrong_t[w]), 4)
            for w, t in self.quarantined_t.items()
            if w in self.first_wrong_t
        }


class Verifier:
    """Decides what to verify, dispatches checks, judges results, quarantines."""

    def __init__(self, config: Config, dispatcher: Dispatcher, seed: int | None = None) -> None:
        self.cfg = config.verification
        self.local_configured = config.workers.local_worker
        self.dispatcher = dispatcher
        rng_seed = config.experiment.seed if seed is None else seed
        self.rng = random.Random(rng_seed)  # noqa: S311 - experiment sampling, not security
        self.checks: dict[int, Check] = {}
        self.stats = VerificationStats()
        self.trusted_verified: dict[str, int] = {}
        self._ids = itertools.count(1)

    # ------------------------------------------------------------ helpers

    @property
    def job(self) -> Job | None:
        return self.dispatcher.job

    def _record(self, worker_id: str) -> WorkerRecord | None:
        return self.dispatcher.registry.get(worker_id)

    def _trusted(self, worker_id: str) -> bool:
        record = self._record(worker_id)
        return record is not None and record.trusted

    def outstanding(self) -> int:
        """Checks queued or in flight (the job may not finish while > 0)."""
        return len(self.checks)

    def start_job(self) -> None:
        """Forget the previous job's checks."""
        self.checks.clear()

    def _new_check(
        self,
        kind: str,
        task_ids: tuple[str, ...],
        holder: str,
        claims: dict[str, dict[str, str]],
        accepted_t: float,
        trusted_only: bool = False,
    ) -> Check:
        check = Check(next(self._ids), kind, task_ids, holder, claims, accepted_t, trusted_only)
        self.checks[check.check_id] = check
        self.stats.checks_created += 1
        job = self.job
        if job is not None:
            for tid in check.task_ids:
                task = job.tasks[tid]
                if task.status is TaskStatus.DONE and not task.verified:
                    task.status = TaskStatus.VERIFYING
        return check

    def _pending_for(self, task_id: str) -> Check | None:
        return next((c for c in self.checks.values() if task_id in c.task_ids), None)

    def _claims_valid(self, job: Job, check: Check) -> dict[str, str]:
        """``{task_id: hash}`` of the holder's claims that are still the stored result."""
        claims = check.claims.get(check.holder, {})
        valid = {}
        for tid, digest in claims.items():
            task = job.tasks[tid]
            if (
                task.status in SATISFIED
                and task.produced_by == check.holder
                and same_hash(task.result_sha256, digest)
            ):
                valid[tid] = digest
        return valid

    def _settle_status(self, job: Job, task_ids: tuple[str, ...]) -> None:
        """VERIFYING tasks without another pending check go back to DONE."""
        for tid in task_ids:
            task = job.tasks[tid]
            if task.status is TaskStatus.VERIFYING and self._pending_for(tid) is None:
                task.status = TaskStatus.DONE

    def _close(self, check: Check) -> None:
        self.checks.pop(check.check_id, None)
        job = self.job
        if job is not None:
            self._settle_status(job, check.task_ids)

    # ------------------------------------------------------ deciding (item 2)

    def should_verify(self, record: WorkerRecord) -> bool:
        """Probation, then probability ``min(1, base + (1 - rep) * boost)``."""
        if not self.cfg.enabled or record.trusted:
            return False
        if record.completed <= self.cfg.probation_tasks:
            return True
        rep = self.dispatcher.reputation.rep(record.worker_id)
        p = min(1.0, self.cfg.base_prob + (1.0 - rep) * self.cfg.low_rep_boost)
        return self.rng.random() < p

    def on_accepted(self, attempt: Attempt, accepted: list[str]) -> None:
        """A primary/replica result was stored for ``accepted`` tasks (dispatcher hook)."""
        job, record = self.job, self._record(attempt.worker_id)
        if job is None or record is None or not accepted:
            return
        if record.trusted:
            for tid in accepted:
                job.tasks[tid].verified = True
            return
        if not self.should_verify(record):
            return
        claims = {attempt.worker_id: {t: str(job.tasks[t].result_sha256) for t in accepted}}
        self._new_check(
            VERIFY,
            tuple(attempt.task_ids),
            attempt.worker_id,
            claims,
            attempt.result_t or monotonic(),
        )

    def collusion_safe(self, worker_a: str, worker_b: str) -> bool:
        """May two matching results from these workers count as a verification?"""
        for worker in (worker_a, worker_b):
            if self._trusted(worker):
                return True
            if (
                self.dispatcher.reputation.rep(worker) >= COLLUSION_MIN_REP
                and self.trusted_verified.get(worker, 0) >= COLLUSION_MIN_TRUSTED_VERIFIED
            ):
                return True
        return False

    def on_duplicate(self, attempt: Attempt, output: Any) -> None:
        """A second result (e.g. a cancelled replica that finished anyway) for tasks that
        already hold another worker's result: compare for free (spec Phase 9 item 3)."""
        job = self.job
        if not self.cfg.enabled or job is None or attempt.purpose not in PRODUCING:
            return
        hashes = self._output_hashes(job, attempt.task_ids, output)
        for holder, ids in self._group_by_holder(job, hashes).items():
            if holder == attempt.worker_id:
                continue
            same = {t for t in ids if same_hash(job.tasks[t].result_sha256, hashes[t])}
            mine = {t: hashes[t] for t in ids}
            if same == set(ids) and self.collusion_safe(holder, attempt.worker_id):
                self.stats.free_replica_matches += 1
                for tid in ids:
                    job.tasks[tid].verified = True
                self._drop_queued(ids)
                continue
            # Disagreement, or agreement we may not trust: a trusted check decides.
            check = self._queued_check_for(ids)
            if check is None:
                stored = {t: str(job.tasks[t].result_sha256) for t in ids}
                check = self._new_check(
                    VERIFY,
                    tuple(attempt.task_ids),
                    holder,
                    {holder: stored},
                    attempt.result_t or monotonic(),
                    trusted_only=True,
                )
            check.claims.setdefault(attempt.worker_id, {}).update(mine)
            check.trusted_only = True

    def _group_by_holder(self, job: Job, hashes: dict[str, str]) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for tid in hashes:
            task = job.tasks[tid]
            if task.status in SATISFIED and not task.verified and task.produced_by is not None:
                groups.setdefault(task.produced_by, []).append(tid)
        return groups

    def _queued_check_for(self, task_ids: list[str]) -> Check | None:
        for check in self.checks.values():
            if set(task_ids) <= set(check.task_ids):
                return check
        return None

    def _drop_queued(self, task_ids: list[str]) -> None:
        for check in list(self.checks.values()):
            if check.attempt_id is None and set(check.task_ids) <= set(task_ids):
                self._close(check)

    # -------------------------------------------------------- dispatching

    def _verifier_for(
        self, check: Check, workers: list[WorkerRecord], capacity: Capacity
    ) -> WorkerRecord | None:
        free = [
            r
            for r in workers
            if capacity(r) > 0
            and r.worker_id not in check.claimants()
            and r.state is WorkerState.ACTIVE
        ]
        local = next((r for r in free if r.trusted), None)
        if check.trusted_only or (self.cfg.prefer_local_verifier and self.local_configured):
            return local  # wait for the trusted worker if it is busy
        if local is not None and self.cfg.prefer_local_verifier:
            return local
        remote = [r for r in free if not r.trusted]
        if not remote:
            return local
        return max(
            remote,
            key=lambda r: (self.dispatcher.reputation.rep(r.worker_id), r.worker_id),
        )

    def _unverifiable(self, check: Check) -> bool:
        """No worker can run this check (no trusted worker configured and every other
        known worker is a claimant, dead, gone or quarantined): skip it rather than hang."""
        if self.local_configured:
            return False
        if check.trusted_only:
            return True
        return not any(
            r.worker_id not in check.claimants() and r.state in ALIVE
            for r in self.dispatcher.registry
        )

    def dispatch(self, workers: list[WorkerRecord], capacity: Capacity) -> int:
        """Send queued checks to verifiers with free capacity; return how many were sent."""
        job = self.job
        if job is None:
            return 0
        sent = 0
        closed = False
        for check in sorted(self.checks.values(), key=lambda c: c.check_id):
            if check.attempt_id is not None:
                continue
            if not self._claims_valid(job, check):
                self._close(check)  # the judged results were replaced meanwhile
                closed = True
                continue
            verifier = self._verifier_for(check, workers, capacity)
            if verifier is None:
                if self._unverifiable(check):
                    self.stats.skipped_no_verifier += 1
                    self.dispatcher.events.emit(
                        "verification_skipped", task_ids=list(check.task_ids), reason="no_verifier"
                    )
                    self._close(check)
                    closed = True
                continue
            attempt = self.dispatcher.send(
                Assignment(check.task_ids, verifier.worker_id, check.kind)
            )
            if attempt is not None:
                check.attempt_id = attempt.attempt_id
                self.stats.attempts_sent += 1
                sent += 1
        if closed:
            self.dispatcher.on_progress()  # the job may be complete now
        return sent

    def _check_of(self, attempt: Attempt) -> Check | None:
        return next((c for c in self.checks.values() if c.attempt_id == attempt.attempt_id), None)

    def on_verify_failed(self, attempt: Attempt) -> None:
        """The verification attempt timed out, was lost, rejected or went stale: requeue."""
        check = self._check_of(attempt)
        if check is not None:
            check.attempt_id = None

    # ------------------------------------------------------------ judging

    @staticmethod
    def _output_hashes(job: Job, task_ids: list[str], output: Any) -> dict[str, str]:
        """``{task_id: hash}`` of a chunk output split per task."""
        if isinstance(output, np.ndarray) and job.grid is not None:
            row, c0, c1 = job.chunk_columns(list(task_ids))
            tiles = job.grid.split_chunk_output(output, row, c0, c1)
            return {job.tile_id(r, c): result_hash(t) for (r, c), t in tiles.items()}
        return {task_ids[0]: result_hash(output)}

    @staticmethod
    def _output_values(job: Job, task_ids: list[str], output: Any) -> dict[str, Any]:
        if isinstance(output, np.ndarray) and job.grid is not None:
            row, c0, c1 = job.chunk_columns(list(task_ids))
            tiles = job.grid.split_chunk_output(output, row, c0, c1)
            return {job.tile_id(r, c): t for (r, c), t in tiles.items()}
        return {task_ids[0]: output}

    def on_verify_result(self, attempt: Attempt, output: Any) -> None:
        """Judge a verification/tiebreak result (dispatcher hook)."""
        job = self.job
        check = self._check_of(attempt)
        if job is None or check is None:
            return
        valid = self._claims_valid(job, check)
        hashes = self._output_hashes(job, attempt.task_ids, output)
        values = self._output_values(job, attempt.task_ids, output)
        if not valid:
            self._close(check)
            return
        if self._trusted(attempt.worker_id):
            self._judge_trusted(job, check, valid, hashes, values, attempt.worker_id)
        else:
            self._judge_untrusted(job, check, valid, hashes, attempt)

    def _judge_trusted(
        self,
        job: Job,
        check: Check,
        valid: dict[str, str],
        truth: dict[str, str],
        values: dict[str, Any],
        verifier_id: str,
    ) -> None:
        wrong_workers: set[str] = set()
        for worker, claims in check.claims.items():
            if any(
                tid in truth and not same_hash(claims[tid], truth[tid])
                for tid in claims
                if tid in valid
            ):
                wrong_workers.add(worker)
        mismatched = [tid for tid, claimed in valid.items() if not same_hash(claimed, truth[tid])]
        self._close(check)
        for tid in valid:
            if tid in mismatched:
                job.replace_result(tid, values[tid], verifier_id)
            else:
                job.tasks[tid].verified = True
        honest = check.claimants() - wrong_workers
        for worker in honest:
            record = self._record(worker)
            if record is not None and not record.trusted:
                record.verified_ok += 1
                self.trusted_verified[worker] = self.trusted_verified.get(worker, 0) + 1
        self.dispatcher.events.emit(
            "verification_result",
            kind=check.kind,
            task_ids=list(check.task_ids),
            holder=check.holder,
            verifier=verifier_id,
            match=not mismatched and not wrong_workers,
            wrong_workers=sorted(wrong_workers),
            replaced=mismatched,
        )
        if mismatched or wrong_workers:
            self.stats.mismatches += 1
        else:
            self.stats.matches += 1
        for worker in sorted(wrong_workers):
            self._strike(worker, check)
        self.dispatcher.on_progress()

    def _judge_untrusted(
        self,
        job: Job,
        check: Check,
        valid: dict[str, str],
        hashes: dict[str, str],
        attempt: Attempt,
    ) -> None:
        agree = all(same_hash(hashes.get(tid), claimed) for tid, claimed in valid.items())
        if agree and self.collusion_safe(check.holder, attempt.worker_id):
            self._close(check)
            for tid in valid:
                job.tasks[tid].verified = True
            record = self._record(check.holder)
            if record is not None:
                record.verified_ok += 1
            self.stats.matches += 1
            self.dispatcher.events.emit(
                "verification_result",
                kind=check.kind,
                task_ids=list(check.task_ids),
                holder=check.holder,
                verifier=attempt.worker_id,
                match=True,
                wrong_workers=[],
                replaced=[],
            )
            self.dispatcher.on_progress()
            return
        # Disagreement (or untrustworthy agreement): the trusted worker breaks the tie.
        self.stats.tiebreaks += 1
        claims = {w: dict(c) for w, c in check.claims.items()}
        claims[attempt.worker_id] = {t: hashes[t] for t in valid if t in hashes}
        self._close(check)
        self._new_check(
            TIEBREAK,
            check.task_ids,
            check.holder,
            claims,
            check.accepted_t,
            trusted_only=True,
        )
        self.dispatcher.on_progress()

    # ---------------------------------------------------- strikes, quarantine

    def _strike(self, worker_id: str, check: Check) -> None:
        record = self._record(worker_id)
        if record is None or record.trusted:
            return
        record.wrong += 1
        record.strikes += 1
        self.dispatcher.reputation.record_failure(worker_id, "wrong_result")
        first = self.stats.first_wrong_t.get(worker_id)
        if first is None or check.accepted_t < first:
            self.stats.first_wrong_t[worker_id] = check.accepted_t
        self.dispatcher.security_log.emit(
            "wrong_result",
            worker_id=worker_id,
            task_ids=list(check.task_ids),
            strikes=record.strikes,
        )
        if record.strikes >= self.cfg.strikes_to_quarantine:
            self.quarantine(worker_id, f"{record.strikes} wrong results")

    def quarantine(self, worker_id: str, reason: str) -> list[str]:
        """Quarantine a worker and repair everything it produced that is unverified."""
        record = self._record(worker_id)
        if record is None or record.state is WorkerState.QUARANTINED:
            return []
        self.dispatcher.registry.set_state(record, WorkerState.QUARANTINED, reason)
        self.stats.quarantined_t[worker_id] = monotonic()
        job = self.job
        for attempt in list(record.inflight.values()):
            self.dispatcher.cancel(attempt, "worker quarantined")
            if job is not None:
                for tid in attempt.task_ids:
                    job.requeue(tid)
        repaired: list[str] = []
        if job is not None:
            for task in list(job.tasks.values()):
                if task.produced_by == worker_id and not task.verified and task.status in SATISFIED:
                    repaired.extend(job.invalidate(task.task_id))
            for check in list(self.checks.values()):
                if check.attempt_id is None and not self._claims_valid(job, check):
                    self._close(check)
        self.stats.repaired_tasks += len(repaired)
        self.dispatcher.events.emit(
            "worker_quarantined",
            worker_id=worker_id,
            name=record.name,
            strikes=record.strikes,
            reason=reason,
            repaired_tasks=sorted(set(repaired)),
        )
        self.dispatcher.security_log.emit(
            "worker_quarantined", worker_id=worker_id, strikes=record.strikes
        )
        self.dispatcher.on_progress()
        return repaired
