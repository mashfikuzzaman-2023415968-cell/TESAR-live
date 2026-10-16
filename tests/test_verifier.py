"""Result verification, strikes, quarantine and taint repair (spec Phase 10, unit level).

A real Dispatcher and Job with fake channels: results are injected as
TASK_RESULT messages, verification attempts are dispatched by the Verifier and
answered by hand — correctly or with a plausible corruption.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from mdc.common import protocol as P
from mdc.common.config import Config, load_config
from mdc.common.events import EventLogger
from mdc.coordinator.dispatcher import Dispatcher
from mdc.coordinator.job import Attempt, TaskStatus
from mdc.coordinator.job_builder import build_image_job_from_array
from mdc.coordinator.policies.base import Assignment
from mdc.coordinator.registry import Registry, WorkerRecord, WorkerState
from mdc.coordinator.scheduler import Scheduler
from mdc.coordinator.security_log import SecurityLog
from mdc.coordinator.verifier import (
    COLLUSION_MIN_TRUSTED_VERIFIED,
    TIEBREAK,
    VERIFY,
    Check,
    same_hash,
)
from mdc.tasks.base import execute
from tests.test_failure_detection import FakeChannel

BASE = [
    "workers.prefetch=0",
    "verification.probation_tasks=1",
    "verification.base_prob=0",
    "verification.low_rep_boost=0",
]


class VH:
    """Verification harness: workers w0, w1 (+ trusted local), a 2 x 3 tile image job."""

    def __init__(self, *overrides: str, remote: tuple[str, ...] = ("w0", "w1"), local: bool = True):
        self.cfg: Config = load_config(overrides=[*BASE, *overrides])
        self.events = EventLogger()
        self.registry = Registry(self.events)
        self.channels: dict[str, FakeChannel] = {}
        self.security: list[tuple[str, dict[str, Any]]] = []
        sec = SecurityLog(None)
        real_emit = sec.emit

        def spy(event_type: str, **fields: Any) -> Any:
            self.security.append((event_type, fields))
            return real_emit(event_type, **fields)  # still rejects unknown event types

        sec.emit = spy  # type: ignore[method-assign]
        self.dispatcher = Dispatcher(
            self.cfg, self.registry, self.events, sec, self.channels.get, lambda: None
        )
        for wid in remote:
            self.add(wid)
        if local:
            self.add("local", trusted=True)
        image = np.random.default_rng(1).integers(0, 256, size=(20, 30), dtype=np.uint8)
        self.job = build_image_job_from_array(image, 10, 1, 1, 64, job_id="j")
        self.dispatcher.start_job(self.job)
        self.v = self.dispatcher.verifier
        self.sched = Scheduler(self.cfg, self.registry, self.dispatcher, _Idle())

    def add(self, wid: str, trusted: bool = False) -> WorkerRecord:
        rec = self.registry.upsert(wid, wid)
        rec.trusted = trusted
        self.registry.set_state(rec, WorkerState.PROBING)
        self.registry.set_state(rec, WorkerState.ACTIVE)
        self.channels[wid] = FakeChannel(wid)
        return rec

    def rec(self, wid: str) -> WorkerRecord:
        record = self.registry.get(wid)
        assert record is not None
        return record

    def truth(self, task_ids: list[str]) -> Any:
        kind, params, data = self.job.task_input(task_ids)
        out = execute(kind, params, data)
        return out.array if out.array is not None else out.json_result

    def reply(self, attempt: Attempt, wrong: bool = False, delta: int = 1) -> None:
        """Answer ``attempt`` from its worker's channel (optionally corrupted)."""
        value = self.truth(attempt.task_ids)
        if isinstance(value, np.ndarray):
            if wrong:
                value = ((value.astype(np.int32) + delta) % 256).astype(np.uint8)
            fields, payload = P.pack_array(value)
        else:
            if wrong:
                value = {**value, "edge_pixels": value["edge_pixels"] + delta}
            fields, payload = {"json_result": value}, b""
        msg = P.make_message(
            P.TASK_RESULT,
            payload,
            task_id=attempt.task_ids[0],
            attempt_id=attempt.attempt_id,
            status="ok",
            error=None,
            compute_ms=1.0,
            queue_ms=0.0,
            t_send_w=0.0,
            **fields,
        )
        self.dispatcher.on_result(self.channels[attempt.worker_id], msg)

    def produce(self, wid: str, ids: tuple[str, ...], wrong: bool = False) -> Attempt:
        attempt = self.dispatcher.send(Assignment(ids, wid))
        assert attempt is not None
        self.reply(attempt, wrong)
        return attempt

    def dispatch(self) -> list[Attempt]:
        """Send queued checks; return the new verification attempts."""
        before = set(self.job.attempts)
        self.v.dispatch(self.registry.active(), self.sched.capacity)
        return [a for k, a in self.job.attempts.items() if k not in before]

    def types(self) -> list[str]:
        return [e["type"] for e in self.events.recent()]


class _Idle:
    name = "idle"

    def assign(self, ctx: Any) -> list[Assignment]:
        return []


T00, T01, T02 = "s1_r00_c00", "s1_r00_c01", "s1_r00_c02"


# ------------------------------------------------------------- deciding


async def test_probation_forces_verification_then_sampling() -> None:
    h = VH("verification.probation_tasks=2")
    h.produce("w0", (T00,))
    h.produce("w0", (T01,))
    assert h.v.outstanding() == 2  # results 1 and 2: probation
    assert h.job.tasks[T00].status is TaskStatus.VERIFYING
    h.produce("w0", (T02,))
    assert h.v.outstanding() == 2  # base_prob 0, boost 0: not sampled
    assert h.job.tasks[T02].status is TaskStatus.DONE
    await h.dispatcher.drain()


def test_verification_probability_formula() -> None:
    h = VH(
        "verification.probation_tasks=0",
        "verification.base_prob=0.1",
        "verification.low_rep_boost=0.5",
    )
    rec = h.rec("w0")
    rec.completed = 10
    rep = h.dispatcher.reputation.rep("w0")  # 0.5 for a new worker
    p = 0.1 + (1 - rep) * 0.5
    h.v.rng.random = lambda: p - 1e-9  # type: ignore[method-assign]
    assert h.v.should_verify(rec)
    h.v.rng.random = lambda: p + 1e-9  # type: ignore[method-assign]
    assert not h.v.should_verify(rec)
    capped = VH(
        "verification.probation_tasks=0",
        "verification.base_prob=0.9",
        "verification.low_rep_boost=5",
    )
    capped.rec("w0").completed = 10
    capped.v.rng.random = lambda: 0.999999  # type: ignore[method-assign]
    assert capped.v.should_verify(capped.rec("w0"))  # p = min(1, ...) = 1
    assert not h.v.should_verify(h.rec("local"))  # trusted: never
    off = VH("verification.enabled=false")
    assert not off.v.should_verify(off.rec("w0"))


async def test_trusted_results_are_never_verified() -> None:
    h = VH()
    h.produce("local", (T00,))
    assert h.v.outstanding() == 0
    assert h.job.tasks[T00].verified
    await h.dispatcher.drain()


# -------------------------------------------------------------- judging


async def test_match_marks_verified_and_counts() -> None:
    h = VH()
    h.produce("w0", (T00, T01))
    (check_attempt,) = h.dispatch()
    assert check_attempt.worker_id == "local" and check_attempt.purpose == VERIFY
    assert check_attempt.task_ids == [T00, T01]
    h.reply(check_attempt)
    assert h.v.outstanding() == 0
    for tid in (T00, T01):
        assert h.job.tasks[tid].verified and h.job.tasks[tid].status is TaskStatus.DONE
    assert h.rec("w0").verified_ok == 1 and h.rec("w0").strikes == 0
    assert h.v.stats.matches == 1
    assert "verification_result" in h.types()
    await h.dispatcher.drain()


async def test_mismatch_with_trusted_verifier_strikes_swaps_and_invalidates() -> None:
    h = VH()
    for tid in (T00, T01, T02):
        h.produce("w1" if tid != T00 else "w0", (tid,), wrong=tid == T00)
    stats = h.job.ready_tasks()[0]
    assert stats.task_id == "s2_r00"
    h.produce("w1", ("s2_r00",))  # row statistics computed from the wrong tile
    assert h.job.tasks["s2_r00"].status in {TaskStatus.DONE, TaskStatus.VERIFYING}
    rep_before = h.dispatcher.reputation.rep("w0")
    check = next(a for a in h.dispatch() if a.task_ids == [T00])
    h.reply(check)
    task = h.job.tasks[T00]
    assert task.verified and task.produced_by == "local"
    assert np.array_equal(task.result, h.truth([T00]))  # the verifier's result replaced it
    rec = h.rec("w0")
    assert (rec.strikes, rec.wrong) == (1, 1)
    assert h.dispatcher.reputation.snapshot("w0")["failures"]["wrong_result"] == 1
    assert h.dispatcher.reputation.rep("w0") < rep_before
    assert h.job.tasks["s2_r00"].status is TaskStatus.READY  # taint: recomputed
    assert h.job.tasks["s2_r00"].result is None
    assert ("wrong_result", {"worker_id": "w0", "task_ids": [T00], "strikes": 1}) in h.security
    assert h.v.stats.mismatches == 1 and h.v.stats.first_wrong_t["w0"] > 0
    await h.dispatcher.drain()


async def test_quarantine_after_two_strikes_repairs_only_unverified_work() -> None:
    h = VH("verification.probation_tasks=3")
    good = h.produce("w0", (T00,))  # honest, will be verified
    (check,) = h.dispatch()
    h.reply(check)
    assert h.job.tasks[T00].verified
    h.produce("w0", (T01,), wrong=True)
    h.produce("w0", (T02,), wrong=True)
    h.rec("w0").completed = 99  # later results are not sampled
    sneaky = h.produce("w0", ("s1_r01_c00",), wrong=True)  # never verified directly
    h.rec("w1").completed = 99  # past probation: not sampled either
    h.produce("w1", ("s1_r01_c01",))
    inflight = h.dispatcher.send(Assignment(("s1_r01_c02",), "w0"))
    assert inflight is not None
    judged = []
    while checks := h.dispatch():  # one at a time: the local worker has one slot
        for attempt in checks:
            judged.append(tuple(attempt.task_ids))
            h.reply(attempt)
    assert judged == [(T01,), (T02,)]
    rec = h.rec("w0")
    assert rec.state is WorkerState.QUARANTINED and rec.strikes == 2
    assert inflight.status == "cancelled"
    assert h.job.tasks["s1_r01_c02"].status is TaskStatus.READY  # requeued
    # Retroactive repair: its unverified tile is re-run; verified and others' work kept.
    assert h.job.tasks["s1_r01_c00"].status is TaskStatus.INVALIDATED
    assert h.job.tasks[T00].produced_by == "w0" and h.job.tasks[T00].verified
    assert h.job.tasks["s1_r01_c01"].produced_by == "w1"
    event = next(e for e in h.events.recent() if e["type"] == "worker_quarantined")
    assert event["repaired_tasks"] == ["s1_r01_c00"]
    assert "w0" in h.v.stats.detection_latency_s()
    assert sneaky.worker_id == good.worker_id
    # A quarantined worker gets nothing and its results are ignored.
    assert rec not in h.registry.active()
    h.reply(sneaky)
    assert any(t == "result_from_quarantined" for t, _ in h.security)
    assert h.v.quarantine("w0", "again") == []  # idempotent
    await h.dispatcher.drain()


async def test_untrusted_verifier_mismatch_goes_to_trusted_tiebreak() -> None:
    h = VH("verification.prefer_local_verifier=false")
    h.dispatcher.reputation.record_success("w1")  # w1 is the better-reputed verifier
    h.produce("w0", (T00,))  # honest producer
    (check,) = h.dispatch()
    assert check.worker_id == "w1"  # highest reputation, not the producer
    h.reply(check, wrong=True)  # the *verifier* lies
    (tie,) = h.dispatch()
    assert tie.worker_id == "local" and tie.purpose == TIEBREAK
    h.reply(tie)
    assert h.rec("w1").strikes == 1 and h.rec("w0").strikes == 0
    assert h.job.tasks[T00].verified and h.job.tasks[T00].produced_by == "w0"
    assert h.v.stats.tiebreaks == 1
    await h.dispatcher.drain()


async def test_untrusted_verifier_agreement_counts_when_collusion_safe() -> None:
    h = VH("verification.prefer_local_verifier=false", "workers.local_worker=false", local=False)
    w1 = h.rec("w1")
    for _ in range(30):
        h.dispatcher.reputation.record_success("w1")
    h.v.trusted_verified["w1"] = COLLUSION_MIN_TRUSTED_VERIFIED
    h.produce("w0", (T00,))
    (check,) = h.dispatch()
    assert check.worker_id == w1.worker_id
    h.reply(check)
    assert h.job.tasks[T00].verified and h.v.outstanding() == 0
    await h.dispatcher.drain()


async def test_collusion_rule_identical_wrong_results_are_both_caught() -> None:
    h = VH("verification.probation_tasks=0")
    for wid in ("w0", "w1"):
        h.rec(wid).completed = 99
    primary = h.dispatcher.send(Assignment((T00,), "w0"))
    replica = h.dispatcher.send(Assignment((T00,), "w1", "replica"))
    assert primary is not None and replica is not None
    h.reply(primary, wrong=True)  # accepted; the replica is cancelled ...
    assert replica.status == "cancelled"
    h.reply(replica, wrong=True)  # ... but finishes anyway with the *same* wrong result
    assert not h.job.tasks[T00].verified  # two low-reputation agreeing workers prove nothing
    (check,) = h.dispatch()
    assert check.worker_id == "local"
    h.reply(check)
    assert h.rec("w0").strikes == 1 and h.rec("w1").strikes == 1
    assert np.array_equal(h.job.tasks[T00].result, h.truth([T00]))
    await h.dispatcher.drain()


async def test_replica_match_from_trusted_reputed_worker_is_a_free_verification() -> None:
    h = VH("verification.probation_tasks=0")
    for wid in ("w0", "w1"):
        h.rec(wid).completed = 99
    for _ in range(30):
        h.dispatcher.reputation.record_success("w1")
    h.v.trusted_verified["w1"] = COLLUSION_MIN_TRUSTED_VERIFIED
    primary = h.dispatcher.send(Assignment((T00,), "w0"))
    replica = h.dispatcher.send(Assignment((T00,), "w1", "replica"))
    assert primary is not None and replica is not None
    h.reply(primary)
    h.reply(replica)
    assert h.job.tasks[T00].verified and h.v.stats.free_replica_matches == 1
    assert h.v.outstanding() == 0
    await h.dispatcher.drain()


async def test_replica_disagreement_forces_a_trusted_check() -> None:
    h = VH("verification.probation_tasks=0")
    for wid in ("w0", "w1"):
        h.rec(wid).completed = 99
    primary = h.dispatcher.send(Assignment((T00,), "w0"))
    replica = h.dispatcher.send(Assignment((T00,), "w1", "replica"))
    assert primary is not None and replica is not None
    h.reply(primary)
    h.reply(replica, wrong=True)
    (check,) = h.dispatch()
    assert check.worker_id == "local"
    h.reply(check)
    assert h.rec("w1").strikes == 1 and h.rec("w0").strikes == 0
    await h.dispatcher.drain()


# --------------------------------------------------- robustness of checks


async def test_failed_verification_attempt_is_requeued() -> None:
    h = VH()
    h.produce("w0", (T00,))
    (check,) = h.dispatch()
    h.dispatcher.sweep_timeouts(check.sent_t + check.timeout_s + 1)  # type: ignore[operator]
    assert h.v.outstanding() == 1
    (again,) = h.dispatch()
    assert again.attempt_id != check.attempt_id
    h.reply(again)
    assert h.v.outstanding() == 0 and h.job.tasks[T00].verified
    await h.dispatcher.drain()


async def test_stale_check_is_dropped_when_the_result_changes() -> None:
    h = VH()
    h.produce("w0", (T00,))
    (check,) = h.dispatch()
    h.job.invalidate(T00)  # e.g. repaired for another reason meanwhile
    h.reply(check)  # stale epoch: the dispatcher drops it, the check is requeued ...
    assert h.v.outstanding() == 1
    assert h.dispatch() == []  # ... and closed at the next dispatch
    assert h.v.outstanding() == 0
    h.produce("w1", (T01,))
    queued = h.v._queued_check_for([T01])
    assert queued is not None
    h.job.invalidate(T01)
    h.v.on_verify_result(Attempt("ghost", [T01], "local"), h.truth([T01]))  # unknown attempt
    queued.attempt_id = "x"
    h.v.on_verify_result(Attempt("x", [T01], "local", purpose=VERIFY), h.truth([T01]))
    assert h.v.outstanding() == 0  # claims no longer valid: closed without judging
    await h.dispatcher.drain()


async def test_checks_wait_for_the_busy_local_worker() -> None:
    h = VH()
    h.produce("w0", (T00,))
    busy = h.dispatcher.send(Assignment((T01,), "local"))
    assert busy is not None
    assert h.dispatch() == []  # local busy (prefetch 0): wait rather than use w1
    assert h.v.outstanding() == 1
    await h.dispatcher.drain()


async def test_unverifiable_checks_are_skipped_not_hung() -> None:
    h = VH("workers.local_worker=false", remote=("w0", "w1"), local=False)
    h.produce("w0", (T00,))
    h.registry.set_state(h.rec("w1"), WorkerState.DEAD)
    assert h.dispatch() == []
    assert h.v.outstanding() == 0 and h.v.stats.skipped_no_verifier == 1
    assert "verification_skipped" in h.types()
    h2 = VH("workers.local_worker=false", local=False)
    h2.produce("w0", (T00,))
    check = h2.v._queued_check_for([T00])
    assert check is not None
    check.trusted_only = True  # needs the trusted worker, and there is none
    assert h2.dispatch() == [] and h2.v.outstanding() == 0
    await h.dispatcher.drain()
    await h2.dispatcher.drain()


async def test_row_stats_json_results_are_verified_too() -> None:
    h = VH()
    for tid in (T00, T01, T02):
        h.produce("local", (tid,))
    h.produce("w0", ("s2_r00",), wrong=True)
    (check,) = h.dispatch()
    h.reply(check)
    assert h.rec("w0").strikes == 1
    assert h.job.tasks["s2_r00"].result == h.truth(["s2_r00"])
    await h.dispatcher.drain()


def test_same_hash_and_check_helpers() -> None:
    assert same_hash("ab", "ab") and not same_hash("ab", "ac")
    assert not same_hash(None, "ab") and not same_hash("ab", None)
    check = Check(1, VERIFY, (T00,), "w0", {"w0": {T00: "x"}, "w1": {T00: "x"}}, 0.0)
    assert check.claimants() == {"w0", "w1"}


@pytest.mark.parametrize("disabled", [True, False])
async def test_disabled_verification_creates_no_checks(disabled: bool) -> None:
    h = VH(f"verification.enabled={'false' if disabled else 'true'}")
    h.produce("w0", (T00,))
    assert h.v.outstanding() == (0 if disabled else 1)
    await h.dispatcher.drain()


async def test_local_worker_takes_over_when_no_remote_worker_is_left() -> None:
    """Every remote worker quarantined + allow_local=false must not hang the job."""
    from mdc.coordinator.policies.round_robin import RoundRobinPolicy  # noqa: PLC0415

    h = VH("scheduler.allow_local=false")
    sched = Scheduler(h.cfg, h.registry, h.dispatcher, RoundRobinPolicy(h.cfg))
    sched.start_job(h.job)
    for wid in ("w0", "w1"):
        h.v.quarantine(wid, "test")
    assert sched.schedule_once() == 1  # local: one slot, prefetch 0
    assert {a.worker_id for a in h.job.attempts.values()} == {"local"}
    fallback = [e for e in h.events.recent() if e["type"] == "task_fallback_local"]
    assert fallback and fallback[0]["reason"] == "no_remote_workers"
    await h.dispatcher.drain()
