"""Failure detection and recovery logic (spec Phase 5), with fake clocks and channels."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from mdc.common import protocol as P
from mdc.common.config import load_config
from mdc.common.events import EventLogger
from mdc.common.ratelimit import SlidingCounter
from mdc.coordinator.dispatcher import Dispatcher
from mdc.coordinator.estimators import Estimators
from mdc.coordinator.failure_detector import FailureDetector
from mdc.coordinator.job import TaskStatus
from mdc.coordinator.job_builder import build_image_job_from_array
from mdc.coordinator.policies.base import Assignment
from mdc.coordinator.policies.round_robin import RoundRobinPolicy
from mdc.coordinator.registry import Registry, WorkerRecord, WorkerState
from mdc.coordinator.reputation import Reputation
from mdc.coordinator.scheduler import Scheduler
from mdc.coordinator.security_log import SecurityLog
from mdc.tasks.image_pipeline import pipeline_valid

CFG = load_config(
    overrides=[
        "liveness.suspect_after_s=3",
        "liveness.dead_after_s=6",
        "liveness.task_timeout_min_s=1",
        "liveness.task_timeout_factor=2",
        "liveness.task_timeout_slack_s=0.5",
        "liveness.max_attempts_per_task=3",
        "workers.prefetch=0",
        # Failure handling in isolation: results stay DONE instead of VERIFYING
        # (verification has its own tests in test_verifier.py, Phase 10).
        "verification.enabled=false",
    ]
)


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class FakeChannel:
    """Stands in for a server connection."""

    def __init__(self, worker_id: str, session_id: str = "s1") -> None:
        self.worker_id = worker_id
        self.session_id = session_id
        self.unsolicited = SlidingCounter(60.0)
        self.sent: list[P.Message] = []
        self.closed = False

    async def send(self, msg: P.Message) -> int:
        self.sent.append(msg)
        return 1

    def close(self) -> None:
        self.closed = True


class Harness:
    """Registry + dispatcher + detector + scheduler around a small image job."""

    def __init__(self, n_workers: int = 2, local: bool = False) -> None:
        self.events = EventLogger()
        self.registry = Registry(self.events)
        self.channels: dict[str, FakeChannel] = {}
        self.closed: list[str] = []
        estimators = Estimators.from_config(CFG)
        self.reputation = estimators.reputation
        self.dispatcher = Dispatcher(
            CFG,
            self.registry,
            self.events,
            SecurityLog(None),
            self.channels.get,
            lambda: None,
            estimators=estimators,
        )
        self.clock = Clock()
        self.detector = FailureDetector(
            CFG,
            self.registry,
            lambda: list(self.channels.values()),
            dispatcher=self.dispatcher,
            close_connection=self.closed.append,
            clock=self.clock,
        )
        for i in range(n_workers):
            self.add_worker(f"w{i}")
        if local:
            self.add_worker("local", trusted=True)
        self.image = np.random.default_rng(0).integers(0, 256, size=(20, 30), dtype=np.uint8)
        self.job = build_image_job_from_array(self.image, 10, 1, 1, 64, job_id="j")
        self.dispatcher.start_job(self.job)

    def add_worker(self, wid: str, trusted: bool = False) -> WorkerRecord:
        record = self.registry.upsert(wid, wid)
        record.trusted = trusted
        record.last_pong_t = self.clock.now if hasattr(self, "clock") else 100.0
        self.registry.set_state(record, WorkerState.PROBING)
        self.registry.set_state(record, WorkerState.ACTIVE)
        self.channels[wid] = FakeChannel(wid)
        return record

    def types(self) -> list[str]:
        return [e["type"] for e in self.events.recent()]


def result_msg(attempt_id: str, task_ids: list[str], harness: Harness) -> P.Message:
    task = harness.job.tasks[task_ids[0]]
    rect = harness.job.grid.chunk_rect(task.row, task.col, task.col)  # type: ignore[union-attr]
    padded = harness.job.task_input(task_ids)[2]
    out = pipeline_valid(padded, 1, 1)  # type: ignore[arg-type]
    assert out.shape == rect.shape
    fields, payload = P.pack_array(out)
    return P.make_message(
        P.TASK_RESULT,
        payload,
        task_id=task_ids[0],
        attempt_id=attempt_id,
        status="ok",
        error=None,
        compute_ms=1.0,
        queue_ms=0.0,
        t_recv_w=0.0,
        t_send_w=0.0,
        **fields,
    )


# --------------------------------------------------------------- liveness


async def test_suspect_then_back_to_active_on_pong() -> None:
    h = Harness(1)
    rec = h.registry.get("w0")
    h.clock.now += 3.5
    assert h.detector.check() == []
    assert rec.state is WorkerState.SUSPECT
    ping = h.detector.next_ping()
    h.detector.on_pong(
        "w0", {"seq": ping.header["seq"], "t_c": ping.header["t_c"], "battery": {"pct": 50.0}}
    )
    assert rec.state is WorkerState.ACTIVE
    assert rec.battery["pct"] == 50.0


async def test_dead_after_silence_reassigns_and_closes() -> None:
    h = Harness(1)
    attempt = h.dispatcher.send(Assignment(("s1_r00_c00",), "w0"))
    assert attempt is not None and h.job.tasks["s1_r00_c00"].status is TaskStatus.RUNNING
    h.clock.now += 3.1
    h.detector.check()
    assert attempt.status == "inflight"  # SUSPECT does not reassign yet
    h.clock.now += 3.0  # 6.1 s silent
    assert h.detector.check() == ["w0"]
    rec = h.registry.get("w0")
    assert rec.state is WorkerState.DEAD and h.closed == ["w0"]
    assert attempt.status == "lost" and not rec.inflight
    assert h.job.tasks["s1_r00_c00"].status is TaskStatus.READY
    assert h.reputation.snapshot("w0")["failures"]["disconnect"] == 1
    assert "attempt_lost" in h.types()
    await h.dispatcher.drain()


async def test_detector_ignores_non_active_states_and_boundaries() -> None:
    h = Harness(2)
    quiet, other = h.registry.get("w0"), h.registry.get("w1")
    h.registry.set_state(other, WorkerState.LEFT)
    h.clock.now += 3.0  # exactly the threshold: not yet suspect
    h.detector.check()
    assert quiet.state is WorkerState.ACTIVE
    h.clock.now += 100
    h.detector.check()
    assert other.state is WorkerState.LEFT  # not touched


def test_pong_rtt_sanity_and_unknown_worker() -> None:
    """RTT is timed on our own clock; falsified echoes and unknown seqs/workers give no sample."""
    h = Harness(1)
    ping = h.detector.next_ping()
    seq, t_c = ping.header["seq"], ping.header["t_c"]
    h.clock.now += 0.2
    assert h.detector.on_pong("w0", {"seq": seq, "t_c": t_c, "battery": {}}) == pytest.approx(0.2)
    # A worker cannot fake a low RTT: a falsified echo or an unknown seq gives no sample.
    assert h.detector.on_pong("w0", {"seq": seq, "t_c": t_c + 0.19, "battery": {}}) is None
    assert h.detector.on_pong("w0", {"seq": 9999, "t_c": t_c, "battery": {}}) is None
    assert h.detector.on_pong("nobody", {"seq": seq, "t_c": t_c, "battery": {}}) is None
    # Even with a correct echo, liveness is updated.
    assert h.registry.get("w0").last_pong_t == h.clock.now


def test_ping_history_is_bounded() -> None:
    h = Harness(1)
    first = h.detector.next_ping()
    for _ in range(200):
        h.detector.next_ping()
    assert len(h.detector._ping_sent) == 64
    assert (
        h.detector.on_pong(
            "w0", {"seq": first.header["seq"], "t_c": first.header["t_c"], "battery": {}}
        )
        is None
    )


async def test_detector_loops_start_and_stop() -> None:
    h = Harness(1)
    h.detector.clock = asyncio.get_running_loop().time
    fast = load_config(overrides=["liveness.ping_interval_s=0.05"])
    h.detector.cfg = fast
    h.detector.start()
    await asyncio.sleep(0.3)
    await h.detector.stop()
    assert any(m.type == P.PING for m in h.channels["w0"].sent)


# --------------------------------------------------------------- timeouts


async def test_timeout_sweep_requeues_and_cancels() -> None:
    h = Harness(1)
    attempt = h.dispatcher.send(Assignment(("s1_r00_c00",), "w0"))
    assert attempt is not None
    assert attempt.timeout_s == pytest.approx(max(1.0, 2 * attempt.predicted_s + 0.5))
    assert h.dispatcher.sweep_timeouts(attempt.sent_t + attempt.timeout_s - 0.01) == 0
    assert h.dispatcher.sweep_timeouts(attempt.sent_t + attempt.timeout_s + 0.01) == 1
    assert attempt.status == "timeout" and attempt.timed_out
    assert h.job.tasks["s1_r00_c00"].status is TaskStatus.READY
    assert h.registry.get("w0").failed == 1
    assert h.reputation.snapshot("w0")["failures"]["timeout"] == 1
    await h.dispatcher.drain()
    assert [m.type for m in h.channels["w0"].sent] == [P.TASK_ASSIGN, P.TASK_CANCEL]
    assert "attempt_timeout" in h.types()


async def test_timeout_waits_for_other_inflight_copy() -> None:
    h = Harness(2)
    first = h.dispatcher.send(Assignment(("s1_r00_c00",), "w0"))
    second = h.dispatcher.send(Assignment(("s1_r00_c00",), "w1", purpose="replica"))
    assert first is not None and second is not None
    second.sent_t = first.sent_t + 100  # the replica is young
    h.dispatcher.sweep_timeouts(first.sent_t + first.timeout_s + 0.1)
    assert first.status == "timeout" and second.status == "inflight"
    assert h.job.tasks["s1_r00_c00"].status is TaskStatus.RUNNING  # still waiting for w1
    await h.dispatcher.drain()


async def test_late_result_accepted_when_task_still_needed() -> None:
    h = Harness(1)
    attempt = h.dispatcher.send(Assignment(("s1_r00_c00",), "w0"))
    h.dispatcher.sweep_timeouts(attempt.sent_t + attempt.timeout_s + 1)
    h.dispatcher.on_result(h.channels["w0"], result_msg(attempt.attempt_id, ["s1_r00_c00"], h))
    assert attempt.status == "late_accepted" and attempt.timed_out
    assert h.job.tasks["s1_r00_c00"].status is TaskStatus.DONE
    late = [e for e in h.events.recent() if e["type"] == "attempt_late"]
    assert late[-1]["accepted"] is True
    await h.dispatcher.drain()


async def test_late_result_ignored_when_task_done_meanwhile() -> None:
    h = Harness(2)
    slow = h.dispatcher.send(Assignment(("s1_r00_c00",), "w0"))
    h.dispatcher.sweep_timeouts(slow.sent_t + slow.timeout_s + 1)
    retry = h.dispatcher.send(Assignment(("s1_r00_c00",), "w1"))
    h.dispatcher.on_result(h.channels["w1"], result_msg(retry.attempt_id, ["s1_r00_c00"], h))
    done_by = h.job.tasks["s1_r00_c00"].produced_by
    h.dispatcher.on_result(h.channels["w0"], result_msg(slow.attempt_id, ["s1_r00_c00"], h))
    assert slow.status == "timeout"  # unchanged
    assert h.job.tasks["s1_r00_c00"].produced_by == done_by == "w1"  # no duplicate DONE
    assert [e["accepted"] for e in h.events.recent() if e["type"] == "attempt_late"] == [False]
    await h.dispatcher.drain()


async def test_late_accepted_result_cancels_the_retry() -> None:
    h = Harness(2)
    slow = h.dispatcher.send(Assignment(("s1_r00_c00",), "w0"))
    h.dispatcher.sweep_timeouts(slow.sent_t + slow.timeout_s + 1)
    retry = h.dispatcher.send(Assignment(("s1_r00_c00",), "w1"))
    h.dispatcher.on_result(h.channels["w0"], result_msg(slow.attempt_id, ["s1_r00_c00"], h))
    assert slow.status == "late_accepted" and retry.status == "cancelled"
    await h.dispatcher.drain()
    assert h.channels["w1"].sent[-1].type == P.TASK_CANCEL


async def test_late_error_report_changes_nothing() -> None:
    h = Harness(1)
    attempt = h.dispatcher.send(Assignment(("s1_r00_c00",), "w0"))
    h.dispatcher.sweep_timeouts(attempt.sent_t + attempt.timeout_s + 1)
    err = P.make_message(
        P.TASK_RESULT,
        task_id="s1_r00_c00",
        attempt_id=attempt.attempt_id,
        status="error",
        error="boom",
        compute_ms=0.0,
        queue_ms=0.0,
        t_recv_w=0.0,
        t_send_w=0.0,
    )
    h.dispatcher.on_result(h.channels["w0"], err)
    assert attempt.status == "timeout" and h.registry.get("w0").failed == 1
    await h.dispatcher.drain()


# --------------------------------------------------- fallback and backoff


async def test_exhausted_task_falls_back_to_local() -> None:
    h = Harness(1, local=True)
    sched = Scheduler(
        CFG,
        h.registry,
        h.dispatcher,
        RoundRobinPolicy(load_config(overrides=["scheduler.allow_local=false"])),
    )
    sched.start_job(h.job)
    for _ in range(3):  # three failed attempts on the remote worker
        a = h.dispatcher.send(Assignment(("s1_r00_c00",), "w0"))
        h.dispatcher.sweep_timeouts(a.sent_t + a.timeout_s + 1)
    h.registry.get("w0").cooldown_until = 0.0
    sched.schedule_once()
    task = h.job.tasks["s1_r00_c00"]
    last = h.job.attempts[task.attempts[-1]]
    assert last.worker_id == "local"
    fallback = [e for e in h.events.recent() if e["type"] == "task_fallback_local"]
    assert fallback and fallback[0]["task_id"] == "s1_r00_c00" and fallback[0]["attempts"] == 3
    # The remote worker got other tiles, never the reserved one.
    remote = [a for a in h.job.attempts.values() if a.worker_id == "w0" and a.status == "inflight"]
    assert all("s1_r00_c00" not in a.task_ids for a in remote)
    await h.dispatcher.drain()


async def test_no_fallback_without_local_worker(caplog: pytest.LogCaptureFixture) -> None:
    cfg = load_config(
        overrides=[
            "liveness.max_attempts_per_task=1",
            "workers.local_worker=false",
            "workers.prefetch=0",
        ]
    )
    h = Harness(1)
    sched = Scheduler(cfg, h.registry, h.dispatcher, RoundRobinPolicy(cfg))
    sched.start_job(h.job)
    a = h.dispatcher.send(Assignment(("s1_r00_c00",), "w0"))
    h.dispatcher.sweep_timeouts(a.sent_t + a.timeout_s + 1)
    sched.schedule_once()
    assert "no local worker" in caplog.text
    assert h.job.tasks["s1_r00_c00"].status is TaskStatus.RUNNING  # retried remotely
    await h.dispatcher.drain()


async def test_error_backoff_makes_worker_temporarily_ineligible() -> None:
    h = Harness(1)
    sched = Scheduler(CFG, h.registry, h.dispatcher, RoundRobinPolicy(CFG))
    sched.start_job(h.job)
    rec = h.registry.get("w0")
    a = h.dispatcher.send(Assignment(("s1_r00_c00",), "w0"))
    err = P.make_message(
        P.TASK_RESULT,
        task_id="s1_r00_c00",
        attempt_id=a.attempt_id,
        status="error",
        error="battery_low",
        compute_ms=0.0,
        queue_ms=0.0,
        t_recv_w=0.0,
        t_send_w=0.0,
    )
    h.dispatcher.on_result(h.channels["w0"], err)
    assert rec.consecutive_errors == 1 and rec.cooldown_until > 0
    assert sched.eligible() == []  # backing off
    rec.cooldown_until = 0.0
    assert sched.eligible() == [rec]
    await h.dispatcher.drain()


def test_cooldown_grows_and_resets() -> None:
    rec = WorkerRecord("w", "w")
    delays = [rec.note_error(0.0) for _ in range(8)]
    assert delays[:4] == [0.25, 0.5, 1.0, 2.0]
    assert delays[-1] == 10.0  # capped
    for _ in range(200):
        rec.note_error(0.0)  # no overflow with many errors
    rec.note_success()
    assert rec.consecutive_errors == 0 and rec.cooldown_until == 0.0


def test_reputation_stub() -> None:
    rep = Reputation()
    rep.record_success("w")
    rep.record_failure("w", "timeout")
    snap = rep.snapshot("w")  # Phase 7: the snapshot gained rep, lambda, s, f, attempts
    assert snap["successes"] == 1
    assert snap["failures"] == {"timeout": 1, "disconnect": 0, "error": 0, "wrong_result": 0}
    assert 0 < snap["rep"] < 1
    assert rep.snapshot("new")["successes"] == 0
    with pytest.raises(ValueError):
        rep.record_failure("w", "sneezed")
