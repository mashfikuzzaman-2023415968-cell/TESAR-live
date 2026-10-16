"""Unit tests for coordinator components that need no network."""

from __future__ import annotations

import asyncio
import socket
from typing import Any

import numpy as np
import pytest

from mdc.common.config import load_config
from mdc.common.events import EventLogger
from mdc.coordinator import beacon as B
from mdc.coordinator.dispatcher import Dispatcher
from mdc.coordinator.job import TaskStatus
from mdc.coordinator.job_builder import build_image_job_from_array, build_synthetic_job
from mdc.coordinator.metrics import RunTiming, build_report, local_estimate_s, wasted_work_s
from mdc.coordinator.policies import UnknownPolicyError, make_policy
from mdc.coordinator.policies.base import Assignment, SchedulingContext, WorkerView
from mdc.coordinator.policies.local_only import LocalOnlyPolicy
from mdc.coordinator.policies.round_robin import RoundRobinPolicy
from mdc.coordinator.registry import Registry, WorkerRecord, WorkerState
from mdc.coordinator.scheduler import Scheduler
from mdc.coordinator.security_log import SecurityLog
from mdc.worker.discovery_listener import parse_beacon


def image_job(rows: int = 2, cols: int = 4):  # type: ignore[no-untyped-def]
    img = np.zeros((rows * 10, cols * 10), dtype=np.uint8)
    return build_image_job_from_array(img, 10, 1, 1, 64, job_id="j")


def view(wid: str, free: int = 1, trusted: bool = False, **kw: Any) -> WorkerView:
    return WorkerView(wid, wid, trusted, "laptop", slots=1, free=free, **kw)


def ctx_for(job, workers: list[WorkerView], *overrides: str) -> SchedulingContext:  # type: ignore[no-untyped-def]
    return SchedulingContext(
        job.ready_tasks(), workers, job, load_config(overrides=list(overrides)), 0.0
    )


# ------------------------------------------------------------- policies


def test_round_robin_cycles_one_tile_each() -> None:
    job = image_job()
    policy = RoundRobinPolicy(load_config())
    workers = [view("b", 2), view("a", 2), view("c", 1)]
    out = policy.assign(ctx_for(job, workers))
    assert [a.worker_id for a in out] == ["a", "b", "c", "a", "b"]
    assert all(len(a.task_ids) == 1 for a in out)
    assert len({a.task_ids for a in out}) == 5  # no tile twice
    assert [a.task_ids[0] for a in out] == job.stage1_ids()[:5]


def test_round_robin_rotation_persists_across_rounds() -> None:
    job = image_job()
    policy = RoundRobinPolicy(load_config())
    first = policy.assign(ctx_for(job, [view("a"), view("b")]))
    second = policy.assign(ctx_for(job, [view("a"), view("b")]))
    assert [a.worker_id for a in first + second] == ["a", "b", "a", "b"]


def test_round_robin_local_participation() -> None:
    job = image_job()
    workers = [view("local", 3, trusted=True), view("r", 1)]
    with_local = RoundRobinPolicy(load_config()).assign(ctx_for(job, workers))
    assert {a.worker_id for a in with_local} == {"local", "r"}
    without = RoundRobinPolicy(load_config(overrides=["scheduler.allow_local=false"]))
    assert {a.worker_id for a in without.assign(ctx_for(job, workers))} == {"r"}


def test_round_robin_without_workers_or_tasks() -> None:
    job = image_job()
    policy = RoundRobinPolicy(load_config())
    assert policy.assign(ctx_for(job, [])) == []
    assert policy.assign(ctx_for(job, [view("a", 0)])) == []


def test_local_only_policy_chunks_and_capacity() -> None:
    job = image_job(rows=1, cols=5)
    workers = [view("remote", 5), view("local", 2, trusted=True)]
    out = LocalOnlyPolicy(load_config(overrides=["chunking.fixed_chunk_tiles=2"])).assign(
        ctx_for(job, workers)
    )
    assert [a.task_ids for a in out] == [("s1_r00_c00", "s1_r00_c01"), ("s1_r00_c02", "s1_r00_c03")]
    assert {a.worker_id for a in out} == {"local"}
    assert LocalOnlyPolicy(load_config()).assign(ctx_for(job, [view("remote", 5)])) == []


def test_contiguous_ready_respects_rows_status_and_taken() -> None:
    job = image_job(rows=2, cols=4)
    ctx = ctx_for(job, [])
    first = job.tasks["s1_r00_c02"]
    assert ctx.contiguous_ready(first, 8, set()) == ["s1_r00_c02", "s1_r00_c03"]  # row end
    assert ctx.contiguous_ready(job.tasks["s1_r00_c00"], 8, {"s1_r00_c02"}) == [
        "s1_r00_c00",
        "s1_r00_c01",
    ]
    job.tasks["s1_r00_c01"].status = TaskStatus.RUNNING
    assert ctx.contiguous_ready(job.tasks["s1_r00_c00"], 8, set()) == ["s1_r00_c00"]
    syn = build_synthetic_job(3, 10, 1)
    assert ctx_for(syn, []).contiguous_ready(syn.tasks["syn_0000"], 5, set()) == ["syn_0000"]


def test_context_helpers() -> None:
    job = image_job()
    ctx = ctx_for(
        job,
        [view("a", ms_per_unit={"img_tile": 2.0}, bench_ms_per_unit=5.0), view("l", trusted=True)],
    )
    assert ctx.worker("a").speed("img_tile") == 2.0 and ctx.worker("a").speed("row_stats") == 5.0
    assert ctx.local_worker() is not None and ctx.local_worker().worker_id == "l"
    with pytest.raises(KeyError):
        ctx.worker("zzz")
    one = job.tasks["s1_r00_c00"].work_units
    assert ctx.compute_s(["s1_r00_c00"], ctx.worker("a")) == pytest.approx(one * 2.0 / 1000)


def test_make_policy() -> None:
    assert make_policy(load_config(overrides=["scheduler.policy=local"])).name == "local"
    assert (
        make_policy(load_config(overrides=["scheduler.policy=round_robin"])).name == "round_robin"
    )
    assert make_policy(load_config()).name == "tesar"  # the default, since Phase 9
    assert (
        make_policy(load_config(overrides=["scheduler.policy=most_reliable"])).name
        == "most_reliable"
    )
    cfg = load_config()
    cfg.scheduler.policy = "greedy"  # bypasses config validation on purpose
    with pytest.raises(UnknownPolicyError, match="not available"):
        make_policy(cfg)


# -------------------------------------------------------------- registry


def test_registry_transitions_and_events() -> None:
    events = EventLogger()
    reg = Registry(events)
    rec = reg.upsert("w1", "phone")
    assert reg.upsert("w1", "renamed") is rec and rec.name == "renamed"
    assert reg.set_state(rec, WorkerState.PROBING, "auth")
    assert reg.set_state(rec, WorkerState.ACTIVE)
    assert not reg.set_state(rec, WorkerState.ACTIVE)  # no-op
    assert not reg.set_state(rec, WorkerState.CONNECTING)  # not allowed from ACTIVE
    assert reg.active() == [rec] and reg.connected() == [rec] and reg.remote_active() == [rec]
    assert reg.set_state(rec, WorkerState.QUARANTINED, "strikes")
    assert not reg.set_state(rec, WorkerState.DEAD)  # quarantine is sticky
    assert reg.active() == [] and len(reg) == 1 and list(reg) == [rec]
    states = [(e["old"], e["new"]) for e in events.recent() if e["type"] == "worker_state"]
    assert states == [("CONNECTING", "PROBING"), ("PROBING", "ACTIVE"), ("ACTIVE", "QUARANTINED")]


def test_worker_speed_estimate() -> None:
    rec = WorkerRecord("w", "w", bench_ms_per_unit=10.0)
    assert rec.ms_per_unit("img_tile") == 10.0
    rec.observe_speed("img_tile", compute_ms=40.0, work_units=2.0)
    assert rec.ms_per_unit("img_tile") == 20.0
    rec.observe_speed("img_tile", compute_ms=0.0, work_units=2.0)  # ignored
    rec.observe_speed("img_tile", compute_ms=10.0, work_units=0.0)  # ignored
    rec.observe_speed("img_tile", compute_ms=10.0, work_units=1.0)
    assert rec.ms_per_unit("img_tile") == pytest.approx(0.3 * 10 + 0.7 * 20)
    assert rec.ms_per_unit("row_stats") == 10.0
    snap = rec.snapshot()
    assert snap["state"] == "CONNECTING" and snap["avg_chunk_tiles"] is None


# ------------------------------------------------------------- scheduler


class FakeDispatcher:
    def __init__(self) -> None:
        self.sent: list[Assignment] = []

    def send(self, assignment: Assignment) -> object:
        self.sent.append(assignment)
        return object()


class EvilPolicy:
    """Returns assignments the scheduler must refuse."""

    name = "evil"

    def assign(self, ctx: SchedulingContext) -> list[Assignment]:
        return [
            Assignment(("s1_r00_c00",), "a"),
            Assignment(("s1_r00_c00",), "a"),  # same tile twice
            Assignment(("s1_r00_c01",), "ghost"),  # unknown worker
            Assignment(("nope",), "a"),  # unknown task
            Assignment((), "a"),  # empty
            Assignment(("s2_r00",), "a"),  # not runnable (PENDING)
            Assignment(("s1_r00_c02",), "a"),
            Assignment(("s1_r00_c03",), "a"),  # over capacity
        ]


def test_scheduler_enforces_capacity_and_validity() -> None:
    cfg = load_config(overrides=["workers.prefetch=1"])
    reg = Registry(EventLogger())
    rec = reg.upsert("a", "a")
    reg.set_state(rec, WorkerState.PROBING)
    reg.set_state(rec, WorkerState.ACTIVE)
    fake = FakeDispatcher()
    sched = Scheduler(cfg, reg, fake, EvilPolicy())  # type: ignore[arg-type]
    assert sched.schedule_once() == 0  # no job yet
    sched.start_job(image_job())
    fake_inflight = {}

    def send(assignment: Assignment) -> object:
        fake.sent.append(assignment)
        fake_inflight[len(fake_inflight)] = assignment
        rec.inflight[str(len(rec.inflight))] = object()  # type: ignore[assignment]
        return object()

    fake.send = send  # type: ignore[method-assign]
    assert sched.schedule_once() == 2  # slots(1) + prefetch(1)
    assert [a.task_ids for a in fake.sent] == [("s1_r00_c00",), ("s1_r00_c02",)]
    sched.stop_job()
    assert sched.schedule_once() == 0


# ------------------------------------------------------------ dispatcher


def test_dispatcher_predictions() -> None:
    cfg = load_config(
        overrides=[
            "liveness.task_timeout_min_s=3",
            "liveness.task_timeout_factor=3",
            "liveness.task_timeout_slack_s=2",
        ]
    )
    disp = Dispatcher(
        cfg,
        Registry(EventLogger()),
        EventLogger(),
        SecurityLog(None),
        lambda _wid: None,
        lambda: None,
    )
    assert disp.predict_s(["x"], WorkerRecord("w", "w")) == 0.0  # no job
    job = image_job()
    disp.start_job(job)
    rec = WorkerRecord("w", "w", bench_ms_per_unit=1000.0)
    task = job.tasks["s1_r00_c00"]
    # queue 0 + rtt 0 + bytes / default bandwidth + compute
    expected = (task.input_bytes_est + task.output_bytes_est) / 1_000_000 + task.work_units
    assert disp.predict_s(["s1_r00_c00"], rec) == pytest.approx(expected)
    disp.netest.on_rtt("w", 0.25)
    assert disp.predict_s(["s1_r00_c00"], rec) == pytest.approx(expected + 0.25)
    assert disp.timeout_s(0.1) == 3.0  # minimum
    assert disp.timeout_s(10.0) == 32.0
    assert disp.send(Assignment(("s1_r00_c00",), "nobody")) is None


# --------------------------------------------------------------- beacon


def test_beacon_payload_is_a_valid_beacon() -> None:
    fp = "sha256:" + "ab" * 32
    payload = B.beacon_payload("192.168.0.10", 50010, "coord_1", "tesar-live", fp)
    assert len(payload) < 1024
    beacon = parse_beacon(payload, fp)
    assert beacon is not None and (beacon.host, beacon.port) == ("192.168.0.10", 50010)


def test_beacon_destinations_and_subnet() -> None:
    assert B.subnet_broadcast("192.168.0.10") == "192.168.0.255"
    assert B.subnet_broadcast("127.0.0.1") is None
    assert B.subnet_broadcast("nonsense") is None
    both = B.beacon_destinations(["broadcast", "multicast"], "10.1.2.3", 50000, "239.255.77.77")
    assert both == [("255.255.255.255", 50000), ("10.1.2.255", 50000), ("239.255.77.77", 50000)]
    assert B.beacon_destinations([], "10.1.2.3", 50000, "239.255.77.77") == []
    assert B.beacon_destinations(["broadcast"], "127.0.0.1", 1, "x") == [("255.255.255.255", 1)]


async def test_beacon_sender_sends_and_survives_errors() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.settimeout(2)
        port = listener.getsockname()[1]
        sender = B.BeaconSender(b'{"x":1}', [("127.0.0.1", port), ("256.1.1.1", port)], 0.05)
        assert sender.send_once() == 1  # the invalid destination fails, is logged once
        assert listener.recv(100) == b'{"x":1}'
        sender.start()
        await asyncio.sleep(0.12)
        await sender.stop()
        assert sender.sent >= 2
    empty = B.BeaconSender(b"x", [], 1.0)
    empty.start()
    await empty.stop()


def test_detect_lan_ip_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    class Failing:
        def __init__(self, *a: object) -> None: ...
        def connect(self, addr: object) -> None:
            raise OSError("no route")

        def close(self) -> None: ...

    monkeypatch.setattr(socket, "socket", Failing)
    assert B.detect_lan_ip() == "127.0.0.1"


# --------------------------------------------------------------- metrics


def test_report_on_untouched_job_and_estimates() -> None:
    job = image_job()
    reg = Registry(EventLogger())
    reg.upsert("w", "phone").battery = {"pct": 50.0}
    disp = Dispatcher(
        load_config(), reg, EventLogger(), SecurityLog(None), lambda _w: None, lambda: None
    )
    report = build_report(job, load_config(), reg, disp.traffic, RunTiming(0.0, 1.0), {}, 42)
    assert report["speedup"] is None and report["avg_chunk_tiles"] is None
    assert report["per_worker"]["phone"]["battery_end"] == 50.0
    assert report["n_tiles"] == 8 and report["seed"] == 42
    est = local_estimate_s(job, {"img_tile": 1000.0}, fallback=1.0)
    assert est > 0
    assert wasted_work_s(job) == 0.0


async def test_scheduler_start_stop_never_hangs() -> None:
    """Regression: on Python <= 3.11 asyncio.wait_for swallows a cancellation that
    arrives right after its inner wait completed (CPython gh-86296). The scheduler is
    woken (trigger) and then stopped at once — exactly what happens when a worker
    disconnects during shutdown — and Coordinator.stop() used to hang forever."""
    reg = Registry(EventLogger())
    sched = Scheduler(load_config(), reg, FakeDispatcher(), RoundRobinPolicy(load_config()))  # type: ignore[arg-type]
    for i in range(300):
        sched.start()
        await asyncio.sleep(0.0005 * (i % 4))
        sched.trigger()  # the wake-up completes the loop's inner wait...
        await asyncio.wait_for(sched.stop(), 2.0)  # ...and the cancel arrives right after
        assert sched._task is None
