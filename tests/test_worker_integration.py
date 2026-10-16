"""Worker against a stub coordinator over real TLS loopback (spec Phase 3 acceptance).

The worker connects, authenticates, answers PINGs while computing, executes
tasks, handles cancel/busy/limits, and reconnects after the coordinator restarts.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import numpy as np
import pytest

from mdc.common import protocol as P
from mdc.common.auth import make_secret
from mdc.common.credentials import JoinCode, read_join_file
from mdc.common.validation import TaskLimits
from mdc.coordinator.certs import CertInfo
from mdc.coordinator.enrollment import DeviceRegistry
from mdc.tasks.image_pipeline import pipeline_valid
from mdc.tasks.synthetic import hash_chain
from mdc.worker.battery import MockBattery
from mdc.worker.faults import FaultSettings
from mdc.worker.shaper import Shaper
from tests.helpers.stub_coordinator import StubCoordinator, StubSession
from tests.helpers.worker_harness import running_worker

pytestmark = [pytest.mark.integration, pytest.mark.timeout(60)]


@pytest.fixture
async def stub(tls_cert: CertInfo, registry: DeviceRegistry) -> AsyncIterator[StubCoordinator]:
    async with StubCoordinator(tls_cert, registry) as s:
        yield s


@pytest.fixture
def join(stub: StubCoordinator, enrolled: Callable[..., Path]) -> JoinCode:
    return read_join_file(enrolled("phone-a", port=stub.port))


def assign(
    attempt: str, kind: str, params: dict, data: np.ndarray | None = None, task_id: str = "t1"
) -> P.Message:
    fields: dict = {}
    payload = b""
    if data is not None:
        fields, payload = P.pack_array(data)
    return P.make_message(
        P.TASK_ASSIGN,
        payload,
        task_id=task_id,
        attempt_id=attempt,
        kind=kind,
        params=params,
        **fields,
    )


async def ping(session: StubSession, seq: int) -> float:
    t0 = time.monotonic()
    await session.send(P.make_message(P.PING, seq=seq, t_c=t0))
    pong = await session.recv(P.PONG, timeout=5)
    assert pong.header["seq"] == seq and pong.header["t_c"] == t0
    return time.monotonic() - t0


async def test_connects_and_authenticates(stub: StubCoordinator, join: JoinCode) -> None:
    async with running_worker(join, stub.port, slots=2):
        session = await stub.next_session()
        h = session.hello
        assert h["worker_id"] == join.worker_id and h["name"] == "phone-a"
        assert h["slots"] == 2 and h["trusted_local"] is False
        assert h["platform"] in {"linux", "android", "darwin", "windows"}
        assert h["battery"]["source"] == "mock"
        assert await ping(session, 1) < 1.0


async def test_answers_pings_while_computing(stub: StubCoordinator, join: JoinCode) -> None:
    async with running_worker(join, stub.port):
        session = await stub.next_session()
        await session.send(assign("att_long", "synthetic", {"seed": 5, "chain_len": 1_500_000}))
        rtts = []
        result = None
        for seq in range(200):
            rtts.append(await ping(session, seq))
            pending = [m for m in session.drain() if m.type == P.TASK_RESULT]
            if pending:
                result = pending[0]
                break
            await asyncio.sleep(0.05)
        if result is None:
            result = await session.recv(P.TASK_RESULT, timeout=30)
        assert len(rtts) >= 3, "the task finished too fast to observe pings during compute"
        assert max(rtts) < 0.5, f"PONG delayed while computing: {max(rtts):.3f} s"
        assert result.header["status"] == "ok"
        assert result.header["json_result"] == {"digest": hash_chain(5, 1_500_000)}
        assert result.header["compute_ms"] > 0


async def test_img_tile_and_row_stats(stub: StubCoordinator, join: JoinCode) -> None:
    data = np.random.default_rng(3).integers(0, 256, size=(40, 60), dtype=np.uint8)
    async with running_worker(join, stub.port):
        session = await stub.next_session()
        await session.send(assign("att_img", "img_tile", {"r": 2, "iters": 2}, data))
        result = await session.recv(P.TASK_RESULT)
        assert result.header["status"] == "ok"
        out = P.unpack_array(result.header, result.payload)
        np.testing.assert_array_equal(out, pipeline_valid(data, 2, 2))
        assert result.header["queue_ms"] >= 0
        assert result.header["t_send_w"] >= result.header["t_recv_w"]
        await session.send(assign("att_rs", "row_stats", {"edge_threshold": 100}, data))
        stats = (await session.recv(P.TASK_RESULT)).header["json_result"]
        assert stats["pixels"] == data.size
        assert stats["edge_pixels"] == int((data >= 100).sum())


async def test_reconnects_after_coordinator_restart(
    tls_cert: CertInfo, registry: DeviceRegistry, enrolled: Callable[..., Path]
) -> None:
    first = await StubCoordinator(tls_cert, registry).start()
    port = first.port
    join = read_join_file(enrolled("phone-r", port=port))
    async with running_worker(join, port) as (worker, _task, _rec):
        session = await first.next_session()
        await ping(session, 1)
        await first.stop()  # coordinator goes away
        await asyncio.sleep(0.5)
        second = await StubCoordinator(tls_cert, registry, port=port).start()
        try:
            again = await second.next_session(timeout=20)
            assert again.hello["worker_id"] == join.worker_id  # same identity
            await ping(again, 2)
            assert worker.sessions_completed >= 1
        finally:
            await second.stop()


async def test_busy_when_queue_full(stub: StubCoordinator, join: JoinCode) -> None:
    async with running_worker(join, stub.port, slots=1, prefetch=0):
        session = await stub.next_session()
        long = {"seed": 1, "chain_len": 1_500_000}
        for i in range(3):  # bound = slots + prefetch + 1 = 2
            await session.send(assign(f"att_{i}", "synthetic", long, task_id=f"t{i}"))
        busy = await session.recv(P.TASK_RESULT, timeout=10)
        assert busy.header["attempt_id"] == "att_2"
        assert (busy.header["status"], busy.header["error"]) == ("error", "busy")
        for _ in range(2):
            assert (await session.recv(P.TASK_RESULT, timeout=30)).header["status"] == "ok"


async def test_cancel_queued_and_running(stub: StubCoordinator, join: JoinCode) -> None:
    async with running_worker(join, stub.port, slots=1, prefetch=2):
        session = await stub.next_session()
        long = {"seed": 1, "chain_len": 4_000_000}  # ~1 s: still running when cancelled
        await session.send(assign("att_run", "synthetic", long, task_id="a"))
        await session.send(assign("att_queued", "synthetic", long, task_id="b"))
        await session.send(
            assign("att_keep", "synthetic", {"seed": 2, "chain_len": 10}, task_id="c")
        )
        await asyncio.sleep(0.2)
        await session.send(P.make_message(P.TASK_CANCEL, attempt_id="att_queued"))
        await session.send(P.make_message(P.TASK_CANCEL, attempt_id="att_run"))
        await session.send(P.make_message(P.TASK_CANCEL, attempt_id="att_unknown"))
        kept = await session.recv(P.TASK_RESULT, timeout=30)
        assert kept.header["attempt_id"] == "att_keep"
        await asyncio.sleep(1.5)
        others = [m for m in session.drain() if m.type == P.TASK_RESULT]
        assert others == []  # neither the cancelled queued nor the running task answered


async def test_tampered_payload_reports_integrity_error(
    stub: StubCoordinator, join: JoinCode
) -> None:
    data = np.zeros((20, 20), dtype=np.uint8)
    msg = assign("att_bad", "img_tile", {"r": 1, "iters": 1}, data)
    msg.header["sha256"] = "0" * 64
    async with running_worker(join, stub.port):
        session = await stub.next_session()
        await session.send(msg)
        reply = await session.recv(P.TASK_RESULT)
        assert (reply.header["status"], reply.header["error"]) == ("error", "integrity")
        await ping(session, 1)  # still connected


async def test_task_errors_are_reported(stub: StubCoordinator, join: JoinCode) -> None:
    data = np.zeros((20, 20), dtype=np.uint8)
    async with running_worker(join, stub.port):
        session = await stub.next_session()
        await session.send(assign("att_r", "img_tile", {"r": 9, "iters": 1}, data))
        reply = await session.recv(P.TASK_RESULT)
        assert reply.header["status"] == "error" and "'r'" in reply.header["error"]
        await session.send(assign("att_small", "img_tile", {"r": 2, "iters": 5}, data))
        reply = await session.recv(P.TASK_RESULT)
        assert reply.header["status"] == "error" and "halo" in reply.header["error"]


async def test_time_limit(stub: StubCoordinator, join: JoinCode) -> None:
    async with running_worker(join, stub.port, task_max_s=0.05):
        session = await stub.next_session()
        await session.send(assign("att_t", "synthetic", {"seed": 1, "chain_len": 3_000_000}))
        reply = await session.recv(P.TASK_RESULT, timeout=20)
        assert (reply.header["status"], reply.header["error"]) == ("error", "time_limit")


async def test_battery_floor_and_charging_bypass(stub: StubCoordinator, join: JoinCode) -> None:
    battery = MockBattery(start_pct=10.0, drain_pct_per_min=0.0)
    async with running_worker(
        join, stub.port, battery=battery, min_battery_pct=15.0, battery_interval_s=0.05
    ) as (worker, _t, _r):
        session = await stub.next_session()
        await session.send(assign("att_low", "synthetic", {"seed": 1, "chain_len": 10}))
        reply = await session.recv(P.TASK_RESULT)
        assert (reply.header["status"], reply.header["error"]) == ("error", "battery_low")
        battery.charging = True
        await asyncio.sleep(0.3)
        assert worker.battery_reading["charging"] is True
        await session.send(assign("att_chg", "synthetic", {"seed": 1, "chain_len": 10}))
        assert (await session.recv(P.TASK_RESULT)).header["status"] == "ok"


async def test_probe_ack_and_shaping(stub: StubCoordinator, join: JoinCode) -> None:
    shaper = Shaper(delay_ms=200, bw_kbps=8000)
    async with running_worker(join, stub.port, shaper=shaper):
        session = await stub.next_session()
        t0 = time.monotonic()
        await session.send(P.make_message(P.PROBE, b"x" * 50_000, probe_id="p1", t_c=t0))
        ack = await session.recv(P.PROBE_ACK)
        elapsed = time.monotonic() - t0
        assert ack.header["received_bytes"] == 50_000
        assert ack.header["t_c"] == t0
        assert elapsed >= 0.2  # shaped
        assert await ping(session, 1) < 0.15  # PONG is never shaped


async def test_unknown_messages_tolerated_then_disconnect(
    stub: StubCoordinator, join: JoinCode
) -> None:
    async with running_worker(join, stub.port):
        session = await stub.next_session()
        for i in range(10):
            await session.send(P.make_message("FUTURE_THING", n=i))
        await ping(session, 1)  # 10 are tolerated (forward compatibility)
        await session.send(P.make_message("FUTURE_THING", n=11))
        await asyncio.wait_for(session.closed.wait(), 5)


async def test_bye_ends_session_and_worker_reconnects(
    stub: StubCoordinator, join: JoinCode
) -> None:
    async with running_worker(join, stub.port):
        session = await stub.next_session()
        await session.send(P.make_message(P.BYE, reason="restarting"))
        await asyncio.wait_for(session.closed.wait(), 5)
        again = await stub.next_session(timeout=10)
        await ping(again, 1)


async def test_idle_timeout_reconnects(stub: StubCoordinator, join: JoinCode) -> None:
    async with running_worker(join, stub.port, idle_timeout_s=0.5):
        first = await stub.next_session()
        await asyncio.wait_for(first.closed.wait(), 5)  # stub sends nothing -> worker gives up
        await stub.next_session(timeout=10)


async def test_coordinator_limits_can_only_tighten(
    tls_cert: CertInfo, registry: DeviceRegistry, enrolled: Callable[..., Path]
) -> None:
    loose = TaskLimits(blur_radius=50, blur_iterations=1000, chain_len=10**8)
    async with StubCoordinator(tls_cert, registry, limits=loose) as stub:
        join = read_join_file(enrolled("phone-l", port=stub.port))
        async with running_worker(join, stub.port):
            session = await stub.next_session()
            data = np.zeros((40, 40), dtype=np.uint8)
            await session.send(assign("a", "img_tile", {"r": 6, "iters": 1}, data))
            reply = await session.recv(P.TASK_RESULT)
            assert reply.header["status"] == "error"  # worker's own limit (5) still applies
    tight = TaskLimits(chain_len=100)
    async with StubCoordinator(tls_cert, registry, limits=tight) as stub:
        join = read_join_file(enrolled("phone-t", port=stub.port))
        async with running_worker(join, stub.port):
            session = await stub.next_session()
            await session.send(assign("b", "synthetic", {"seed": 1, "chain_len": 101}))
            reply = await session.recv(P.TASK_RESULT)
            assert reply.header["status"] == "error"  # coordinator tightened it


async def test_accepted_slots_respected(
    tls_cert: CertInfo, registry: DeviceRegistry, enrolled: Callable[..., Path]
) -> None:
    async with StubCoordinator(tls_cert, registry, accepted_slots=1) as stub:
        join = read_join_file(enrolled("phone-s", port=stub.port))
        async with running_worker(join, stub.port, slots=4):
            session = await stub.next_session()
            for i in range(2):
                await session.send(
                    assign(
                        f"a{i}", "synthetic", {"seed": i, "chain_len": 1_500_000}, task_id=f"t{i}"
                    )
                )
            seen = []
            for seq in range(40):
                await session.send(P.make_message(P.PING, seq=seq, t_c=0.0))
                pong = await session.recv(P.PONG)
                seen.append((pong.header["active_tasks"], pong.header["queued_tasks"]))
                await asyncio.sleep(0.05)
            # The coordinator accepted 1 slot: never two tasks running at once,
            # and the second one waited in the queue.
            assert max(active for active, _ in seen) == 1
            assert (1, 1) in seen


# ------------------------------------------------------------- faults


async def test_malicious_result_is_plausible_but_wrong(
    stub: StubCoordinator, join: JoinCode
) -> None:
    data = np.random.default_rng(8).integers(0, 256, size=(50, 50), dtype=np.uint8)
    async with running_worker(
        join, stub.port, faults=FaultSettings(malicious_prob=1.0), lab_mode=True
    ):
        session = await stub.next_session()
        await session.send(assign("a", "img_tile", {"r": 1, "iters": 1}, data))
        reply = await session.recv(P.TASK_RESULT)
        out = P.unpack_array(reply.header, reply.payload)  # transport integrity passes
        expected = pipeline_valid(data, 1, 1)
        assert out.shape == expected.shape and out.dtype == expected.dtype
        assert not np.array_equal(out, expected)
        await session.send(assign("b", "synthetic", {"seed": 1, "chain_len": 5}))
        bad = (await session.recv(P.TASK_RESULT)).header["json_result"]["digest"]
        assert bad != hash_chain(1, 5) and len(bad) == 64


async def test_drop_fault(stub: StubCoordinator, join: JoinCode) -> None:
    async with running_worker(join, stub.port, faults=FaultSettings(drop_prob=1.0), lab_mode=True):
        session = await stub.next_session()
        await session.send(assign("a", "synthetic", {"seed": 1, "chain_len": 5}))
        await ping(session, 1)
        await asyncio.sleep(0.5)
        assert [m for m in session.drain() if m.type == P.TASK_RESULT] == []


async def test_crash_faults(stub: StubCoordinator, join: JoinCode) -> None:
    async with running_worker(
        join, stub.port, faults=FaultSettings(crash_prob=1.0), lab_mode=True
    ) as (_w, _t, recorder):
        session = await stub.next_session()
        await session.send(assign("a", "synthetic", {"seed": 1, "chain_len": 5}))
        await ping(session, 1)
        assert recorder.codes == [1]
    async with running_worker(
        join, stub.port, faults=FaultSettings(crash_after_s=0.2), lab_mode=True
    ) as (_w, _t, recorder):
        await asyncio.sleep(0.6)
        assert recorder.codes == [1]


async def test_slowdown_fault(stub: StubCoordinator, join: JoinCode) -> None:
    async with running_worker(join, stub.port, faults=FaultSettings(slowdown=4.0), lab_mode=True):
        session = await stub.next_session()
        await session.send(assign("a", "synthetic", {"seed": 1, "chain_len": 200_000}))
        t0 = time.monotonic()
        reply = await session.recv(P.TASK_RESULT, timeout=30)
        assert reply.header["compute_ms"] / 1000 <= time.monotonic() - t0 + 0.05
    async with running_worker(join, stub.port):
        session = await stub.next_session()
        await session.send(assign("b", "synthetic", {"seed": 1, "chain_len": 200_000}))
        normal = (await session.recv(P.TASK_RESULT, timeout=30)).header["compute_ms"]
    assert reply.header["compute_ms"] > 2.5 * normal


async def test_flaky_fault_disconnects_and_returns(stub: StubCoordinator, join: JoinCode) -> None:
    async with running_worker(
        join, stub.port, faults=FaultSettings(flaky_mean_s=0.2), lab_mode=True
    ) as (worker, _t, _r):
        worker.faults.flaky_reconnect_s = lambda: 0.2  # type: ignore[method-assign]
        first = await stub.next_session()
        await asyncio.wait_for(first.closed.wait(), 10)
        await stub.next_session(timeout=10)


async def test_control_applies_in_lab_mode(
    stub: StubCoordinator, join: JoinCode, caplog: pytest.LogCaptureFixture
) -> None:
    async with running_worker(join, stub.port, lab_mode=True, allow_demo_control=True) as (
        worker,
        task,
        recorder,
    ):
        session = await stub.next_session()
        await session.send(P.make_message(P.CONTROL, action="set_malicious", value=0.5))
        await session.send(P.make_message(P.CONTROL, action="set_slowdown", value=2.0))
        await session.send(P.make_message(P.CONTROL, action="set_drop", value=0.25))
        await ping(session, 1)
        s = worker.faults.settings
        assert (s.malicious_prob, s.slowdown, s.drop_prob) == (0.5, 2.0, 0.25)
        await session.send(P.make_message(P.CONTROL, action="crash"))
        await ping(session, 2)
        assert recorder.codes == [1]
        await session.send(P.make_message(P.CONTROL, action="leave"))
        bye = await session.recv(P.BYE)
        assert "leaving" in bye.header["reason"]
        assert await asyncio.wait_for(task, 5) == 0  # worker exits cleanly


async def test_rejected_worker_exits(
    stub: StubCoordinator, join: JoinCode, registry: DeviceRegistry
) -> None:
    registry.revoke("phone-a")
    async with running_worker(join, stub.port) as (_w, task, _r):
        assert await asyncio.wait_for(task, 10) == 3
    assert stub.rejections == ["revoked"]


async def test_wrong_proof_is_not_trusted(
    tls_cert: CertInfo,
    registry: DeviceRegistry,
    enrolled: Callable[..., Path],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    async with StubCoordinator(tls_cert, registry, proof_secret_override=make_secret()) as stub:
        join = read_join_file(enrolled("phone-p", port=stub.port))
        async with running_worker(join, stub.port) as (worker, task, _r):
            await asyncio.sleep(1.5)
            assert not task.done()  # keeps retrying (logged), never serves this peer
            assert worker.sessions_completed == 0
    assert "coordinator_proof_failed" in caplog.text


async def test_slowdown_fault_slows_the_reported_benchmark(
    stub: StubCoordinator, join: JoinCode
) -> None:
    async with running_worker(join, stub.port, faults=FaultSettings(slowdown=4.0), lab_mode=True):
        session = await stub.next_session()
        assert session.hello["bench_ms_per_unit"] == 4.0  # harness benchmark 1.0 x slowdown 4
