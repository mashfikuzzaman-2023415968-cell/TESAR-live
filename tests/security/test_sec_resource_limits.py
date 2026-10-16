"""Server resource limits and DoS controls (spec S.6, S.12 row 4, G.3 row 4)."""

from __future__ import annotations

import asyncio
import contextlib
import struct
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from mdc.common import protocol as P
from mdc.common import tls
from mdc.common.credentials import JoinCode, read_join_file
from mdc.coordinator.coordinator import Coordinator
from mdc.coordinator.security_log import read_security_log
from tests.helpers.raw_worker import connect_raw

pytestmark = [pytest.mark.security, pytest.mark.integration, pytest.mark.timeout(60)]

StartCoordinator = Callable[..., Awaitable[Coordinator]]


def sec_events(coord: Coordinator, event_type: str) -> list[dict]:
    path = Path(coord.cfg.security.state_dir) / "logs" / "security.jsonl"
    return [r for r in read_security_log(path) if r["type"] == event_type] if path.exists() else []


async def is_closed(reader: asyncio.StreamReader, timeout: float) -> bool:
    try:
        data = await asyncio.wait_for(reader.read(1024), timeout)
    except asyncio.TimeoutError:
        # First: on Python >= 3.11 TimeoutError subclasses OSError.
        return False
    except (ConnectionError, OSError):
        return True
    return data == b"" or await is_closed(reader, timeout)


async def tls_connect(coord: Coordinator) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    return await tls.open_pinned_connection("127.0.0.1", coord.port, coord.cert.fingerprint, 5)


async def close_all(writers: list[asyncio.StreamWriter]) -> None:
    for w in writers:
        w.close()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(w.wait_closed(), 1)


async def test_connection_flood_beyond_max_connections(coordinator: StartCoordinator) -> None:
    coord = await coordinator(
        "network.max_connections=3",
        "network.max_connections_per_ip=50",
        "network.handshake_timeout_s=10",
    )
    held = [await tls_connect(coord) for _ in range(3)]
    for reader, _ in held:  # each receives a CHALLENGE and is now "open"
        assert (await asyncio.wait_for(P.read_frame(reader), 5)).type == P.CHALLENGE
    extra_r, extra_w = await tls_connect(coord)
    assert await is_closed(extra_r, 5)  # the 4th is closed straight away
    assert sec_events(coord, "connection_limit")
    await close_all([w for _, w in held] + [extra_w])


async def test_per_ip_limit(coordinator: StartCoordinator) -> None:
    coord = await coordinator("network.max_connections_per_ip=2", "network.handshake_timeout_s=10")
    held = [await tls_connect(coord) for _ in range(2)]
    for reader, _ in held:
        await asyncio.wait_for(P.read_frame(reader), 5)
    extra_r, extra_w = await tls_connect(coord)
    assert await is_closed(extra_r, 5)
    events = sec_events(coord, "connection_limit")
    assert events and events[-1]["open_from_ip"] == 2
    await close_all([w for _, w in held] + [extra_w])
    # Slots are released when connections close.
    await asyncio.sleep(0.3)
    r, w = await tls_connect(coord)
    assert (await asyncio.wait_for(P.read_frame(r), 5)).type == P.CHALLENGE
    await close_all([w])


async def test_slowloris_without_tls_is_dropped(coordinator: StartCoordinator) -> None:
    coord = await coordinator("network.handshake_timeout_s=0.5")
    reader, writer = await asyncio.open_connection("127.0.0.1", coord.port)
    writer.write(b"\x16")  # one byte of a TLS ClientHello, then silence
    await writer.drain()
    assert await is_closed(reader, 5)
    await close_all([writer])


async def test_silent_after_tls_is_dropped_and_counted(coordinator: StartCoordinator) -> None:
    coord = await coordinator("network.handshake_timeout_s=0.5")
    reader, writer = await tls_connect(coord)
    assert (await asyncio.wait_for(P.read_frame(reader), 5)).type == P.CHALLENGE
    assert await is_closed(reader, 5)  # never sent HELLO
    failures = sec_events(coord, "auth_failed")
    assert failures and failures[-1]["detail"] == "handshake timeout"
    await close_all([writer])


async def test_mid_frame_stall_after_auth_hits_idle_timeout(
    coordinator: StartCoordinator, enrolled: Callable[..., Path]
) -> None:
    coord = await coordinator("network.idle_read_timeout_s=0.6", "liveness.ping_interval_s=60")
    raw = await connect_raw(read_join_file(enrolled("staller")), coord.port)
    assert raw.welcomed
    raw.writer.write(struct.pack("!4sII", P.MAGIC, 20, 0) + b'{"type":')  # half a frame
    await raw.writer.drain()
    assert await raw.closed(5)


async def test_oversize_frame_closes_connection(
    coordinator: StartCoordinator, enrolled: Callable[..., Path]
) -> None:
    coord = await coordinator("network.max_payload_bytes=1000")
    raw = await connect_raw(read_join_file(enrolled("big")), coord.port)
    raw.writer.write(struct.pack("!4sII", P.MAGIC, 30, 10_000_000))  # claims 10 MB
    await raw.writer.drain()
    assert await raw.closed(5)
    assert sec_events(coord, "frame_rejected")


async def test_invalid_message_after_auth_closes(
    coordinator: StartCoordinator, enrolled: Callable[..., Path]
) -> None:
    coord = await coordinator()
    raw = await connect_raw(read_join_file(enrolled("bad-msg")), coord.port)
    await raw.send(P.make_message(P.TASK_RESULT, attempt_id="x y z"))  # schema violation
    assert await raw.closed(5)
    assert sec_events(coord, "validation_failed")


async def test_unexpected_messages_tolerated_up_to_limit(
    coordinator: StartCoordinator, enrolled: Callable[..., Path]
) -> None:
    coord = await coordinator()
    raw = await connect_raw(read_join_file(enrolled("chatty")), coord.port)
    for i in range(10):
        await raw.send(P.make_message("FUTURE_TYPE", n=i))
    assert not await raw.closed(0.5)
    await raw.send(P.make_message("FUTURE_TYPE", n=11))
    assert await raw.closed(5)


async def test_unsolicited_results_disconnect_and_strike(
    coordinator: StartCoordinator, enrolled: Callable[..., Path]
) -> None:
    coord = await coordinator()
    code = read_join_file(enrolled("spammer"))
    raw = await connect_raw(code, coord.port)
    for i in range(4):
        await raw.send(
            P.make_message(
                P.TASK_RESULT,
                task_id="t",
                attempt_id=f"att_fake{i}",
                status="error",
                error="x",
                compute_ms=0.0,
                queue_ms=0.0,
                t_recv_w=0.0,
                t_send_w=0.0,
            )
        )
    assert await raw.closed(5)
    assert len(sec_events(coord, "unsolicited_result")) == 4
    assert coord.registry.get(code.worker_id).strikes == 1


# ------------------------------------------------------- auth bans (live)


async def _bad_login(coord: Coordinator, code: JoinCode) -> str:
    forged = JoinCode(code.host, code.port, code.worker_id, code.name, b"\x00" * 32, code.fp)
    try:
        raw = await connect_raw(forged, coord.port)
    except (ConnectionError, OSError, P.WireError, asyncio.TimeoutError):
        return "closed"
    reason = raw.reply.header.get("reason", raw.reply.type)
    await raw.close()
    return reason


async def test_auth_failures_ban_the_ip_then_recover(
    coordinator: StartCoordinator, enrolled: Callable[..., Path]
) -> None:
    coord = await coordinator(
        "security.auth_fail_limit=3", "security.auth_fail_window_s=60", "security.auth_ban_s=1.5"
    )
    code = read_join_file(enrolled("victim"))
    assert [await _bad_login(coord, code) for _ in range(3)] == ["auth", "auth", "auth"]
    assert sec_events(coord, "ip_banned")
    # Banned: even the correct credential is turned away for now.
    with pytest.raises((ConnectionError, OSError, P.WireError, asyncio.TimeoutError)):
        good = await connect_raw(code, coord.port)
        await good.close()
    banned = sec_events(coord, "connection_limit")
    assert banned and banned[-1]["reason"] == "banned"
    await asyncio.sleep(1.6)  # ban expires
    good = await connect_raw(code, coord.port)
    assert good.welcomed
    await good.close()
