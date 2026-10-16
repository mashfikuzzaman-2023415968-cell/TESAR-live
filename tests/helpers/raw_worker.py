"""A raw protocol client for tests that need to misbehave like a hostile worker."""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from typing import Any

from mdc.common import auth as A
from mdc.common import protocol as P
from mdc.common import tls
from mdc.common.credentials import JoinCode


def hello_header(code: JoinCode, nonce_c: bytes, nonce_w: bytes, **overrides: Any) -> dict:
    """HELLO fields for ``code`` (overridable, e.g. ``trusted_local=True``)."""
    header: dict[str, Any] = {
        "worker_id": code.worker_id,
        "name": code.name,
        "platform": "linux",
        "device_class": "laptop",
        "cpu_count": 4,
        "slots": 1,
        "bench_ms_per_unit": 5.0,
        "battery": {
            "pct": 90.0,
            "charging": False,
            "current_ma": -300.0,
            "voltage_v": 3.8,
            "source": "mock",
            "ts_w": 1.0,
        },
        "version": "0.1.0",
        "trusted_local": False,
        "nonce_w": nonce_w.hex(),
        "auth": A.worker_auth(code.secret, nonce_c, nonce_w, code.worker_id, code.fp),
    }
    header.update(overrides)
    return header


@dataclass
class RawWorker:
    """An authenticated (or rejected) raw connection."""

    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    reply: P.Message
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _stash: list[P.Message] = field(default_factory=list)

    @property
    def welcomed(self) -> bool:
        return self.reply.type == P.WELCOME

    async def send(self, msg: P.Message) -> None:
        await P.write_frame(self.writer, msg, self.lock)

    async def recv(self, msg_type: str, timeout: float = 10.0) -> P.Message:
        """Next message of ``msg_type``; answers PINGs meanwhile so we stay alive."""
        for i, msg in enumerate(self._stash):
            if msg.type == msg_type:
                return self._stash.pop(i)
        deadline = time.monotonic() + timeout
        while True:
            msg = await asyncio.wait_for(
                P.read_frame(self.reader), max(0.01, deadline - time.monotonic())
            )
            if msg.type == msg_type:
                return msg
            if msg.type == P.PROBE:  # behave like a real worker: acknowledge the probe
                await self.send(
                    P.make_message(
                        P.PROBE_ACK,
                        probe_id=msg.header["probe_id"],
                        t_c=msg.header["t_c"],
                        received_bytes=len(msg.payload),
                    )
                )
                continue
            if msg.type == P.PING:
                await self.send(
                    P.make_message(
                        P.PONG,
                        seq=msg.header["seq"],
                        t_c=msg.header["t_c"],
                        active_tasks=0,
                        queued_tasks=0,
                        battery={"pct": 90.0},
                    )
                )
                continue
            self._stash.append(msg)

    async def closed(self, timeout: float = 10.0) -> bool:
        """True if the server closes the connection within ``timeout``."""
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                await asyncio.wait_for(P.read_frame(self.reader), deadline - time.monotonic())
        except asyncio.TimeoutError:
            # Must come first: on Python >= 3.11 asyncio.TimeoutError is the builtin
            # TimeoutError, a subclass of OSError — a timeout is NOT a closed socket.
            return False
        except (P.WireError, OSError, asyncio.IncompleteReadError):
            # The peer closed it; close our side too (on Python 3.10 an SSL transport
            # closed only by the peer is otherwise reported as leaked when collected).
            self.writer.close()
            return True
        return False

    async def close(self) -> None:
        """Drop the connection at once.

        A graceful TLS close waits for the peer's ``close_notify``, which can
        never be read if unread data (e.g. the 256 KB bandwidth probe) has
        paused reading — the socket would then leak. Tests do not need grace.
        """
        self.writer.transport.abort()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self.writer.wait_closed(), 2)


async def connect_raw(code: JoinCode, port: int, **hello_overrides: Any) -> RawWorker:
    """TLS (pinned) + CHALLENGE/HELLO; returns after WELCOME or REJECT."""
    reader, writer = await tls.open_pinned_connection("127.0.0.1", port, code.fp, 5)
    try:
        challenge = await asyncio.wait_for(P.read_frame(reader), 5)
        nonce_c = A.nonce_from_hex(challenge.header["nonce"])
        nonce_w = A.make_nonce()
        lock = asyncio.Lock()
        hello = P.Message(P.HELLO, hello_header(code, nonce_c, nonce_w, **hello_overrides))
        await P.write_frame(writer, hello, lock)
        reply = await asyncio.wait_for(P.read_frame(reader), 5)
    except BaseException:
        writer.close()  # never leak the connection when the handshake fails
        raise
    return RawWorker(reader, writer, reply, lock)
