"""A minimal coordinator for testing workers (spec Phase 3 "Manual check").

It accepts TLS connections, performs the coordinator side of the handshake
(CHALLENGE / HELLO / WELCOME, with a real device registry), and then lets a
test send arbitrary messages and inspect everything the worker sends back.

Programmatic use (tests)::

    async with StubCoordinator(cert, registry) as stub:
        ... start a worker pointed at 127.0.0.1:stub.port ...
        session = await stub.next_session()
        await session.send(P.make_message(P.PING, seq=1, t_c=0.0))
        pong = await session.recv(P.PONG)

Manual check (prints the worker command, then pings and sends a long task)::

    python -m tests.helpers.stub_coordinator
"""

from __future__ import annotations

import asyncio
import contextlib
import ssl
import tempfile
import time
from pathlib import Path
from typing import Any

from mdc.common import auth as A
from mdc.common import protocol as P
from mdc.common import tls
from mdc.common.credentials import write_join_file
from mdc.common.validation import DEFAULT_TASK_LIMITS, TaskLimits, validate_message
from mdc.coordinator.certs import CertInfo, generate_certificate
from mdc.coordinator.enrollment import DeviceRegistry, LookupStatus


class StubSession:
    """One authenticated worker connection."""

    def __init__(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, hello: dict[str, Any]
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.hello = hello
        self.lock = asyncio.Lock()
        self.inbox: asyncio.Queue[P.Message] = asyncio.Queue()
        self.closed = asyncio.Event()
        self._stash: list[P.Message] = []
        self._reader_task = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        try:
            while True:
                msg = await P.read_frame(self.reader)
                await self.inbox.put(msg)
        except (P.WireError, OSError, ssl.SSLError, asyncio.IncompleteReadError):
            pass
        finally:
            self.closed.set()

    async def send(self, msg: P.Message) -> None:
        """Send a frame to the worker."""
        await P.write_frame(self.writer, msg, self.lock)

    async def recv(self, msg_type: str | None = None, timeout: float = 10.0) -> P.Message:
        """Next message (of ``msg_type`` if given).

        Messages of other types are kept (not lost) for later ``recv``/``drain`` calls.
        """
        for i, msg in enumerate(self._stash):
            if msg_type is None or msg.type == msg_type:
                return self._stash.pop(i)
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError(f"no {msg_type or 'message'} within {timeout} s")
            msg = await asyncio.wait_for(self.inbox.get(), remaining)
            if msg_type is None or msg.type == msg_type:
                return msg
            self._stash.append(msg)

    def drain(self) -> list[P.Message]:
        """All messages received so far and not yet consumed."""
        out, self._stash = self._stash, []
        while not self.inbox.empty():
            out.append(self.inbox.get_nowait())
        return out

    async def close(self) -> None:
        """Close the connection."""
        self.writer.close()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self.writer.wait_closed(), 2)
        self._reader_task.cancel()
        with contextlib.suppress(BaseException):
            await self._reader_task


class StubCoordinator:
    """TLS server that authenticates workers against ``registry``."""

    def __init__(
        self,
        cert: CertInfo,
        registry: DeviceRegistry,
        port: int = 0,
        accepted_slots: int = 8,
        limits: TaskLimits = DEFAULT_TASK_LIMITS,
        proof_secret_override: bytes | None = None,
    ) -> None:
        self.cert = cert
        self.registry = registry
        self.requested_port = port
        self.port = port
        self.accepted_slots = accepted_slots
        self.limits = limits
        self.proof_secret_override = proof_secret_override
        self.sessions: asyncio.Queue[StubSession] = asyncio.Queue()
        self.all_sessions: list[StubSession] = []
        self.connections = 0
        self.first_messages: list[str] = []
        self.rejections: list[str] = []
        self._server: asyncio.Server | None = None
        self._handlers: set[asyncio.Task[None]] = set()

    async def start(self) -> StubCoordinator:
        """Listen on 127.0.0.1 (ephemeral port unless one was given)."""
        ctx = tls.server_context(self.cert.cert_path, self.cert.key_path)
        self._server = await asyncio.start_server(
            self._handle, "127.0.0.1", self.requested_port, ssl=ctx
        )
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def stop(self) -> None:
        """Close the listener and every connection (explicitly — see DECISIONS 3.10)."""
        if self._server is not None:
            self._server.close()
        for session in self.all_sessions:
            await session.close()
        for task in list(self._handlers):
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
        if self._server is not None:
            await self._server.wait_closed()
            self._server = None

    async def __aenter__(self) -> StubCoordinator:
        return await self.start()

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    async def next_session(self, timeout: float = 15.0) -> StubSession:
        """Wait for the next authenticated worker."""
        return await asyncio.wait_for(self.sessions.get(), timeout)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        current = asyncio.current_task()
        if current is not None:
            self._handlers.add(current)
        self.connections += 1
        handed_over = False
        try:
            lock = asyncio.Lock()
            nonce_c = A.make_nonce()
            await P.write_frame(
                writer,
                P.make_message(P.CHALLENGE, nonce=nonce_c.hex(), server_version="0.1.0"),
                lock,
            )
            hello = await asyncio.wait_for(P.read_frame(reader), 5)
            self.first_messages.append(hello.type)
            validate_message(hello)
            h = hello.header
            status, secret = self.registry.lookup(h["worker_id"])
            nonce_w = A.nonce_from_hex(h["nonce_w"])
            if status is LookupStatus.REVOKED:
                reason = "revoked"
            elif secret is None or not A.verify_worker_auth(
                secret, nonce_c, nonce_w, h["worker_id"], self.cert.fingerprint, h["auth"]
            ):
                reason = "auth"
            else:
                reason = ""
            if reason or secret is None:
                self.rejections.append(reason or "auth")
                await P.write_frame(writer, P.make_message(P.REJECT, reason=reason or "auth"), lock)
                return
            proof = A.coordinator_proof(
                self.proof_secret_override or secret,
                nonce_c,
                nonce_w,
                h["worker_id"],
                self.cert.fingerprint,
            )
            await P.write_frame(
                writer,
                P.make_message(
                    P.WELCOME,
                    session_id=f"s_{self.connections}",
                    ping_interval_s=1.0,
                    accepted_slots=self.accepted_slots,
                    proof=proof,
                    limits=self.limits.to_dict(),
                ),
                lock,
            )
            session = StubSession(reader, writer, h)
            session.lock = lock
            self.all_sessions.append(session)
            handed_over = True
            await self.sessions.put(session)
        except (P.WireError, OSError, ssl.SSLError, asyncio.TimeoutError):
            pass
        finally:
            if not handed_over:
                writer.close()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(writer.wait_closed(), 2)
            if current is not None:
                self._handlers.discard(current)


async def _manual_demo() -> None:  # pragma: no cover - interactive helper
    tmp = Path(tempfile.mkdtemp(prefix="mdc-stub-"))
    cert = generate_certificate(tmp / "state", 30)
    registry = DeviceRegistry(tmp / "state")
    async with StubCoordinator(cert, registry, port=50099) as stub:
        code = registry.enroll("stub-worker", "127.0.0.1", stub.port, cert.fingerprint)
        join = write_join_file(tmp / "stub-worker.join", code)
        print(
            f"Start a worker in another terminal:\n  python -m mdc.worker --join-file {join} "
            f"--coordinator 127.0.0.1:{stub.port}\n"
        )
        session = await stub.next_session(timeout=600)
        print(f"worker authenticated: {session.hello['name']} slots={session.hello['slots']}")
        await session.send(
            P.make_message(
                P.TASK_ASSIGN,
                task_id="long",
                attempt_id="att_long",
                kind="synthetic",
                params={"seed": 1, "chain_len": 3_000_000},
            )
        )
        t_assign = time.monotonic()
        for seq in range(30):
            t0 = time.monotonic()
            await session.send(P.make_message(P.PING, seq=seq, t_c=t0))
            pong = await session.recv(P.PONG)
            print(
                f"PONG {seq}: {1000 * (time.monotonic() - t0):6.1f} ms  "
                f"active={pong.header['active_tasks']}"
            )
            for msg in session.drain():
                if msg.type == P.TASK_RESULT:
                    print(
                        f"TASK_RESULT after {time.monotonic() - t_assign:.2f} s: "
                        f"{msg.header['json_result']}"
                    )
            await asyncio.sleep(0.2)


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(_manual_demo())
