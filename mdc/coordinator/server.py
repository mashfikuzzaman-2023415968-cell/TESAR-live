"""TLS server: accept, limit, authenticate, serve (spec Phase 4 item 2, S.6, S.8).

Per connection::

    accept (TLS handshake, bounded by handshake_timeout_s)
      -> banned IP?            close (connection_limit, reason=banned)
      -> too many connections? close (connection_limit)            [S.6.1]
      -> handshake within the remaining handshake_timeout_s:        [S.6.2]
           CHALLENGE -> HELLO (validated) -> lookup -> verify MAC -> trusted-local
           rules -> WELCOME
         any failure: REJECT{reason}, close, count a failure for the IP
         (limit failures in a window -> temporary ban)               [S.6.3]
      -> reader loop: PONG / PROBE_ACK / TASK_RESULT / BYE, with an idle
         timeout; >10 unexpected messages or any invalid frame -> close
      -> on close: worker DEAD (or LEFT after BYE), in-flight attempts lost

Trusted local worker (S.8.4): only a connection that uses the ephemeral
local credential *and* comes from a loopback address may claim
``trusted_local``. Any other combination is rejected and security-logged as
``trusted_local_spoof``.

Live revocation (S.4.5): every 2 s the device registry is re-read if its file
changed; sessions of devices that were revoked (or whose secret was rotated)
are closed immediately (``device_revoked_session_closed``).

Shutdown closes every connection explicitly: on Python 3.10,
``Server.wait_closed()`` does not wait for connection handlers.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import ipaddress
import logging
import os
import ssl
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mdc.common import auth as A
from mdc.common import protocol as P
from mdc.common.config import Config
from mdc.common.events import EventLogger
from mdc.common.ids import new_id
from mdc.common.ratelimit import FailureTracker, SlidingCounter
from mdc.common.safe_names import sanitize_for_log
from mdc.common.timeutil import monotonic
from mdc.common.validation import (
    TaskLimits,
    UnknownMessageType,
    ValidationError,
    clamp_hello,
    validate_message,
)
from mdc.coordinator.dispatcher import Dispatcher
from mdc.coordinator.enrollment import DeviceRegistry, LookupStatus
from mdc.coordinator.failure_detector import FailureDetector
from mdc.coordinator.local_worker import LOCAL_NAME, LocalCredential
from mdc.coordinator.netestimator import NetEstimator
from mdc.coordinator.registry import Registry, WorkerRecord, WorkerState
from mdc.coordinator.security_log import SecurityLog

log = logging.getLogger(__name__)

MAX_UNEXPECTED = 10
SERVER_VERSION = "0.1.0"
REVOCATION_CHECK_S = 2.0


def is_loopback(ip: str) -> bool:
    """True for 127.0.0.0/8 and ::1 (including IPv4-mapped IPv6)."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return addr.is_loopback


def trusted_local_decision(
    uses_local_credential: bool, claims_trusted: bool, peer_ip: str
) -> str | None:
    """Apply the trusted-local rules; return a rejection reason or ``None`` if fine.

    * the local credential is only valid from a loopback peer;
    * only the local credential may claim ``trusted_local``;
    * the local credential must claim it (it is never a normal worker).
    """
    if uses_local_credential and not is_loopback(peer_ip):
        return "local credential from non-loopback address"
    if claims_trusted and not uses_local_credential:
        return "trusted_local claimed with an enrolled credential"
    if uses_local_credential and not claims_trusted:
        return "local credential without trusted_local"
    return None


class HandshakeFailed(Exception):
    """The peer failed authentication; ``reason`` is the REJECT reason."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


@dataclass
class Connection:
    """One authenticated worker connection (implements the dispatcher's Channel)."""

    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    peer_ip: str
    worker_id: str = ""
    session_id: str = ""
    secret: bytes = b""
    trusted: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    unsolicited: SlidingCounter = field(default_factory=lambda: SlidingCounter(60.0))
    unexpected: int = 0
    left: bool = False
    closed: bool = False

    async def send(self, msg: P.Message) -> int:
        """Send one frame; returns bytes written."""
        return await P.write_frame(self.writer, msg, self.lock)

    def close(self) -> None:
        """Drop the connection at once (no TLS close_notify wait)."""
        if not self.closed:
            self.closed = True
            self.writer.transport.abort()


@dataclass
class ServerContext:
    """The coordinator components a :class:`Server` works with."""

    config: Config
    fingerprint: str
    ssl_context: ssl.SSLContext | None
    devices: DeviceRegistry
    registry: Registry
    dispatcher: Dispatcher
    detector: FailureDetector
    events: EventLogger
    security_log: SecurityLog
    on_change: Callable[[], None]
    netest: NetEstimator | None = None


class Server:
    """Accepts and serves worker connections."""

    def __init__(
        self,
        ctx: ServerContext,
        local_credential: LocalCredential | None = None,
        revocation_check_s: float = REVOCATION_CHECK_S,
    ) -> None:
        self.cfg = config = ctx.config
        self.fp = ctx.fingerprint
        self.ssl_context = ctx.ssl_context
        self.devices = ctx.devices
        self.local_credential = local_credential
        self.registry = ctx.registry
        self.dispatcher = ctx.dispatcher
        self.detector = ctx.detector
        self.events = ctx.events
        self.security_log = ctx.security_log
        self.on_change = ctx.on_change
        self.netest = ctx.netest
        self.revocation_check_s = revocation_check_s
        self._aux_tasks: set[asyncio.Task[None]] = set()
        sec = config.security
        self.failures = FailureTracker(sec.auth_fail_limit, sec.auth_fail_window_s, sec.auth_ban_s)
        self.connections: dict[str, Connection] = {}  # worker_id -> authenticated connection
        self.port = 0
        self._open_total = 0
        self._open_per_ip: dict[str, int] = {}
        self._server: asyncio.Server | None = None
        self._handlers: set[asyncio.Task[Any]] = set()
        self._writers: set[asyncio.StreamWriter] = set()
        self._watcher: asyncio.Task[None] | None = None

    # ------------------------------------------------------------ lifecycle

    async def start(self, host: str, port: int) -> int:
        """Listen; return the bound port."""
        self._server = await asyncio.start_server(
            self._handle,
            host,
            port,
            ssl=self.ssl_context,
            ssl_handshake_timeout=(
                self.cfg.network.handshake_timeout_s if self.ssl_context is not None else None
            ),
        )
        self.port = self._server.sockets[0].getsockname()[1]
        self._watcher = asyncio.create_task(self._revocation_watcher())
        return self.port

    async def stop(self) -> None:
        """Stop listening and close every connection explicitly."""
        if self._watcher is not None:
            self._watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watcher
        if self._server is not None:
            self._server.close()
        for conn in list(self.connections.values()):
            conn.close()
        for writer in list(self._writers):
            with contextlib.suppress(Exception):
                writer.transport.abort()
        for task in [*self._handlers, *self._aux_tasks]:
            task.cancel()
        for task in [*self._handlers, *self._aux_tasks]:
            with contextlib.suppress(BaseException):
                await task
        if self._server is not None:
            await self._server.wait_closed()
            self._server = None

    def channel_for(self, worker_id: str) -> Connection | None:
        """The live connection of ``worker_id`` (dispatcher hook)."""
        conn = self.connections.get(worker_id)
        return conn if conn is not None and not conn.closed else None

    def close_connection(self, worker_id: str) -> None:
        """Drop ``worker_id``'s connection (failure detector: worker declared DEAD)."""
        conn = self.connections.get(worker_id)
        if conn is not None:
            conn.close()

    def live_connections(self) -> list[Connection]:
        """All authenticated, open connections (failure detector hook)."""
        return [c for c in self.connections.values() if not c.closed]

    # ------------------------------------------------------------- accept

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._handlers.add(task)
        self._writers.add(writer)
        peer = writer.get_extra_info("peername")
        ip = str(peer[0]) if peer else "?"
        counted = False
        try:
            if self.failures.is_banned(ip):
                self.security_log.emit(
                    "connection_limit",
                    peer=ip,
                    reason="banned",
                    ban_remaining_s=round(self.failures.ban_remaining_s(ip)),
                )
                return
            net = self.cfg.network
            if self._open_total >= net.max_connections or (
                self._open_per_ip.get(ip, 0) >= net.max_connections_per_ip
            ):
                record = self.security_log.emit(
                    "connection_limit",
                    peer=ip,
                    open_total=self._open_total,
                    open_from_ip=self._open_per_ip.get(ip, 0),
                )
                if record is not None:  # same rate limit as the security log
                    log.warning(
                        "refused a connection from %s: %d open from that address "
                        "(network.max_connections_per_ip=%d, max_connections=%d). Local "
                        "clusters run every worker on one address: raise the limit.",
                        ip,
                        self._open_per_ip.get(ip, 0),
                        net.max_connections_per_ip,
                        net.max_connections,
                    )
                return
            self._open_total += 1
            self._open_per_ip[ip] = self._open_per_ip.get(ip, 0) + 1
            counted = True
            conn = Connection(reader, writer, ip)
            try:
                await asyncio.wait_for(self._handshake(conn), net.handshake_timeout_s)
            except HandshakeFailed as exc:
                self._auth_failure(conn, exc.reason, exc.detail)
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(
                        conn.send(P.make_message(P.REJECT, reason=exc.reason)), 1.0
                    )
                return
            except asyncio.TimeoutError:
                self._auth_failure(conn, None, "handshake timeout")
                return
            except (P.WireError, OSError, ssl.SSLError) as exc:
                self._auth_failure(conn, None, f"handshake {type(exc).__name__}")
                return
            self._start_probe(conn)
            await self._serve(conn)
        except Exception:
            log.exception("connection handler failed (peer %s)", ip)
        finally:
            if counted:
                self._open_total -= 1
                remaining = self._open_per_ip.get(ip, 1) - 1
                if remaining <= 0:
                    self._open_per_ip.pop(ip, None)
                else:
                    self._open_per_ip[ip] = remaining
            writer.transport.abort()
            self._writers.discard(writer)
            if task is not None:
                self._handlers.discard(task)

    def _auth_failure(self, conn: Connection, reason: str | None, detail: str) -> None:
        """Count a failed handshake for the peer IP; ban it if over the limit."""
        self.security_log.emit(
            "auth_failed",
            peer=conn.peer_ip,
            reason=reason or "handshake",
            detail=detail,
            worker_id=conn.worker_id or None,
        )
        self.events.emit(
            "worker_auth_failed", peer=conn.peer_ip, reason=sanitize_for_log(detail, 128)
        )
        if self.failures.record_failure(conn.peer_ip):
            self.security_log.emit(
                "ip_banned", peer=conn.peer_ip, ban_s=self.cfg.security.auth_ban_s
            )

    # ----------------------------------------------------------- handshake

    async def _handshake(self, conn: Connection) -> None:
        """CHALLENGE/HELLO/WELCOME; registers the worker. Raises HandshakeFailed."""
        nonce_c = A.make_nonce()
        await conn.send(
            P.make_message(P.CHALLENGE, nonce=nonce_c.hex(), server_version=SERVER_VERSION)
        )
        challenge_t = monotonic()
        hello = await P.read_frame(conn.reader, self._limits())
        handshake_rtt = monotonic() - challenge_t  # one round trip (+ an HMAC on the worker)
        if hello.type != P.HELLO:
            raise HandshakeFailed("protocol", f"first message was {hello.type}")
        try:
            validate_message(hello)
        except ValidationError as exc:
            self.security_log.emit(
                "validation_failed", peer=conn.peer_ip, msg_type="HELLO", detail=str(exc)
            )
            raise HandshakeFailed("protocol", "invalid HELLO") from None
        h = hello.header
        worker_id = h["worker_id"]
        conn.worker_id = worker_id
        local = self.local_credential
        uses_local = local is not None and hmac.compare_digest(
            worker_id.encode(), local.worker_id.encode()
        )
        if uses_local and local is not None:
            secret: bytes | None = local.secret
            name = LOCAL_NAME
        else:
            status, secret = self.devices.lookup(worker_id)
            if status is LookupStatus.UNKNOWN and self.devices.reload_if_changed():
                # Enrolled after our last registry read: look again (once).
                status, secret = self.devices.lookup(worker_id)
            if status is LookupStatus.REVOKED:
                raise HandshakeFailed("revoked", "revoked device")
            if status is LookupStatus.UNKNOWN or secret is None:
                raise HandshakeFailed("auth", "unknown worker_id")
            device = next((d for d in self.devices.devices() if d.worker_id == worker_id), None)
            name = device.name if device is not None else worker_id
        if secret is None:  # unreachable by construction; fail closed anyway
            raise HandshakeFailed("auth", "no secret")
        nonce_w = A.nonce_from_hex(h["nonce_w"])
        if not A.verify_worker_auth(secret, nonce_c, nonce_w, worker_id, self.fp, h["auth"]):
            raise HandshakeFailed("auth", "bad MAC")
        spoof = trusted_local_decision(uses_local, bool(h["trusted_local"]), conn.peer_ip)
        if spoof is not None:
            self.security_log.emit(
                "trusted_local_spoof", peer=conn.peer_ip, worker_id=worker_id, detail=spoof
            )
            raise HandshakeFailed("auth", spoof)
        self.failures.record_success(conn.peer_ip)
        conn.secret = secret
        conn.trusted = uses_local
        conn.session_id = new_id("s")
        record = self._register(conn, h, name)
        await conn.send(
            P.make_message(
                P.WELCOME,
                session_id=conn.session_id,
                ping_interval_s=self.cfg.liveness.ping_interval_s,
                accepted_slots=record.slots,
                proof=A.coordinator_proof(secret, nonce_c, nonce_w, worker_id, self.fp),
                limits=self._task_limits().to_dict(),
            )
        )
        if self.netest is not None:
            self.netest.on_rtt(worker_id, handshake_rtt)
        self.events.emit(
            "worker_connected",
            worker_id=worker_id,
            name=record.name,
            platform=record.platform,
            device_class=record.device_class,
            slots=record.slots,
            bench_ms_per_unit=record.bench_ms_per_unit,
            battery=record.battery,
            trusted=record.trusted,
        )
        if not uses_local:
            with contextlib.suppress(Exception):
                self.devices.mark_seen(worker_id)
        self.on_change()

    def _register(self, conn: Connection, h: dict[str, Any], name: str) -> WorkerRecord:
        """Create/update the WorkerRecord for a fresh session (closing any old one)."""
        old = self.connections.get(conn.worker_id)
        if old is not None:
            log.info("worker %s reconnected; closing its previous session", name)
            old.close()
            self._session_ended(old, "replaced by a new session")
        record = self.registry.upsert(conn.worker_id, name)
        clamped = clamp_hello(h)
        record.platform = h["platform"]
        record.device_class = h["device_class"]
        record.trusted = conn.trusted
        record.slots = (
            min(clamped["slots"], self.cfg.workers.local_slots)
            if conn.trusted
            else clamped["slots"]
        )
        record.bench_ms_per_unit = clamped["bench_ms_per_unit"]
        record.battery = clamped["battery"]
        if record.battery_start is None:
            record.battery_start = record.battery.get("pct")
        record.session_id = conn.session_id
        record.last_pong_t = monotonic()
        if record.state is not WorkerState.QUARANTINED:
            self.registry.set_state(record, WorkerState.CONNECTING, "new session")
            self.registry.set_state(record, WorkerState.PROBING, "authenticated")
        self.connections[conn.worker_id] = conn
        return record

    def _limits(self) -> P.FrameLimits:
        net = self.cfg.network
        return P.FrameLimits(net.max_header_bytes, net.max_payload_bytes, net.max_raw_array_bytes)

    def _task_limits(self) -> TaskLimits:
        m = self.cfg.security.max_task_params
        return TaskLimits(m.blur_radius, m.blur_iterations, m.chain_len, m.max_tile_pixels)

    # --------------------------------------------------------------- serve

    async def _serve(self, conn: Connection) -> None:
        reason = "connection closed"
        try:
            while not conn.closed:
                try:
                    msg = await asyncio.wait_for(
                        P.read_frame(conn.reader, self._limits()),
                        self.cfg.network.idle_read_timeout_s,
                    )
                except asyncio.TimeoutError:
                    reason = "idle timeout"
                    return
                try:
                    validate_message(msg)
                except UnknownMessageType:
                    self._unexpected(conn, msg.type)
                    continue
                except ValidationError as exc:
                    self.security_log.emit(
                        "validation_failed",
                        worker_id=conn.worker_id,
                        msg_type=msg.type,
                        detail=str(exc),
                    )
                    reason = "invalid message"
                    return
                if msg.type == P.PONG:
                    rtt = self.detector.on_pong(conn.worker_id, msg.header)
                    if rtt is not None and self.netest is not None:
                        self.netest.on_rtt(conn.worker_id, rtt)
                elif msg.type == P.TASK_RESULT:
                    self.dispatcher.on_result(conn, msg)
                elif msg.type == P.PROBE_ACK:
                    self._on_probe_ack(conn, msg.header)
                elif msg.type == P.BYE:
                    conn.left = True
                    reason = f"bye: {sanitize_for_log(msg.header['reason'], 64)}"
                    return
                else:
                    self._unexpected(conn, msg.type)
        except P.ConnectionClosed:
            reason = "connection closed"
        except P.ProtocolError as exc:
            self.security_log.emit("frame_rejected", worker_id=conn.worker_id, detail=str(exc))
            reason = "protocol error"
        except (OSError, ssl.SSLError, ConnectionError) as exc:
            reason = f"connection error ({type(exc).__name__})"
        finally:
            conn.close()
            self._session_ended(conn, reason)

    # --------------------------------------------------------------- probe

    def _start_probe(self, conn: Connection) -> None:
        """Send the bandwidth PROBE (spec 6.4 step 6) in the background.

        The worker stays PROBING until the PROBE_ACK arrives, or for at most
        ``PROBE_TIMEOUT_S`` (the failure detector then activates it with the
        default bandwidth). Sending in a task lets the reader loop run
        meanwhile — a big probe on a slow link must not starve PONGs.
        """
        record = self.registry.get(conn.worker_id)
        if record is None or record.state is not WorkerState.PROBING:
            return
        probe_bytes = self.cfg.netest.probe_bytes
        record.probe_id = new_id("p")
        record.probe_sent_t = monotonic()
        msg = P.make_message(
            P.PROBE,
            os.urandom(probe_bytes),  # random: compression cannot shrink it
            probe_id=record.probe_id,
            t_c=record.probe_sent_t,
        )
        task = asyncio.create_task(self._send_quietly(conn, msg))
        self._aux_tasks.add(task)
        task.add_done_callback(self._aux_tasks.discard)

    @staticmethod
    async def _send_quietly(conn: Connection, msg: P.Message) -> None:
        with contextlib.suppress(OSError, ssl.SSLError, P.WireError, ConnectionError):
            await conn.send(msg)

    def _on_probe_ack(self, conn: Connection, header: dict[str, Any]) -> None:
        """Bandwidth sample from the probe; PROBING -> ACTIVE."""
        record = self.registry.get(conn.worker_id)
        if record is None or record.probe_id is None or record.probe_sent_t is None:
            return
        if not hmac.compare_digest(str(header["probe_id"]).encode(), record.probe_id.encode()):
            return  # not our current probe: ignore
        # Elapsed time from *our* record of when we sent it — the echoed t_c is
        # not trusted (a worker could fake it to look faster).
        elapsed = monotonic() - record.probe_sent_t
        sample = None
        if self.netest is not None and header["received_bytes"] == self.cfg.netest.probe_bytes:
            sample = self.netest.on_probe_ack(conn.worker_id, self.cfg.netest.probe_bytes, elapsed)
        record.probe_id = None
        self.finish_probe(record, sample, timed_out=False)

    def finish_probe(self, record: WorkerRecord, sample: float | None, timed_out: bool) -> None:
        """End the PROBING phase: the worker becomes ACTIVE (eligible for tasks)."""
        record.probe_id = None
        if record.state is not WorkerState.PROBING:
            return
        self.registry.set_state(
            record, WorkerState.ACTIVE, "probe timeout" if timed_out else "probe done"
        )
        bw = self.netest.bw(record.worker_id) if self.netest is not None else None
        rtt = self.netest.rtt(record.worker_id) if self.netest is not None else None
        self.events.emit(
            "probe_done",
            worker_id=record.worker_id,
            bw_bytes_per_s=bw,
            rtt_s=rtt,
            sample_bytes_per_s=sample,
            timed_out=timed_out,
        )
        self.on_change()

    def _unexpected(self, conn: Connection, msg_type: str) -> None:
        conn.unexpected += 1
        log.info("unexpected %s from %s", sanitize_for_log(msg_type, 32), conn.worker_id)
        if conn.unexpected > MAX_UNEXPECTED:
            conn.close()

    def _session_ended(self, conn: Connection, reason: str) -> None:
        """Clean up after a session: state change, lost attempts, scheduler wake-up."""
        if self.connections.get(conn.worker_id) is not conn:
            return  # already replaced by a newer session
        del self.connections[conn.worker_id]
        record = self.registry.get(conn.worker_id)
        if record is None or record.session_id != conn.session_id:
            return
        new_state = WorkerState.LEFT if conn.left else WorkerState.DEAD
        self.registry.set_state(record, new_state, reason)
        self.dispatcher.on_disconnect(record, reason)
        self.on_change()

    # ---------------------------------------------------------- revocation

    def check_revocations(self) -> int:
        """Close sessions of revoked or re-keyed devices; return how many were closed."""
        with contextlib.suppress(Exception):
            self.devices.reload_if_changed()
        closed = 0
        for conn in list(self.connections.values()):
            if conn.trusted or conn.closed:
                continue
            status, secret = self.devices.lookup(conn.worker_id)
            if (
                status is LookupStatus.OK
                and secret is not None
                and hmac.compare_digest(secret, conn.secret)
            ):
                continue
            self.security_log.emit(
                "device_revoked_session_closed", worker_id=conn.worker_id, status=status.value
            )
            conn.close()
            closed += 1
        return closed

    async def _revocation_watcher(self) -> None:
        while True:
            await asyncio.sleep(self.revocation_check_s)
            self.check_revocations()
