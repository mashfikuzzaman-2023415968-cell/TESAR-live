"""The worker: connect, authenticate, serve tasks, reconnect (spec Phase 3, S.7).

Life of a worker::

    run():  loop until stopped
              resolve target   (--coordinator, last good address, UDP beacon, join code)
              _session():      pinned TLS -> CHALLENGE/HELLO/WELCOME -> Session.serve()
              on disconnect:   back off 1, 2, 4, ... 15 s (or the flaky-fault delay)

    Session.serve(): one reader loop + background tasks
              reader loop  - reads frames, validates, dispatches; never computes
              battery task - refreshes the cached battery reading
              task coroutines - one per TASK_ASSIGN, limited by a slot semaphore,
                                compute in the thread pool, then send TASK_RESULT

Why the reader loop must never block: the coordinator decides a worker is
dead when PONGs stop. Computation runs in pool threads; the reader only
parses small headers and writes small replies, so PINGs are answered within
milliseconds even while every slot is busy.

What the worker refuses (it protects its owner's phone, spec S.7):

* any task kind outside the registry, parameters above its *own* limits;
* more than ``slots + prefetch + 1`` tasks at once (``error: "busy"``);
* any task while the battery is below ``--min-battery`` and not charging;
* CONTROL messages unless started with ``--lab-mode --allow-demo-control``;
* a coordinator whose certificate is not the pinned one, or that cannot
  prove knowledge of the device secret.

All writes on a connection go through one ``asyncio.Lock`` (``write_frame``).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import ssl
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

import mdc
from mdc.common import auth as A
from mdc.common import protocol as P
from mdc.common.credentials import JoinCode
from mdc.common.safe_names import sanitize_for_log
from mdc.common.timeutil import wait_event
from mdc.common.tls import PinMismatchError, open_pinned_connection
from mdc.common.validation import (
    DEFAULT_TASK_LIMITS,
    TaskLimits,
    UnknownMessageType,
    ValidationError,
    validate_limits,
    validate_message,
)
from mdc.tasks.errors import TaskError
from mdc.worker.battery import BatteryProvider, MockBattery, battery_allows_work
from mdc.worker.discovery_listener import NO_BEACON_HELP, listen_for_beacon
from mdc.worker.executor import TaskExecutor
from mdc.worker.faults import FaultInjector, FaultSettings, corrupt_array, corrupt_json
from mdc.worker.shaper import Shaper

log = logging.getLogger(__name__)
security_log = logging.getLogger("mdc.worker.security")

MAX_UNKNOWN_MESSAGES = 10
MAX_SLOTS = 8
LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
EXIT_OK = 0
EXIT_CRASH = 1
EXIT_REJECTED = 3


class WorkerRejected(Exception):
    """The coordinator refused us for a reason that retrying cannot fix."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class CoordinatorProofFailed(Exception):
    """The WELCOME proof was wrong: the peer does not know our device secret."""


class SessionEnded(Exception):
    """The coordinator said BYE, or a CONTROL 'leave' was received."""


@dataclass
class WorkerOptions:
    """Everything the worker needs besides its credentials."""

    coordinator: tuple[str, int] | None = None
    name: str | None = None
    slots: int = 1
    prefetch: int = 1
    device_class: str = "laptop"
    local: bool = False
    lab_mode: bool = False
    allow_demo_control: bool = False
    min_battery_pct: float = 15.0
    task_max_s: float = 120.0
    handshake_timeout_s: float = 5.0
    idle_timeout_s: float = 30.0
    battery_interval_s: float = 2.0
    discovery_port: int = 50000
    multicast_group: str | None = "239.255.77.77"
    discovery_timeout_s: float = 10.0
    max_backoff_s: float = 15.0
    compression: str = P.COMPRESSION_ZLIB
    insecure_dev: bool = False
    limits: TaskLimits = field(default_factory=lambda: DEFAULT_TASK_LIMITS)


def detect_platform() -> str:
    """``android`` | ``linux`` | ``windows`` | ``darwin`` (the HELLO ``platform`` field)."""
    if "ANDROID_ROOT" in os.environ or "com.termux" in sys.prefix:
        return "android"
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "darwin"
    return "linux"


def stricter_limits(own: TaskLimits, theirs: TaskLimits) -> TaskLimits:
    """Element-wise minimum: the coordinator may tighten our limits, never loosen them."""
    return TaskLimits(
        blur_radius=min(own.blur_radius, theirs.blur_radius),
        blur_iterations=min(own.blur_iterations, theirs.blur_iterations),
        chain_len=min(own.chain_len, theirs.chain_len),
        max_tile_pixels=min(own.max_tile_pixels, theirs.max_tile_pixels),
    )


@dataclass
class TaskState:
    """One assigned attempt on this worker."""

    attempt_id: str
    task_id: str
    kind: str
    params: dict[str, Any]
    data: Any
    t_recv: float  # perf_counter at receipt
    t_recv_w: float  # monotonic at receipt (reported to the coordinator)
    running: bool = False
    cancelled: bool = False
    future: asyncio.Task[None] | None = None


class Worker:
    """A worker process. Create it, then ``await run()``."""

    def __init__(
        self,
        join: JoinCode,
        options: WorkerOptions,
        battery: BatteryProvider,
        faults: FaultInjector | None = None,
        shaper: Shaper | None = None,
        bench_ms_per_unit: float | None = None,
        exit_func: Callable[[int], None] = os._exit,
    ) -> None:
        self.join = join
        self.opts = options
        self.battery = battery
        self.faults = faults or FaultInjector(FaultSettings())
        self.shaper = shaper or Shaper()
        self.exit_func = exit_func
        self.slots = max(1, min(options.slots, MAX_SLOTS))
        self.name = "local" if options.local else (options.name or join.name)
        if bench_ms_per_unit is None:
            from mdc.tasks.benchmark import run_benchmark  # noqa: PLC0415

            bench_ms_per_unit = run_benchmark()
        # An emulated slow device (--slowdown F) also benchmarks F times slower, as a
        # real slow phone would; otherwise the coordinator's first chunks treat it as fast.
        self.bench_ms_per_unit = bench_ms_per_unit * self.faults.settings.slowdown
        self.executor = TaskExecutor(self.slots)
        self.battery_reading: dict[str, Any] = battery.read()
        self.last_target: tuple[str, int] | None = None
        self.sessions_completed = 0
        self.tasks_completed = 0
        self._stop: asyncio.Event | None = None
        self._session: Session | None = None
        self._flaky_pause: float | None = None

    # ---------------------------------------------------------- lifecycle

    def stop(self) -> None:
        """Ask ``run()`` to finish (signal handlers call this)."""
        if self._stop is not None:
            self._stop.set()
        if self._session is not None:
            self._session.close(abort=True)

    @property
    def stopping(self) -> bool:
        """True once ``stop()`` was called."""
        return self._stop is not None and self._stop.is_set()

    async def run(self) -> int:
        """Connect/serve/reconnect until stopped. Returns a process exit code."""
        self._stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        crash_handle = None
        if self.faults.settings.crash_after_s is not None:
            crash_handle = loop.call_later(
                self.faults.settings.crash_after_s, self._crash, "crash-after-s fault"
            )
        backoff = 1.0
        try:
            while not self.stopping:
                target = await self._resolve_target()
                welcomed = False
                try:
                    welcomed = await self._run_session(*target)
                except WorkerRejected as exc:
                    log.error("coordinator rejected this worker: %s", exc.reason)
                    return EXIT_REJECTED
                except PinMismatchError:
                    security_log.warning("SECURITY pin_mismatch peer=%s:%d", *target)
                    self.last_target = None
                except CoordinatorProofFailed:
                    security_log.warning("SECURITY coordinator_proof_failed peer=%s:%d", *target)
                    self.last_target = None
                except (OSError, ssl.SSLError, asyncio.TimeoutError, P.WireError) as exc:
                    log.info("connection to %s:%d ended: %s", *target, type(exc).__name__)
                    if not welcomed:
                        self.last_target = None
                if self.stopping:
                    break
                if welcomed:
                    backoff = 1.0
                delay = self._flaky_pause if self._flaky_pause is not None else backoff
                self._flaky_pause = None
                log.info("reconnecting in %.1f s", delay)
                await wait_event(self._stop, delay)
                backoff = min(backoff * 2, self.opts.max_backoff_s)
            return EXIT_OK
        finally:
            if crash_handle is not None:
                crash_handle.cancel()
            self.executor.shutdown()

    def _crash(self, why: str) -> None:
        log.warning("lab-mode fault: exiting abruptly (%s)", why)
        self.exit_func(EXIT_CRASH)

    async def _resolve_target(self) -> tuple[str, int]:
        """Where to connect: --coordinator, last good address, a beacon, or the join code."""
        if self.opts.coordinator is not None:
            return self.opts.coordinator
        if self.last_target is not None:
            return self.last_target
        beacon = await listen_for_beacon(
            self.join.fp,
            self.opts.discovery_port,
            self.opts.multicast_group,
            self.opts.discovery_timeout_s,
        )
        if beacon is not None:
            log.info("beacon from coordinator %r at %s:%d", beacon.name, beacon.host, beacon.port)
            return (beacon.host, beacon.port)
        log.warning(
            "%s (trying the join code address %s:%d)",
            NO_BEACON_HELP,
            self.join.host,
            self.join.port,
        )
        return (self.join.host, self.join.port)

    # ------------------------------------------------------------ session

    async def _connect(
        self, host: str, port: int
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        if self.opts.insecure_dev:
            if host not in LOOPBACK:
                raise WorkerRejected("--insecure-dev only connects to a loopback coordinator")
            return await asyncio.wait_for(
                asyncio.open_connection(host, port), self.opts.handshake_timeout_s
            )
        return await open_pinned_connection(host, port, self.join.fp, self.opts.handshake_timeout_s)

    async def _run_session(self, host: str, port: int) -> bool:
        """One connection. Returns True if the handshake succeeded."""
        reader, writer = await self._connect(host, port)
        session = Session(self, reader, writer)
        welcomed = False
        try:
            await asyncio.wait_for(session.handshake(), self.opts.handshake_timeout_s)
            welcomed = True
            self.last_target = (host, port)
            log.info(
                "authenticated to %s:%d as %s (%s), session %s",
                host,
                port,
                self.name,
                self.join.worker_id,
                session.session_id,
            )
            self._session = session
            try:
                await session.serve()
            except (OSError, ssl.SSLError, asyncio.TimeoutError, P.WireError) as exc:
                # After a successful handshake a broken connection is routine
                # (WiFi drop, coordinator restart): reconnect to the same place.
                log.info("disconnected from %s:%d: %s", host, port, type(exc).__name__)
        except SessionEnded as exc:
            log.info("session ended: %s", exc)
        finally:
            self._session = None
            await session.shutdown()
            if welcomed:
                self.sessions_completed += 1
        return welcomed

    # ----------------------------------------------------------- messages

    def hello(self, nonce_c: bytes, nonce_w: bytes) -> P.Message:
        """The HELLO message answering ``nonce_c``."""
        return P.make_message(
            P.HELLO,
            worker_id=self.join.worker_id,
            name=self.name,
            platform=detect_platform(),
            device_class=self.opts.device_class,
            cpu_count=max(1, min(os.cpu_count() or 1, 1024)),
            slots=self.slots,
            bench_ms_per_unit=float(self.bench_ms_per_unit),
            battery=self.battery_reading,
            version=mdc.__version__,
            trusted_local=self.opts.local,
            nonce_w=nonce_w.hex(),
            auth=A.worker_auth(
                self.join.secret, nonce_c, nonce_w, self.join.worker_id, self.join.fp
            ),
        )


class Session:
    """State of one authenticated connection."""

    def __init__(
        self, worker: Worker, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self.worker = worker
        self.reader = reader
        self.writer = writer
        self.lock = asyncio.Lock()
        self.session_id = ""
        self.slots = worker.slots
        self.limits = worker.opts.limits
        self.tasks: dict[str, TaskState] = {}
        self.unknown_messages = 0
        self._semaphore = asyncio.Semaphore(self.slots)
        self._background: list[asyncio.Task[None]] = []
        self._closed = False

    @property
    def opts(self) -> WorkerOptions:
        """The worker's options."""
        return self.worker.opts

    # ---------------------------------------------------------- handshake

    async def handshake(self) -> None:
        """CHALLENGE -> HELLO -> WELCOME (spec 6.4), verifying the coordinator's proof."""
        first = validate_message(await P.read_frame(self.reader))
        if first.type == P.REJECT:
            self._raise_reject(first.header["reason"])
        if first.type != P.CHALLENGE:
            raise P.ProtocolError(f"expected CHALLENGE, got {first.type}")
        nonce_c = A.nonce_from_hex(first.header["nonce"])
        nonce_w = A.make_nonce()
        await self.send(self.worker.hello(nonce_c, nonce_w))
        reply = validate_message(await P.read_frame(self.reader))
        if reply.type == P.REJECT:
            self._raise_reject(reply.header["reason"])
        if reply.type != P.WELCOME:
            raise P.ProtocolError(f"expected WELCOME, got {reply.type}")
        join = self.worker.join
        if not A.verify_coordinator_proof(
            join.secret, nonce_c, nonce_w, join.worker_id, join.fp, reply.header["proof"]
        ):
            raise CoordinatorProofFailed
        self.session_id = reply.header["session_id"]
        self.slots = max(1, min(self.worker.slots, int(reply.header["accepted_slots"])))
        self._semaphore = asyncio.Semaphore(self.slots)
        self.limits = stricter_limits(self.opts.limits, validate_limits(reply.header["limits"]))

    @staticmethod
    def _raise_reject(reason: str) -> None:
        if reason == "busy":
            raise ConnectionRefusedError("coordinator busy")
        raise WorkerRejected(reason)

    # -------------------------------------------------------------- serve

    async def serve(self) -> None:
        """Run the reader loop and background tasks until the connection ends."""
        self._background.append(asyncio.create_task(self._battery_loop()))
        flaky_s = self.worker.faults.flaky_session_s()
        if flaky_s is not None:
            self._background.append(asyncio.create_task(self._flaky_disconnect(flaky_s)))
        await self._reader_loop()

    async def _reader_loop(self) -> None:
        while True:
            try:
                msg = await asyncio.wait_for(P.read_frame(self.reader), self.opts.idle_timeout_s)
            except asyncio.TimeoutError:
                log.warning("no message for %.0f s; reconnecting", self.opts.idle_timeout_s)
                return
            try:
                validate_message(msg)
            except UnknownMessageType:
                self._unexpected(msg.type)
                continue
            except ValidationError as exc:
                security_log.warning(
                    "SECURITY validation_failed type=%s detail=%s",
                    msg.type,
                    sanitize_for_log(str(exc)),
                )
                raise
            await self._dispatch(msg)

    def _unexpected(self, msg_type: str) -> None:
        self.unknown_messages += 1
        log.info("ignoring unexpected message type %s", sanitize_for_log(msg_type, 32))
        if self.unknown_messages > MAX_UNKNOWN_MESSAGES:
            raise P.ProtocolError("too many unexpected messages")

    async def _dispatch(self, msg: P.Message) -> None:
        h = msg.header
        log.debug("received %s", msg.type)
        if msg.type == P.PING:
            await self.send(
                P.make_message(
                    P.PONG,
                    seq=h["seq"],
                    t_c=h["t_c"],
                    active_tasks=sum(1 for t in self.tasks.values() if t.running),
                    queued_tasks=sum(1 for t in self.tasks.values() if not t.running),
                    battery=self.worker.battery_reading,
                )
            )
        elif msg.type == P.PROBE:
            self._spawn(self._probe_ack(h["probe_id"], h["t_c"], len(msg.payload)))
        elif msg.type == P.TASK_ASSIGN:
            await self._on_assign(msg)
        elif msg.type == P.TASK_CANCEL:
            self._on_cancel(h["attempt_id"])
        elif msg.type == P.CONTROL:
            await self._on_control(h["action"], h.get("value"))
        elif msg.type == P.BYE:
            raise SessionEnded(f"coordinator said bye ({sanitize_for_log(h['reason'], 64)})")
        else:  # a known type that a coordinator never sends
            self._unexpected(msg.type)

    # -------------------------------------------------------------- sends

    async def send(self, msg: P.Message) -> None:
        """Send one frame (serialized by the connection lock)."""
        await P.write_frame(self.writer, msg, self.lock)
        log.debug("sent %s", msg.type)

    async def _send_shaped(self, msg: P.Message) -> None:
        """Bulk send with emulated network delay/bandwidth (TASK_RESULT, PROBE_ACK)."""
        if self.worker.shaper.enabled:
            size = len(P.encode_frame(msg))
            await asyncio.sleep(self.worker.shaper.delay_s(size))
        await self.send(msg)

    async def _probe_ack(self, probe_id: str, t_c: float, received: int) -> None:
        await self._send_shaped(
            P.make_message(P.PROBE_ACK, probe_id=probe_id, t_c=t_c, received_bytes=received)
        )

    def _spawn(self, coro: Any) -> asyncio.Task[None]:
        task: asyncio.Task[None] = asyncio.create_task(self._guard(coro))
        self._background.append(task)
        task.add_done_callback(
            lambda t: self._background.remove(t) if t in self._background else None
        )
        return task

    async def _guard(self, coro: Any) -> None:
        """Run a send coroutine; a broken connection just ends it quietly."""
        with contextlib.suppress(OSError, ssl.SSLError, P.WireError, ConnectionError):
            await coro

    # -------------------------------------------------------------- tasks

    def _result_message(
        self,
        state: TaskState,
        status: str,
        error: str | None,
        compute_ms: float = 0.0,
        queue_ms: float = 0.0,
        **output: Any,
    ) -> tuple[P.Message, bytes]:
        payload = output.pop("payload", b"")
        msg = P.make_message(
            P.TASK_RESULT,
            payload,
            task_id=state.task_id,
            attempt_id=state.attempt_id,
            status=status,
            error=None if error is None else sanitize_for_log(error, 200),
            compute_ms=round(max(0.0, compute_ms), 3),
            queue_ms=round(max(0.0, queue_ms), 3),
            t_recv_w=state.t_recv_w,
            t_send_w=time.monotonic(),
            **output,
        )
        return msg, payload

    async def _send_error(self, state: TaskState, error: str) -> None:
        msg, _ = self._result_message(state, "error", error)
        await self.send(msg)

    async def _on_assign(self, msg: P.Message) -> None:
        h = msg.header
        faults = self.worker.faults
        state = TaskState(
            attempt_id=h["attempt_id"],
            task_id=h["task_id"],
            kind=h["kind"],
            params=h["params"],
            data=None,
            t_recv=time.perf_counter(),
            t_recv_w=time.monotonic(),
        )
        log.info("task %s (attempt %s, %s)", state.task_id, state.attempt_id, state.kind)
        if faults.should_crash():
            self.worker._crash("crash-prob fault")
            return
        if state.attempt_id in self.tasks:
            log.warning("duplicate attempt id %s ignored", state.attempt_id)
            return
        if len(self.tasks) >= self.slots + self.opts.prefetch + 1:
            await self._send_error(state, "busy")
            return
        if not battery_allows_work(self.worker.battery_reading, self.opts.min_battery_pct):
            await self._send_error(state, "battery_low")
            return
        if faults.should_drop():
            log.info("lab-mode fault: dropping attempt %s", state.attempt_id)
            return
        if "shape" in h:
            try:
                state.data = P.unpack_array(h, msg.payload)
            except P.IntegrityError:
                await self._send_error(state, "integrity")
                return
            except ValidationError as exc:
                await self._send_error(state, f"invalid_task: {exc}")
                return
        self.tasks[state.attempt_id] = state
        state.future = asyncio.create_task(self._run_task(state))

    async def _run_task(self, state: TaskState) -> None:
        worker = self.worker
        try:
            async with self._semaphore:
                if state.cancelled:
                    return
                state.running = True
                queue_ms = (time.perf_counter() - state.t_recv) * 1000.0
                self._update_busy()
                output, compute_s = await worker.executor.run(
                    state.kind, state.params, state.data, self.limits, self.opts.task_max_s
                )
                extra = worker.faults.slowdown_sleep_s(compute_s)
                if extra > 0:
                    await asyncio.sleep(extra)
                compute_ms = (compute_s + extra) * 1000.0
            if state.cancelled:
                log.info("attempt %s cancelled; result discarded", state.attempt_id)
                return
            corrupt = worker.faults.should_corrupt()
            rng = (
                worker.faults.corruption_rng(
                    (state.data.tobytes() if state.data is not None else b"")
                    + json.dumps(state.params, sort_keys=True).encode()
                )
                if corrupt
                else worker.faults.rng
            )
            fields: dict[str, Any]
            if output.array is not None:
                array = output.array
                if corrupt:
                    array = corrupt_array(array, rng)
                packed, payload = P.pack_array(np.ascontiguousarray(array), self.opts.compression)
                fields = {**packed, "payload": payload}
            else:
                result = output.json_result or {}
                if corrupt:
                    result = corrupt_json(state.kind, result, rng)
                fields = {"json_result": result}
            if corrupt:
                log.info("lab-mode fault: corrupted result of %s", state.attempt_id)
            msg, _ = self._result_message(state, "ok", None, compute_ms, queue_ms, **fields)
            await self._send_shaped(msg)
            worker.tasks_completed += 1
        except (TaskError, ValidationError) as exc:
            await self._send_error(state, str(exc))
        except (OSError, ssl.SSLError, P.WireError, ConnectionError):
            pass  # connection gone; the coordinator reassigns the attempt
        except Exception:
            log.exception("task %s failed unexpectedly", state.attempt_id)
            with contextlib.suppress(Exception):
                await self._send_error(state, "internal_error")
        finally:
            state.data = None  # task inputs stay in memory only as long as needed
            self.tasks.pop(state.attempt_id, None)
            self._update_busy()

    def _update_busy(self) -> None:
        battery = self.worker.battery
        if isinstance(battery, MockBattery):
            battery.set_busy(sum(1 for t in self.tasks.values() if t.running))

    def _on_cancel(self, attempt_id: str) -> None:
        state = self.tasks.get(attempt_id)
        if state is None:
            return
        state.cancelled = True
        if not state.running and state.future is not None:
            state.future.cancel()
            self.tasks.pop(attempt_id, None)
        log.info(
            "attempt %s cancelled (%s)",
            attempt_id,
            "result will be discarded" if state.running else "removed from queue",
        )

    async def _on_control(self, action: str, value: float | None) -> None:
        if not (self.opts.lab_mode and self.opts.allow_demo_control):
            security_log.warning("SECURITY control_ignored action=%s", action)
            return
        log.warning("lab-mode CONTROL: %s %s", action, value)
        if action == "crash":
            self.worker._crash("CONTROL crash")
        elif action == "leave":
            await self.send(P.make_message(P.BYE, reason="leaving (CONTROL)"))
            self.worker.stop()
            raise SessionEnded("CONTROL leave")
        else:
            self.worker.faults.apply_control(action, value)

    # --------------------------------------------------------- background

    async def _battery_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await asyncio.sleep(self.opts.battery_interval_s)
            # termux-battery-status can take a second: never on the event loop.
            reading = await loop.run_in_executor(None, self.worker.battery.read)
            self.worker.battery_reading = reading

    async def _flaky_disconnect(self, after_s: float) -> None:
        await asyncio.sleep(after_s)
        pause = self.worker.faults.flaky_reconnect_s()
        log.warning("lab-mode fault: flaky disconnect, back in %.1f s", pause)
        self.worker._flaky_pause = pause
        self.close(abort=True)

    # ----------------------------------------------------------- shutdown

    def close(self, abort: bool = False) -> None:
        """Close the connection (the reader loop then ends).

        ``abort=True`` drops the TCP connection immediately. A graceful TLS
        close sends ``close_notify`` and waits for the peer's reply (asyncio's
        default timeout is 30 s), during which the reader stays blocked — too
        slow for a stop request, and unrealistic for a simulated network drop.
        """
        if self._closed:
            return
        self._closed = True
        if abort:
            self.writer.transport.abort()
        else:
            self.writer.close()

    async def shutdown(self) -> None:
        """Cancel everything belonging to this session and close the socket."""
        current = asyncio.current_task()
        pending = [t for t in self._background if t is not current]
        pending += [s.future for s in self.tasks.values() if s.future is not None]
        for task in pending:
            task.cancel()
        for task in pending:
            with contextlib.suppress(BaseException):
                await task
        self.tasks.clear()
        self.close()
        with contextlib.suppress(OSError, ssl.SSLError, asyncio.TimeoutError):
            await asyncio.wait_for(self.writer.wait_closed(), 2.0)


def with_options(options: WorkerOptions, **changes: Any) -> WorkerOptions:
    """A copy of ``options`` with some fields changed (tests, local mode)."""
    return replace(options, **changes)
