"""Liveness: PING/PONG, SUSPECT/DEAD, attempt timeouts (spec Phase 5 items 1-2, 7.1).

Two loops:

* **ping loop** — every ``liveness.ping_interval_s`` send ``PING{seq, t_c}``
  to every connected worker. RTT samples are timed on the coordinator's own
  clock (spec 4.1 decision 4): the send time of each ``seq`` is remembered and
  the PONG's arrival time is compared with it. The echoed ``t_c`` must match
  what we sent — it is a consistency check only, never trusted for timing (a
  worker could otherwise fake a low RTT to attract work).

* **check loop** — every 0.25 s:

  - ``ACTIVE`` worker silent for more than ``suspect_after_s`` → ``SUSPECT``
    (gets no new tasks, but its in-flight work is *not* yet reassigned: a
    phone that hiccups for two seconds should not lose its work);
  - a PONG brings a SUSPECT worker back to ``ACTIVE``;
  - silent for more than ``dead_after_s`` → ``DEAD``: every in-flight attempt
    is marked ``lost`` (tasks back to READY, ``disconnect`` failures recorded)
    and the TCP connection is **closed**. Closing matters: a frozen phone that
    later resumes must reconnect with a fresh session instead of delivering
    stale state on the old one;
  - attempts older than their ``timeout_s`` are timed out
    (``Dispatcher.sweep_timeouts``);
  - a worker still PROBING ``PROBE_TIMEOUT_S`` after its bandwidth probe was
    sent becomes ACTIVE with the default bandwidth.

A single periodic sweep (rather than one timer per attempt) is simpler and
cheap: it is a pass over at most a few hundred in-flight attempts.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from typing import Any

from mdc.common import protocol as P
from mdc.common.config import Config
from mdc.common.timeutil import monotonic
from mdc.common.validation import clamp_battery
from mdc.coordinator.dispatcher import Dispatcher
from mdc.coordinator.registry import Registry, WorkerRecord, WorkerState

log = logging.getLogger(__name__)

CHECK_INTERVAL_S = 0.25
MAX_SANE_RTT_S = 600.0
PROBE_TIMEOUT_S = 3.0  # spec 6.4 step 6
MAX_PINGS_REMEMBERED = 64  # bounded memory for seq -> send time
ECHO_TOLERANCE_S = 1e-6  # echoed t_c must equal what we sent (JSON float round trip)


class FailureDetector:
    """Sends PINGs, applies the SUSPECT/DEAD state machine, sweeps timeouts."""

    def __init__(
        self,
        config: Config,
        registry: Registry,
        connections: Callable[[], list[Any]],
        dispatcher: Dispatcher | None = None,
        close_connection: Callable[[str], None] | None = None,
        on_change: Callable[[], None] | None = None,
        clock: Callable[[], float] = monotonic,
        finish_probe: Callable[[WorkerRecord, float | None, bool], None] | None = None,
    ) -> None:
        self.cfg = config
        self.registry = registry
        self.connections = connections  # -> objects with .worker_id and async .send()
        self.dispatcher = dispatcher
        self.close_connection = close_connection
        self.on_change = on_change
        self.clock = clock
        self.finish_probe = finish_probe
        self._seq = 0
        self._ping_sent: dict[int, float] = {}
        self._tasks: list[asyncio.Task[None]] = []

    # ---------------------------------------------------------------- pongs

    def on_pong(self, worker_id: str, header: dict[str, Any]) -> float | None:
        """Record a PONG; return the RTT sample in seconds (``None`` if implausible)."""
        now = self.clock()
        record = self.registry.get(worker_id)
        if record is None:
            return None
        record.last_pong_t = now
        record.battery = clamp_battery(header["battery"])
        if self.dispatcher is not None:  # measured energy (spec Phase 8 item 4)
            energy = self.dispatcher.energy
            energy.on_battery_sample(worker_id, record.battery, now)
            record.energy_measured_j = energy.ledger(worker_id).measured_j
        if record.battery_start is None:
            record.battery_start = record.battery.get("pct")
        if record.state is WorkerState.SUSPECT:
            self.registry.set_state(record, WorkerState.ACTIVE, "pong received")
            if self.on_change is not None:
                self.on_change()
        sent = self._ping_sent.get(int(header["seq"]))
        if sent is None or abs(float(header["t_c"]) - sent) > ECHO_TOLERANCE_S:
            return None  # unknown seq or a falsified echo: no RTT sample
        rtt = now - sent
        return rtt if 0.0 <= rtt <= MAX_SANE_RTT_S else None

    # ---------------------------------------------------------------- check

    def check(self, now: float | None = None) -> list[str]:
        """Apply the state machine once; return worker ids declared DEAD."""
        now = self.clock() if now is None else now
        lv = self.cfg.liveness
        dead: list[str] = []
        changed = False
        for record in self.registry:
            if record.state not in {WorkerState.ACTIVE, WorkerState.SUSPECT}:
                continue
            silent = now - record.last_pong_t
            if silent > lv.dead_after_s:
                self.declare_dead(record, f"no pong for {silent:.1f} s")
                dead.append(record.worker_id)
                changed = True
            elif silent > lv.suspect_after_s and record.state is WorkerState.ACTIVE:
                self.registry.set_state(record, WorkerState.SUSPECT, f"no pong for {silent:.1f} s")
                changed = True
        for record in self.registry:
            if (
                record.state is WorkerState.PROBING
                and record.probe_sent_t is not None
                and now - record.probe_sent_t > PROBE_TIMEOUT_S
                and self.finish_probe is not None
            ):
                self.finish_probe(record, None, True)
                changed = True
        if self.dispatcher is not None and self.dispatcher.sweep_timeouts(now):
            changed = True
        if changed and self.on_change is not None:
            self.on_change()
        return dead

    def declare_dead(self, record: WorkerRecord, reason: str) -> None:
        """DEAD: reassign the worker's work and drop its connection."""
        self.registry.set_state(record, WorkerState.DEAD, reason)
        if self.dispatcher is not None:
            self.dispatcher.on_disconnect(record, f"dead: {reason}")
        if self.close_connection is not None:
            self.close_connection(record.worker_id)
        log.warning("worker %s declared DEAD (%s)", record.name, reason)

    # ---------------------------------------------------------------- loops

    def next_ping(self) -> P.Message:
        """A new PING; its send time is remembered for RTT (bounded history)."""
        self._seq += 1
        sent = self.clock()
        self._ping_sent[self._seq] = sent
        while len(self._ping_sent) > MAX_PINGS_REMEMBERED:
            self._ping_sent.pop(next(iter(self._ping_sent)))
        return P.make_message(P.PING, seq=self._seq, t_c=sent)

    async def _ping_all(self) -> None:
        msg = self.next_ping()
        for conn in self.connections():
            with contextlib.suppress(OSError, P.WireError, ConnectionError):
                await conn.send(msg)

    async def _ping_loop(self) -> None:
        while True:
            await asyncio.sleep(self.cfg.liveness.ping_interval_s)
            await self._ping_all()

    async def _check_loop(self) -> None:
        while True:
            await asyncio.sleep(CHECK_INTERVAL_S)
            try:
                self.check()
            except Exception:
                log.exception("liveness check failed")

    def start(self) -> None:
        """Start both loops."""
        if not self._tasks:
            self._tasks = [
                asyncio.create_task(self._ping_loop()),
                asyncio.create_task(self._check_loop()),
            ]

    async def stop(self) -> None:
        """Stop both loops."""
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks = []
