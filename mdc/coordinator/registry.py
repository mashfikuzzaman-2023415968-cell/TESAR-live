"""Worker records and the worker state machine (spec 7.1).

::

    CONNECTING --HELLO ok--> PROBING --probe done/timeout--> ACTIVE
    ACTIVE --no pong > suspect_after_s--> SUSPECT --pong--> ACTIVE
    SUSPECT --no pong > dead_after_s--> DEAD (reassign in-flight)
    any --TCP closed--> DEAD (reassign in-flight)
    ACTIVE/SUSPECT --strikes >= threshold--> QUARANTINED (never scheduled again)
    any --BYE--> LEFT
    DEAD/LEFT --same worker_id reconnects--> CONNECTING (new session, same record)

Records are keyed by ``worker_id`` (the enrolled identity), not by session:
a phone that drops off WiFi and comes back keeps its history (speed
estimates now, reputation from Phase 7). QUARANTINED is sticky for the
coordinator's lifetime, even across reconnects (spec S.8.6).

Every transition is logged as a ``worker_state`` event.
"""

from __future__ import annotations

import enum
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from mdc.common.events import EventLogger
from mdc.common.ewma import EWMA
from mdc.common.timeutil import monotonic
from mdc.coordinator.job import Attempt

SPEED_EWMA_ALPHA = 0.3  # spec 8.6
BASE_ERROR_COOLDOWN_S = 0.25
MAX_ERROR_COOLDOWN_S = 10.0


class WorkerState(enum.Enum):
    """Worker life-cycle states."""

    CONNECTING = "CONNECTING"
    PROBING = "PROBING"
    ACTIVE = "ACTIVE"
    SUSPECT = "SUSPECT"
    DEAD = "DEAD"
    QUARANTINED = "QUARANTINED"
    LEFT = "LEFT"


# Allowed transitions; anything else is a programming error.
_TRANSITIONS: dict[WorkerState, frozenset[WorkerState]] = {
    WorkerState.CONNECTING: frozenset(
        {WorkerState.PROBING, WorkerState.DEAD, WorkerState.LEFT, WorkerState.QUARANTINED}
    ),
    WorkerState.PROBING: frozenset(
        {WorkerState.ACTIVE, WorkerState.DEAD, WorkerState.LEFT, WorkerState.QUARANTINED}
    ),
    WorkerState.ACTIVE: frozenset(
        {WorkerState.SUSPECT, WorkerState.DEAD, WorkerState.LEFT, WorkerState.QUARANTINED}
    ),
    WorkerState.SUSPECT: frozenset(
        {WorkerState.ACTIVE, WorkerState.DEAD, WorkerState.LEFT, WorkerState.QUARANTINED}
    ),
    WorkerState.DEAD: frozenset({WorkerState.CONNECTING, WorkerState.QUARANTINED}),
    WorkerState.LEFT: frozenset({WorkerState.CONNECTING, WorkerState.QUARANTINED}),
    WorkerState.QUARANTINED: frozenset(),  # sticky
}


@dataclass
class WorkerRecord:
    """Everything the coordinator knows about one worker (field names per spec 7.1)."""

    worker_id: str
    name: str
    platform: str = "linux"
    device_class: str = "laptop"
    trusted: bool = False
    slots: int = 1
    session_id: str | None = None
    state: WorkerState = WorkerState.CONNECTING
    last_pong_t: float = 0.0
    bench_ms_per_unit: float = 1.0
    battery: dict[str, Any] = field(default_factory=dict)
    inflight: dict[str, Attempt] = field(default_factory=dict)
    completed: int = 0
    failed: int = 0
    wrong: int = 0
    strikes: int = 0
    verified_ok: int = 0
    energy_spent_j: float = 0.0
    energy_measured_j: float = 0.0
    joined_t: float = 0.0
    # Additional (not in the spec list):
    speed: dict[str, EWMA] = field(default_factory=dict)  # kind -> ms per work unit
    tiles_done: int = 0  # accepted task results (stage-1 tiles and stage-2 rows)
    attempts_sent: int = 0
    chunk_tiles_sent: int = 0
    battery_start: float | None = None
    consecutive_errors: int = 0
    probe_id: str | None = None  # bandwidth probe in flight (PROBING state)
    probe_sent_t: float | None = None
    cooldown_until: float = 0.0  # not scheduled before this time (error backoff)

    def ms_per_unit(self, kind: str) -> float:
        """Measured ms per work unit for ``kind``, else the benchmark value."""
        estimate = self.speed.get(kind)
        return estimate.value_or(self.bench_ms_per_unit) if estimate else self.bench_ms_per_unit

    def observe_speed(self, kind: str, compute_ms: float, work_units: float) -> None:
        """Refine the speed estimate from one completed attempt (EWMA, alpha 0.3)."""
        if work_units <= 0 or compute_ms <= 0:
            return
        self.speed.setdefault(kind, EWMA(SPEED_EWMA_ALPHA)).update(compute_ms / work_units)

    def note_error(self, now: float) -> float:
        """An error result: back off exponentially (0.25 s, 0.5 s, ... up to 10 s).

        Without this, a worker that keeps refusing (``battery_low``, ``busy``) would
        be sent a new task the instant each refusal arrives — a hot loop of
        thousands of attempts per minute. Returns the cooldown length.
        """
        self.consecutive_errors += 1
        exponent = min(self.consecutive_errors - 1, 16)
        delay = min(MAX_ERROR_COOLDOWN_S, BASE_ERROR_COOLDOWN_S * float(2**exponent))
        self.cooldown_until = now + delay
        return delay

    def note_success(self) -> None:
        """An accepted result ends any error backoff."""
        self.consecutive_errors = 0
        self.cooldown_until = 0.0

    @property
    def connected(self) -> bool:
        """True while a session is open (PROBING, ACTIVE or SUSPECT)."""
        return self.state in {WorkerState.PROBING, WorkerState.ACTIVE, WorkerState.SUSPECT}

    def snapshot(self) -> dict[str, Any]:
        """JSON-friendly view (dashboard, snapshot.json, report)."""
        return {
            "worker_id": self.worker_id,
            "name": self.name,
            "platform": self.platform,
            "device_class": self.device_class,
            "trusted": self.trusted,
            "slots": self.slots,
            "state": self.state.value,
            "bench_ms_per_unit": self.bench_ms_per_unit,
            "ms_per_unit": {k: e.value for k, e in self.speed.items()},
            "battery": dict(self.battery),
            "inflight": len(self.inflight),
            "completed": self.completed,
            "failed": self.failed,
            "wrong": self.wrong,
            "strikes": self.strikes,
            "verified_ok": self.verified_ok,
            "tiles_done": self.tiles_done,
            "energy_spent_j": self.energy_spent_j,
            "energy_measured_j": self.energy_measured_j,
            "avg_chunk_tiles": (
                self.chunk_tiles_sent / self.attempts_sent if self.attempts_sent else None
            ),
        }


class Registry:
    """All worker records of this coordinator run, keyed by ``worker_id``."""

    def __init__(self, events: EventLogger) -> None:
        self.events = events
        self._records: dict[str, WorkerRecord] = {}

    def __iter__(self) -> Iterator[WorkerRecord]:
        return iter(list(self._records.values()))

    def __len__(self) -> int:
        return len(self._records)

    def get(self, worker_id: str) -> WorkerRecord | None:
        """The record of ``worker_id``, if it ever connected."""
        return self._records.get(worker_id)

    def upsert(self, worker_id: str, name: str) -> WorkerRecord:
        """Return the existing record (reconnect) or create a new one."""
        record = self._records.get(worker_id)
        if record is None:
            record = WorkerRecord(worker_id=worker_id, name=name, joined_t=monotonic())
            self._records[worker_id] = record
        else:
            record.name = name
        return record

    def set_state(self, record: WorkerRecord, new: WorkerState, reason: str = "") -> bool:
        """Apply a transition; returns False (no change) if it is not allowed.

        Disallowed transitions are refused rather than raised: e.g. a late
        TCP-close for an already QUARANTINED worker must not un-quarantine it.
        """
        old = record.state
        if new is old:
            return False
        if new not in _TRANSITIONS[old]:
            return False
        record.state = new
        self.events.emit(
            "worker_state",
            worker_id=record.worker_id,
            name=record.name,
            old=old.value,
            new=new.value,
            reason=reason,
        )
        return True

    def active(self) -> list[WorkerRecord]:
        """Workers that may receive new tasks (ACTIVE only; SUSPECT gets none)."""
        return [r for r in self._records.values() if r.state is WorkerState.ACTIVE]

    def connected(self) -> list[WorkerRecord]:
        """Workers with an open session."""
        return [r for r in self._records.values() if r.connected]

    def remote_active(self) -> list[WorkerRecord]:
        """ACTIVE workers other than the trusted local one."""
        return [r for r in self.active() if not r.trusted]
