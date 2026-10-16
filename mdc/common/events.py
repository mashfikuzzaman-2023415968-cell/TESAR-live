"""Structured event log (``events.jsonl``) with in-memory subscribers.

Every interesting thing that happens in a run (a worker joins, an attempt is
sent, a result is verified, ...) is emitted as one JSON object per line:

    {"t": 12.345, "wall": "2026-10-07T10:00:00.123+00:00", "type": "attempt_sent", ...}

* ``t``    — seconds since the logger was created/reset (monotonic, i.e. since
             job start). Experiments compute durations from this.
* ``wall`` — human-readable UTC time, for correlating with other logs.
* ``type`` — the event type from the catalogue in spec Section 10.

The same events are pushed to subscribers (the dashboard) and kept in a small
ring buffer of recent events.

Thread/async safety: one ``threading.Lock`` guards the file write and the ring
buffer. Writes are small, so holding a lock briefly is fine even when called
from the asyncio event loop. Subscribers are called **outside** the lock so a
subscriber that itself emits an event cannot deadlock.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

from mdc.common.timeutil import monotonic, wall_iso

log = logging.getLogger(__name__)

EventCallback = Callable[[dict[str, Any]], None]
_TYPE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_RESERVED = frozenset({"t", "wall", "type"})
_MAX_DEPTH = 32  # bound recursion on hostile/accidentally cyclic structures


def to_jsonable(value: Any, _depth: int = 0) -> Any:
    """Convert a value into something ``json.dumps`` accepts, safely.

    * non-finite floats become ``None`` (JSON has no NaN/Infinity and our own
      readers reject them),
    * numpy scalars become Python numbers (duck-typed via ``.item()`` so this
      module does not need numpy),
    * tuples/sets become lists, dict keys become strings,
    * anything else unknown becomes its ``str()``.
    """
    if _depth > _MAX_DEPTH:
        return "<too deep>"
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): to_jsonable(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset, deque)):
        return [to_jsonable(v, _depth + 1) for v in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return to_jsonable(item(), _depth + 1)
        except (TypeError, ValueError):
            pass
    return str(value)


class EventLogger:
    """Writes events to a JSONL file and fans them out to subscribers.

    Args:
        path: JSONL file to append to, or ``None`` for an in-memory logger
            (tests, or before an output directory exists).
        recent_max: how many recent events to keep in memory for the dashboard.
    """

    def __init__(self, path: str | Path | None = None, recent_max: int = 200) -> None:
        self._lock = threading.Lock()
        self._t0 = monotonic()
        self._subscribers: list[EventCallback] = []
        self._recent: deque[dict[str, Any]] = deque(maxlen=recent_max)
        self._file: IO[str] | None = None
        self.path: Path | None = None
        if path is not None:
            self.open(path)

    def open(self, path: str | Path) -> None:
        """Start (or switch to) appending to ``path``; parent dirs are created."""
        new_path = Path(path)
        new_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if self._file is not None:
                self._file.close()
            # Line buffering: each event reaches the OS as soon as it is written,
            # so a crash loses at most the event being written.
            self._file = new_path.open("a", encoding="utf-8", buffering=1)
            self.path = new_path

    def reset_clock(self) -> None:
        """Make ``t`` count from now (called at job start)."""
        with self._lock:
            self._t0 = monotonic()

    @property
    def t0(self) -> float:
        """Monotonic time that ``t`` is measured from."""
        return self._t0

    def now(self) -> float:
        """Current ``t`` value (seconds since the clock was reset)."""
        return monotonic() - self._t0

    def emit(self, type: str, **fields: Any) -> dict[str, Any]:
        """Record one event and return it as a dict.

        Raises:
            ValueError: if ``type`` is not a lowercase identifier, or a field
                tries to overwrite ``t``/``wall``/``type``.
        """
        if not isinstance(type, str) or not _TYPE_RE.fullmatch(type):
            raise ValueError(f"invalid event type {type!r}")
        clash = _RESERVED.intersection(fields)
        if clash:
            raise ValueError(f"event fields may not override {sorted(clash)}")
        event: dict[str, Any] = {"t": round(self.now(), 6), "wall": wall_iso(), "type": type}
        event.update(to_jsonable(fields))
        line = json.dumps(event, separators=(",", ":"), allow_nan=False)
        with self._lock:
            if self._file is not None:
                self._file.write(line + "\n")
            self._recent.append(event)
            subscribers = list(self._subscribers)
        for callback in subscribers:
            try:
                callback(event)
            except Exception:
                # A broken subscriber (e.g. a closed dashboard socket) must never
                # break the coordinator; log it and carry on.
                log.exception("event subscriber %r failed", callback)
        return event

    def subscribe(self, callback: EventCallback) -> None:
        """Call ``callback(event)`` for every future event."""
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: EventCallback) -> None:
        """Stop calling ``callback``; unknown callbacks are ignored."""
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def recent(self, n: int | None = None) -> list[dict[str, Any]]:
        """The most recent events, oldest first (at most ``n``)."""
        with self._lock:
            events = list(self._recent)
        return events if n is None else events[-n:] if n > 0 else []

    def close(self) -> None:
        """Flush and close the file (safe to call twice)."""
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None

    def __enter__(self) -> EventLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_events(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL event file (used by tests, chaos.py and analysis scripts)."""
    events: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                obj = json.loads(stripped)
                if isinstance(obj, dict):
                    events.append(obj)
    return events
