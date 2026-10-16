"""Security audit log: ``state_dir/logs/security.jsonl`` (spec S.10.4).

One JSON object per line, e.g.::

    {"wall": "...", "type": "auth_failed", "peer": "192.168.0.23", "reason": "bad_mac"}

* Only event types from :data:`SECURITY_EVENTS` are accepted (a typo is a bug).
* Every value is redacted and every string sanitized (no control characters,
  max 256 chars), because many fields come from untrusted peers.
* The file rotates (5 MB x 5) so an attacker cannot fill the disk; every
  file is created with mode 0600.
* Per-type rate limiting (default 20 events/s): during a connection flood the
  log records a count of suppressed events instead of millions of lines.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mdc.common.events import to_jsonable
from mdc.common.logging_setup import PrivateRotatingFileHandler
from mdc.common.redact import redact
from mdc.common.safe_names import sanitize_for_log
from mdc.common.statedir import ensure_private_subdir
from mdc.common.timeutil import monotonic, wall_iso

SECURITY_EVENTS = frozenset(
    {
        "auth_failed",
        "ip_banned",
        "pin_mismatch",
        "coordinator_proof_failed",
        "frame_rejected",
        "validation_failed",
        "connection_limit",
        "result_from_wrong_worker",
        "unsolicited_result",
        "trusted_local_spoof",
        "device_enrolled",
        "device_revoked",
        "device_secret_rotated",
        "device_revoked_session_closed",
        "certificate_rotated",
        "worker_quarantined",
        "wrong_result",  # Phase 10: a verification proved a result wrong (a strike)
        "result_from_quarantined",  # Phase 10: ignored result from a quarantined worker
        "dashboard_login_failed",
        "csrf_rejected",
        "host_rejected",
        "insecure_mode_enabled",
        "control_ignored",
    }
)


RESERVED_FIELDS = frozenset({"type", "wall", "suppressed_before"})


def _sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_for_log(value)
    if isinstance(value, dict):
        return {sanitize_for_log(str(k), 64): _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value[:50]]
    return value


class SecurityLog:
    """Append-only, rotated, rate-limited JSONL audit log.

    Args:
        path: log file, or ``None`` for an in-memory log (tests).
        max_per_second: per event type; extra events are counted, not written.
        clock: monotonic clock (injectable for tests).
    """

    def __init__(
        self,
        path: str | Path | None,
        max_per_second: int = 20,
        max_bytes: int = 5 * 1024 * 1024,
        backups: int = 5,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._lock = threading.Lock()
        self._clock = clock
        self._max_per_second = max_per_second
        self._window: dict[str, tuple[float, int]] = {}
        self._suppressed: dict[str, int] = {}
        self.records: list[dict[str, Any]] = []  # in-memory copy (bounded)
        self._handler: logging.Handler | None = None
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._handler = PrivateRotatingFileHandler(
                path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8"
            )

    @classmethod
    def for_state_dir(cls, state_dir: str | Path, filename: str = "security.jsonl") -> SecurityLog:
        """The standard log at ``state_dir/logs/<filename>`` (directory 0700)."""
        logs = ensure_private_subdir(state_dir, "logs")
        return cls(logs / filename)

    def _allow(self, event_type: str) -> tuple[bool, int]:
        """Rate limit per type with a 1-second fixed window. Returns (allowed, suppressed)."""
        now = self._clock()
        start, count = self._window.get(event_type, (now, 0))
        if now - start >= 1.0:
            start, count = now, 0
        if count >= self._max_per_second:
            self._window[event_type] = (start, count)
            self._suppressed[event_type] = self._suppressed.get(event_type, 0) + 1
            return False, 0
        self._window[event_type] = (start, count + 1)
        return True, self._suppressed.pop(event_type, 0)

    def emit(self, event_type: str, **fields: Any) -> dict[str, Any] | None:
        """Record a security event; returns the written record (``None`` if rate-limited)."""
        if event_type not in SECURITY_EVENTS:
            raise ValueError(f"unknown security event type {event_type!r}")
        clash = RESERVED_FIELDS.intersection(fields)
        if clash:
            # A field named "type" would overwrite the event type — and such values
            # often come from the peer, which would let an attacker forge audit records.
            raise ValueError(f"security event fields may not be named {sorted(clash)}")
        with self._lock:
            allowed, suppressed = self._allow(event_type)
            if not allowed:
                return None
            record: dict[str, Any] = {"wall": wall_iso(), "type": event_type}
            record.update(_sanitize(redact(to_jsonable(fields))))
            if suppressed:
                record["suppressed_before"] = suppressed
            line = json.dumps(record, separators=(",", ":"), allow_nan=False)
            if self._handler is not None:
                self._handler.emit(logging.makeLogRecord({"msg": line, "levelno": logging.INFO}))
            self.records.append(record)
            del self.records[:-1000]
            return record

    def close(self) -> None:
        """Close the file."""
        with self._lock:
            if self._handler is not None:
                self._handler.close()
                self._handler = None


def read_security_log(path: str | Path) -> list[dict[str, Any]]:
    """Read a security log file (tests and analysis)."""
    out: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out
