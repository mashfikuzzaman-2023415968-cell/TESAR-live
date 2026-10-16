"""Dashboard access control: admin token, sessions, CSRF, Host and Origin checks (spec S.9).

* **Admin token** — 32 random bytes (``secrets``), created on first run in
  ``state_dir/admin_token`` with mode 0600; a file readable by others is
  refused. Only someone who can read the coordinator's state directory can log
  in.
* **Login** — ``POST /login`` with the token, compared with
  ``hmac.compare_digest``, at most ``login_rate_limit_per_min`` attempts per
  client address per minute.
* **Sessions** — server-side, random 32-byte ids, an expiry of
  ``session_ttl_s``, at most :data:`MAX_SESSIONS` (oldest evicted). The cookie
  is ``HttpOnly``, ``SameSite=Strict`` and ``Secure`` on HTTPS.
* **CSRF** — every state-changing request carries the session's CSRF token in
  ``X-MDC-CSRF`` and an ``Origin`` equal to the dashboard's own origin.
* **DNS rebinding** — the ``Host`` header must name the dashboard itself
  (configured host, ``localhost`` or ``127.0.0.1``, with the port).

Pure functions and small classes, no aiohttp here, so they are easy to test.
"""

from __future__ import annotations

import hmac
import ipaddress
import secrets
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

from mdc.common.ratelimit import SlidingCounter
from mdc.common.statedir import read_private_file, write_private_file

TOKEN_BYTES = 32
ADMIN_FILE = "admin_token"  # name of the file holding the admin token
SESSION_COOKIE = "mdc_session"
CSRF_HEADER = "X-MDC-CSRF"
MAX_SESSIONS = 64
MAX_TOKEN_CHARS = 256
MAX_LIMITER_KEYS = 4096
LOOPBACK_NAMES = frozenset({"localhost", "127.0.0.1", "::1"})


def admin_token_path(state_dir: str | Path) -> Path:
    """Where the admin token lives (``state_dir/admin_token``)."""
    return Path(state_dir).expanduser() / ADMIN_FILE


def load_or_create_admin_token(state_dir: str | Path) -> str:
    """The dashboard admin token (created with mode 0600 on first use)."""
    path = admin_token_path(state_dir)
    if not path.exists():
        write_private_file(path, secrets.token_urlsafe(TOKEN_BYTES) + "\n")
    return read_private_file(path).decode("utf-8").strip()


def token_matches(given: str, expected: str) -> bool:
    """Constant-time comparison of a submitted token (bounded length, ASCII only)."""
    if not given or len(given) > MAX_TOKEN_CHARS or not given.isascii():
        return False
    return hmac.compare_digest(given.encode("ascii"), expected.encode("ascii"))


def is_loopback_host(host: str) -> bool:
    """``localhost`` or a loopback IP address."""
    if host in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class Session:
    """One logged-in browser."""

    session_id: str
    csrf: str
    expires: float


class SessionStore:
    """Server-side sessions with expiry and a size bound."""

    def __init__(self, ttl_s: float, clock: Callable[[], float] = monotonic) -> None:
        self.ttl_s = ttl_s
        self.clock = clock
        self._sessions: OrderedDict[str, Session] = OrderedDict()

    def create(self) -> Session:
        """A fresh session (evicts the oldest beyond :data:`MAX_SESSIONS`)."""
        session = Session(
            secrets.token_urlsafe(TOKEN_BYTES),
            secrets.token_urlsafe(TOKEN_BYTES),
            self.clock() + self.ttl_s,
        )
        self._sessions[session.session_id] = session
        while len(self._sessions) > MAX_SESSIONS:
            self._sessions.popitem(last=False)
        return session

    def get(self, session_id: str | None) -> Session | None:
        """The valid session for a cookie value, or ``None`` (expired ones are dropped)."""
        if not session_id or len(session_id) > MAX_TOKEN_CHARS:
            return None
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if self.clock() >= session.expires:
            del self._sessions[session_id]
            return None
        return session

    def delete(self, session_id: str | None) -> None:
        """Log out."""
        if session_id:
            self._sessions.pop(session_id, None)

    def __len__(self) -> int:
        return len(self._sessions)


def csrf_matches(header: str | None, session: Session) -> bool:
    """The request's ``X-MDC-CSRF`` equals the session's token."""
    if not header or len(header) > MAX_TOKEN_CHARS or not header.isascii():
        return False
    return hmac.compare_digest(header.encode("ascii"), session.csrf.encode("ascii"))


class LoginLimiter:
    """At most ``per_minute`` login attempts per client address."""

    def __init__(self, per_minute: int, clock: Callable[[], float] = monotonic) -> None:
        self.per_minute = per_minute
        self.clock = clock
        self._counters: OrderedDict[str, SlidingCounter] = OrderedDict()

    def allow(self, client: str) -> bool:
        """Record an attempt; False once the client exceeded its budget."""
        counter = self._counters.get(client)
        if counter is None:
            counter = SlidingCounter(60.0, self.clock)
            self._counters[client] = counter
            while len(self._counters) > MAX_LIMITER_KEYS:
                self._counters.popitem(last=False)
        return counter.add() <= self.per_minute


def allowed_hosts(host: str, port: int) -> frozenset[str]:
    """``Host`` header values the dashboard answers to (DNS-rebinding defence)."""
    names = {host.lower(), "localhost", "127.0.0.1"}
    out = set()
    for name in names:
        shown = f"[{name}]" if ":" in name else name
        out.add(f"{shown}:{port}")
        if port in {80, 443}:
            out.add(shown)
    return frozenset(out)


def host_ok(host_header: str | None, allowed: frozenset[str]) -> bool:
    """The request's ``Host`` is one of ours."""
    return host_header is not None and host_header.lower() in allowed


def origin_ok(origin: str | None, scheme: str, allowed: frozenset[str]) -> bool:
    """``Origin`` is exactly ``scheme://<allowed host>``."""
    if not origin:
        return False
    prefix = f"{scheme}://"
    return origin.startswith(prefix) and origin[len(prefix) :].lower() in allowed
