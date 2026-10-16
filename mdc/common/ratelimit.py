"""Rate limiting: token buckets and per-IP failure tracking with bans (spec S.6.3).

Two tools:

* :class:`TokenBucket` — allows bursts up to ``capacity`` and refills at
  ``rate`` tokens per second. Used for things like "at most 10 dashboard logins
  per minute" or "at most N log lines per second".

* :class:`FailureTracker` — counts authentication failures per source IP in
  a sliding window. When an IP reaches ``limit`` failures within
  ``window_s`` seconds it is **banned** for ``ban_s`` seconds: its new
  connections are closed immediately. This turns an online password-guessing
  or flooding attack into a slow trickle. Memory is bounded: at most
  ``max_entries`` IPs are remembered (least recently used are forgotten), so
  an attacker spraying from many addresses cannot exhaust memory.

Clocks are injectable (``clock=``) so the logic is testable without sleeping;
the default is ``time.monotonic`` (never affected by wall-clock changes).
"""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from collections.abc import Callable

Clock = Callable[[], float]


class TokenBucket:
    """Classic token bucket.

    Args:
        rate: tokens added per second (> 0).
        capacity: maximum stored tokens, i.e. the largest burst (>= 1).
    """

    def __init__(self, rate: float, capacity: float, clock: Clock = time.monotonic) -> None:
        if rate <= 0 or capacity < 1:
            raise ValueError("rate must be > 0 and capacity >= 1")
        self.rate = float(rate)
        self.capacity = float(capacity)
        self._clock = clock
        self._tokens = float(capacity)
        self._last = clock()

    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._last)
        self._last = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)

    def allow(self, cost: float = 1.0) -> bool:
        """Take ``cost`` tokens if available; return whether the action is allowed."""
        if cost <= 0:
            raise ValueError("cost must be > 0")
        self._refill()
        if self._tokens >= cost:
            self._tokens -= cost
            return True
        return False

    @property
    def tokens(self) -> float:
        """Tokens currently available (after refilling)."""
        self._refill()
        return self._tokens


class _Entry:
    """Per-IP state: recent failure times and ban expiry."""

    __slots__ = ("banned_until", "failures")

    def __init__(self) -> None:
        self.failures: deque[float] = deque()
        self.banned_until = 0.0


class FailureTracker:
    """Sliding-window failure counter per key (IP) with temporary bans.

    Args:
        limit: failures within ``window_s`` that trigger a ban (>= 1).
        window_s: sliding window length in seconds (> 0).
        ban_s: ban duration in seconds (>= 0; 0 disables bans).
        max_entries: LRU bound on remembered keys.
    """

    def __init__(
        self,
        limit: int,
        window_s: float,
        ban_s: float,
        max_entries: int = 1024,
        clock: Clock = time.monotonic,
    ) -> None:
        if limit < 1 or window_s <= 0 or ban_s < 0 or max_entries < 1:
            raise ValueError("invalid failure-tracker parameters")
        self.limit = limit
        self.window_s = float(window_s)
        self.ban_s = float(ban_s)
        self.max_entries = max_entries
        self._clock = clock
        self._entries: OrderedDict[str, _Entry] = OrderedDict()

    def _entry(self, key: str) -> _Entry:
        entry = self._entries.get(key)
        if entry is None:
            entry = _Entry()
            self._entries[key] = entry
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)  # forget the least recently used
        else:
            self._entries.move_to_end(key)
        return entry

    def _prune(self, entry: _Entry, now: float) -> None:
        while entry.failures and now - entry.failures[0] > self.window_s:
            entry.failures.popleft()

    def is_banned(self, key: str) -> bool:
        """True while ``key`` is serving a ban."""
        entry = self._entries.get(key)
        return entry is not None and self._clock() < entry.banned_until

    def ban_remaining_s(self, key: str) -> float:
        """Seconds left on ``key``'s ban (0 if not banned)."""
        entry = self._entries.get(key)
        if entry is None:
            return 0.0
        return max(0.0, entry.banned_until - self._clock())

    def record_failure(self, key: str) -> bool:
        """Count one failure. Returns True if this failure *started* a ban."""
        now = self._clock()
        entry = self._entry(key)
        self._prune(entry, now)
        entry.failures.append(now)
        if self.ban_s > 0 and len(entry.failures) >= self.limit and now >= entry.banned_until:
            entry.banned_until = now + self.ban_s
            entry.failures.clear()
            return True
        return False

    def record_success(self, key: str) -> None:
        """A successful authentication clears the key's failure history."""
        entry = self._entries.get(key)
        if entry is not None:
            entry.failures.clear()

    def failures(self, key: str) -> int:
        """Failures of ``key`` inside the current window."""
        entry = self._entries.get(key)
        if entry is None:
            return 0
        self._prune(entry, self._clock())
        return len(entry.failures)

    def __len__(self) -> int:
        return len(self._entries)


class SlidingCounter:
    """Counts events in a sliding window (e.g. unsolicited results per minute)."""

    def __init__(self, window_s: float, clock: Clock = time.monotonic) -> None:
        if window_s <= 0:
            raise ValueError("window must be > 0")
        self.window_s = float(window_s)
        self._clock = clock
        self._times: deque[float] = deque()

    def add(self) -> int:
        """Record one event; return how many events are in the window now."""
        now = self._clock()
        self._times.append(now)
        while self._times and now - self._times[0] > self.window_s:
            self._times.popleft()
        return len(self._times)
