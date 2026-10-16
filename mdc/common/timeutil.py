"""Time helpers.

Rule (spec Section 12): every duration and timeout uses ``time.monotonic()``,
which never jumps when the wall clock is adjusted (NTP, the user changing the
time, phones syncing with the network). Wall-clock time is used only for
human-readable timestamps in logs and reports.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone


def monotonic() -> float:
    """Seconds from an arbitrary fixed point; only differences are meaningful."""
    return time.monotonic()


def elapsed_since(t0: float) -> float:
    """Seconds elapsed since monotonic time ``t0``."""
    return time.monotonic() - t0


def wall_iso() -> str:
    """Current UTC wall-clock time as ISO-8601 with milliseconds."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


async def wait_event(event: asyncio.Event, timeout: float) -> bool:
    """Wait until ``event`` is set or ``timeout`` seconds pass; True if it was set.

    Use this instead of ``asyncio.wait_for(event.wait(), timeout)`` in loops that
    are stopped by cancellation. On Python <= 3.11, ``wait_for`` can swallow a
    cancellation that arrives exactly as its timeout fires (CPython gh-86296),
    so the loop keeps running and whoever awaits the cancelled task hangs —
    which actually happened on Python 3.10 (see DECISIONS.md). ``asyncio.wait``
    never cancels or hides the caller's cancellation.
    """
    waiter = asyncio.ensure_future(event.wait())
    try:
        await asyncio.wait({waiter}, timeout=timeout)
    finally:
        waiter.cancel()
    return event.is_set()
