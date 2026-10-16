"""Measure the peak extra memory used by a piece of code (tracemalloc)."""

from __future__ import annotations

import tracemalloc
from collections.abc import Callable


def peak_extra_bytes(func: Callable[[], object]) -> int:
    """Run ``func`` and return its peak memory above the starting level.

    Works whether or not tracing is already active (e.g. ``PYTHONTRACEMALLOC``
    set by a developer): the peak is reset first and measured against the
    current traced size, so earlier allocations are never counted.
    """
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        baseline, _ = tracemalloc.get_traced_memory()
        func()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not was_tracing:
            tracemalloc.stop()
    return peak - baseline
