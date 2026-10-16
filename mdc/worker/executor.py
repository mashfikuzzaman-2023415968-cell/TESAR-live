"""Thread-pool task execution (spec Phase 3 item 3, rule 0.1.6).

Why threads (not processes)? Termux on Android lacks a working ``sem_open``,
so ``multiprocessing`` and ``ProcessPoolExecutor`` crash there. Threads are
enough because numpy releases the GIL during heavy array operations, and the
asyncio reader loop only needs short time slices to answer PINGs.

The event loop never runs a kernel itself: ``run_in_executor`` hands the work
to a pool thread and the coroutine awaits the result, so a long computation
can never delay a PONG (the failure detector would otherwise think the
worker is dead).
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy.typing as npt

from mdc.common.validation import TaskLimits
from mdc.tasks.base import TaskOutput, execute


class TaskExecutor:
    """Runs registered tasks on ``slots`` worker threads."""

    def __init__(self, slots: int) -> None:
        if slots < 1:
            raise ValueError("slots must be >= 1")
        self.slots = slots
        self._pool = ThreadPoolExecutor(max_workers=slots, thread_name_prefix="mdc-task")

    async def run(
        self,
        kind: str,
        params: dict[str, Any],
        data: npt.NDArray[Any] | None,
        limits: TaskLimits,
        max_seconds: float,
    ) -> tuple[TaskOutput, float]:
        """Execute one task in the pool; return ``(output, compute_seconds)``.

        ``max_seconds`` becomes the kernel's cooperative deadline (spec S.6.5).
        """
        loop = asyncio.get_running_loop()
        deadline = time.monotonic() + max_seconds

        def work() -> tuple[TaskOutput, float]:
            t0 = time.perf_counter()
            output = execute(kind, params, data, limits, deadline)
            return output, time.perf_counter() - t0

        return await loop.run_in_executor(self._pool, work)

    def shutdown(self) -> None:
        """Stop accepting work; queued (not started) work is cancelled.

        Running threads cannot be killed in Python; they finish in the
        background and their results are simply discarded.
        """
        self._pool.shutdown(wait=False, cancel_futures=True)
