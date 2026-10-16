"""Run an in-process :class:`Worker` against a stub coordinator (tests)."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from mdc.common.credentials import JoinCode
from mdc.worker.battery import BatteryProvider, MockBattery
from mdc.worker.faults import FaultInjector, FaultSettings
from mdc.worker.shaper import Shaper
from mdc.worker.worker import Worker, WorkerOptions


@dataclass
class ExitRecorder:
    """Stands in for ``os._exit`` so crash faults can be observed, not suffered."""

    codes: list[int] = field(default_factory=list)

    def __call__(self, code: int) -> None:
        self.codes.append(code)


@contextlib.asynccontextmanager
async def running_worker(
    join: JoinCode,
    port: int,
    battery: BatteryProvider | None = None,
    faults: FaultSettings | None = None,
    shaper: Shaper | None = None,
    seed: int = 1,
    **options: Any,
) -> AsyncIterator[tuple[Worker, asyncio.Task[int], ExitRecorder]]:
    """Start ``Worker.run()`` as a task; stop it and wait for it on exit."""
    opts = WorkerOptions(coordinator=("127.0.0.1", port), **options)
    recorder = ExitRecorder()
    worker = Worker(
        join,
        opts,
        battery or MockBattery(90.0, 0.0),
        FaultInjector(faults or FaultSettings(), seed),
        shaper,
        bench_ms_per_unit=1.0,
        exit_func=recorder,
    )
    task = asyncio.create_task(worker.run())
    try:
        yield worker, task, recorder
    finally:
        worker.stop()
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(task, 10)
