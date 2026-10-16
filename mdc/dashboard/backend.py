"""The coordinator as seen by the dashboard (implements :class:`web.Backend`).

Builds the snapshot (coordinator, workers with their estimates, tile grid and
per-tile status, recent events, live metrics), starts jobs requested from the
browser and forwards demo controls to workers.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mdc
from mdc.common import protocol as P
from mdc.common.timeutil import monotonic
from mdc.coordinator.job import Job, TaskNode, TaskStatus
from mdc.coordinator.policies import make_policy
from mdc.dashboard.web import JobBusyError, JobRequest

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mdc.coordinator.coordinator import Coordinator

log = logging.getLogger(__name__)

IMAGES_DIR = Path(mdc.__file__).resolve().parent.parent / "data" / "images"
PRODUCING = {"primary", "replica"}
HAS_RESULT = frozenset({TaskStatus.DONE, TaskStatus.VERIFYING})


def tile_status(task: TaskNode) -> str:
    """Dashboard tile state: pending|running|done|verifying|verified|invalidated."""
    if task.status is TaskStatus.RUNNING:
        return "running"
    if task.status is TaskStatus.VERIFYING:
        return "verifying"
    if task.status is TaskStatus.DONE:
        return "verified" if task.verified else "done"
    if task.status is TaskStatus.INVALIDATED:
        return "invalidated"
    return "pending"


class CoordinatorBackend:
    """Adapter between a running :class:`Coordinator` and the dashboard."""

    def __init__(self, coordinator: Coordinator, images_dir: Path = IMAGES_DIR) -> None:
        self.coord = coordinator
        self.events = coordinator.events
        self.images_dir = images_dir
        self.job_task: asyncio.Task[None] | None = None

    # ---------------------------------------------------------------- state

    def _job(self) -> Job | None:
        return self.coord.job or self.coord.last_job

    def _workers(self) -> list[dict[str, Any]]:
        coord = self.coord
        energy = coord.estimators.energy
        out = []
        for record in coord.registry:
            wid = record.worker_id
            ledger = energy.ledgers.get(wid)
            out.append(
                {
                    **record.snapshot(),
                    "rep": round(coord.reputation.rep(wid), 4),
                    "lambda": round(coord.reputation.failure_rate(wid), 6),
                    "rtt_ms": round(coord.netest.rtt(wid) * 1000, 2),
                    "bw_kbps": round(coord.netest.bw(wid) * 8 / 1000, 1),
                    "energy_budget_j": (
                        round(ledger.budget_j, 2)
                        if ledger is not None and ledger.budget_j != float("inf")
                        else None
                    ),
                }
            )
        return out

    @staticmethod
    def _tiles(job: Job) -> list[list[Any]]:
        """``[row, col, status, producer]`` for every stage-1 tile."""
        return [
            [t.row, t.col, tile_status(t), t.produced_by]
            for t in job.tasks.values()
            if t.stage == 1
        ]

    def _metrics(self, job: Job | None) -> dict[str, Any]:
        if job is None:
            return {}
        work = [a for a in job.attempts.values() if a.purpose in PRODUCING]
        statuses = [a.status for a in work]
        tiles = [t for t in job.tasks.values() if t.stage == 1]
        end = job.end_t if job.end_t is not None else monotonic()
        stats = self.coord.dispatcher.verifier.stats
        return {
            "elapsed_s": round(end - job.start_t, 3) if job.start_t is not None else 0.0,
            "tiles_done": sum(
                1 for t in tiles if t.status in {TaskStatus.DONE, TaskStatus.VERIFYING}
            ),
            "tiles_total": len(tiles),
            "attempts": len(work),
            "ok": statuses.count("ok") + statuses.count("late_accepted"),
            "timeout": sum(1 for a in work if a.timed_out),
            "lost": statuses.count("lost"),
            "error": statuses.count("error") + statuses.count("rejected"),
            "late": statuses.count("late") + statuses.count("late_accepted"),
            "replicas": sum(1 for a in work if a.purpose == "replica"),
            "verifications": sum(
                1 for a in job.attempts.values() if a.purpose in {"verify", "tiebreak"}
            ),
            "mismatches": stats.mismatches,
        }

    def state(self) -> dict[str, Any]:
        """The full snapshot (``GET /api/state`` and the websocket's first message)."""
        coord, cfg = self.coord, self.coord.cfg
        job = self._job()
        grid = job.grid if job is not None else None
        report = coord.last_report
        return {
            "coordinator": {
                "address": f"{coord.advertise_ip}:{coord.port}",
                "fingerprint": coord.cert.fingerprint,
                "policy": cfg.scheduler.policy,
                "alpha": cfg.scheduler.alpha,
                "verification": cfg.verification.enabled,
                "adaptive": cfg.chunking.adaptive,
                "job_status": "running"
                if coord.job is not None
                else ("finished" if job else "idle"),
                "job_id": job.job_id if job is not None else None,
                "lab_mode": cfg.security.lab_mode,
            },
            "workers": self._workers(),
            "grid": (
                {
                    "rows": grid.rows,
                    "cols": grid.cols,
                    "base_tile": grid.tile,
                    "height": grid.height,
                    "width": grid.width,
                }
                if grid is not None
                else None
            ),
            "tiles": self._tiles(job) if job is not None else [],
            "events": self.events.recent(200),
            "metrics": self._metrics(job),
            "result": (
                {
                    "output_correct": report.get("output_correct"),
                    "makespan_s": report.get("makespan_s"),
                    "speedup": report.get("speedup"),
                }
                if report is not None and coord.job is None
                else None
            ),
        }

    def state_update(self) -> dict[str, Any]:
        """The compact update pushed every 0.5 s."""
        full = self.state()
        return {
            k: full[k] for k in ("coordinator", "workers", "grid", "tiles", "metrics", "result")
        }

    # ---------------------------------------------------------------- jobs

    def start_job(self, request: JobRequest) -> None:
        """Apply the browser's settings and start a job in the background."""
        coord, cfg = self.coord, self.coord.cfg
        if coord.job is not None or (self.job_task is not None and not self.job_task.done()):
            raise JobBusyError
        cfg.scheduler.policy = request.policy
        cfg.scheduler.alpha = request.alpha
        cfg.job.image_path = str(request.image)
        cfg.job.blur_iterations = request.blur_iterations
        cfg.verification.enabled = request.verification
        cfg.chunking.adaptive = request.adaptive
        coord.scheduler.policy = make_policy(cfg)
        self.job_task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        try:
            await self.coord.run_job()
        except Exception:  # a failed job is shown in the event log, not raised into aiohttp
            log.exception("dashboard job failed")

    async def stop(self) -> None:
        """Cancel a running dashboard job (coordinator shutdown)."""
        if self.job_task is not None and not self.job_task.done():
            self.job_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.job_task

    # ------------------------------------------------------------- controls

    async def control(self, worker_id: str, action: str, value: float | None) -> bool:
        """Send a CONTROL message; False if the worker is not connected."""
        channel = self.coord.server.channel_for(worker_id)
        if channel is None:
            return False
        fields: dict[str, Any] = {"action": action}
        if value is not None:
            fields["value"] = value
        await channel.send(P.make_message(P.CONTROL, **fields))
        self.events.emit("demo_control", worker_id=worker_id, action=action, value=value)
        return True

    # -------------------------------------------------------------- results

    def output_png(self) -> Path | None:
        path = self.coord.output_dir / "output.png"
        return path if self.coord.job is None and path.is_file() else None

    def report(self) -> dict[str, Any] | None:
        return self.coord.last_report
