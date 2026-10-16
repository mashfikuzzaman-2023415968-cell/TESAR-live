"""Top-level orchestration of the coordinator (spec Phase 4, Section 4).

::

    Coordinator.start()
        state dir (0700) -> TLS certificate -> device registry -> security log
        output dir: events.jsonl, config_effective.yaml
        Server (TLS :tcp_port) + FailureDetector (pings) + Scheduler (policy)
        trusted local worker subprocess (ephemeral credential)
        UDP beacons
    Coordinator.run_job()
        build the job (thread) -> start the reference (thread) -> snapshot.json
        scheduler + dispatcher run it -> stage 3 in-process (thread)
        output.png / reference.png / report.json -> job_finished event
    Coordinator.stop()
        beacons, scheduler, pings, server (every connection closed), local
        worker, event log — in that order, so nothing writes after its
        dependency is gone.

All blocking work (image loading, reference, PNG encoding, benchmark) runs in
threads; the event loop only does networking and bookkeeping (spec Section 12).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

from mdc.common.config import Config, dump_yaml
from mdc.common.events import EventLogger, to_jsonable
from mdc.common.ids import new_id
from mdc.common.redact import redact
from mdc.common.statedir import ensure_private_dir, ensure_private_subdir
from mdc.common.timeutil import monotonic, wait_event
from mdc.common.tls import server_context
from mdc.coordinator.assembler import Assembler
from mdc.coordinator.beacon import (
    BeaconSender,
    beacon_destinations,
    beacon_payload,
    detect_lan_ip,
)
from mdc.coordinator.certs import CertInfo, ensure_certificate
from mdc.coordinator.dispatcher import Dispatcher
from mdc.coordinator.enrollment import DeviceRegistry
from mdc.coordinator.estimators import Estimators
from mdc.coordinator.failure_detector import FailureDetector
from mdc.coordinator.job import Job
from mdc.coordinator.job_builder import build_job
from mdc.coordinator.local_worker import LocalCredential, LocalWorker
from mdc.coordinator.metrics import RunTiming, build_report, local_estimate_s
from mdc.coordinator.netestimator import NetEstimator
from mdc.coordinator.policies import make_policy
from mdc.coordinator.registry import Registry
from mdc.coordinator.reputation import Reputation
from mdc.coordinator.scheduler import Scheduler
from mdc.coordinator.security_log import SecurityLog
from mdc.coordinator.server import REVOCATION_CHECK_S, Server, ServerContext
from mdc.tasks.benchmark import run_benchmark

log = logging.getLogger(__name__)

WAIT_POLL_S = 0.1


class JobFailed(RuntimeError):
    """The job could not complete (e.g. it timed out)."""


class Coordinator:
    """The coordinator process. ``await start()``, ``await run_job()``, ``await stop()``."""

    def __init__(
        self,
        config: Config,
        insecure_dev: bool = False,
        output_dir: str | Path | None = None,
        cache_dir: str | Path | None = "runs/.cache",
        revocation_check_s: float = REVOCATION_CHECK_S,
    ) -> None:
        self.cfg = config
        self.insecure_dev = insecure_dev
        self.output_dir = Path(output_dir or config.job.output_dir)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.revocation_check_s = revocation_check_s
        self.coord_id = new_id("coord")
        self.port = 0
        self.advertise_ip = ""
        self.job: Job | None = None
        self.last_job: Job | None = None  # shown by the dashboard after the job ends
        self.last_report: dict[str, Any] | None = None
        self.local_bench_ms_per_unit = 1.0
        self._job_done = asyncio.Event()
        self._started = False
        # Components (created in start()).
        self.cert: CertInfo
        self.security_log: SecurityLog
        self.devices: DeviceRegistry
        self.events: EventLogger
        self.registry: Registry
        self.estimators: Estimators
        self.reputation: Reputation
        self.netest: NetEstimator
        self.dispatcher: Dispatcher
        self.detector: FailureDetector
        self.scheduler: Scheduler
        self.server: Server
        self.assembler: Assembler
        self.local_credential: LocalCredential | None = None
        self.local_worker: LocalWorker | None = None
        self.beacon: BeaconSender | None = None

    # ------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        """Bring every component up; returns once the server is listening."""
        cfg = self.cfg
        state = ensure_private_dir(cfg.security.state_dir)
        self.security_log = SecurityLog.for_state_dir(state, cfg.security.security_log)
        if not cfg.security.tls:
            self.security_log.emit("insecure_mode_enabled", bind=cfg.network.coordinator_bind)
        loop = asyncio.get_running_loop()
        self.cert = await loop.run_in_executor(
            None, ensure_certificate, state, cfg.security.cert_days
        )
        self.devices = DeviceRegistry(state, self.security_log)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.events = EventLogger(self.output_dir / cfg.logging.events_file)
        (self.output_dir / "config_effective.yaml").write_text(dump_yaml(cfg))
        self.registry = Registry(self.events)
        self.local_bench_ms_per_unit = await loop.run_in_executor(None, run_benchmark)
        policy = make_policy(cfg)  # fails early for policies not implemented yet
        self.estimators = Estimators.from_config(cfg)
        self.reputation = self.estimators.reputation
        self.netest = self.estimators.netest
        self.dispatcher = Dispatcher(
            cfg,
            self.registry,
            self.events,
            self.security_log,
            channel_for=lambda wid: self.server.channel_for(wid),  # noqa: PLW0108 - late binding
            on_progress=self._on_change,
            estimators=self.estimators,
        )
        self.detector = FailureDetector(
            cfg,
            self.registry,
            lambda: self.server.live_connections(),  # noqa: PLW0108 - late binding
            dispatcher=self.dispatcher,
            close_connection=lambda wid: self.server.close_connection(wid),  # noqa: PLW0108
            on_change=self._on_change,
            finish_probe=lambda rec, sample, timed_out: self.server.finish_probe(  # noqa: PLW0108
                rec, sample, timed_out
            ),
        )
        self.scheduler = Scheduler(cfg, self.registry, self.dispatcher, policy, self.estimators)
        self.assembler = Assembler(self.output_dir, self.cache_dir)
        ssl_ctx = (
            server_context(self.cert.cert_path, self.cert.key_path) if cfg.security.tls else None
        )
        # The local credential needs the real port; create the server first with a
        # placeholder and fill the credential in before any local worker exists.
        self.server = Server(
            ServerContext(
                config=cfg,
                fingerprint=self.cert.fingerprint,
                ssl_context=ssl_ctx,
                devices=self.devices,
                registry=self.registry,
                dispatcher=self.dispatcher,
                detector=self.detector,
                events=self.events,
                security_log=self.security_log,
                on_change=self._on_change,
                netest=self.netest,
            ),
            revocation_check_s=self.revocation_check_s,
        )
        self.port = await self.server.start(cfg.network.coordinator_bind, cfg.network.tcp_port)
        if cfg.workers.local_worker:
            self.local_credential = LocalCredential(self.cert.fingerprint, self.port)
            self.server.local_credential = self.local_credential
            log_dir = ensure_private_subdir(state, "logs")
            self.local_worker = LocalWorker(
                self.local_credential,
                self.port,
                cfg.workers.local_slots,
                log_dir,
                self.events,
                insecure_dev=self.insecure_dev,
            )
            self.local_worker.start()
        self.advertise_ip = cfg.network.advertise_host or detect_lan_ip()
        destinations = beacon_destinations(
            cfg.network.beacon_modes,
            self.advertise_ip,
            cfg.network.discovery_port,
            cfg.network.multicast_group,
        )
        self.beacon = BeaconSender(
            beacon_payload(
                self.advertise_ip, self.port, self.coord_id, "tesar-live", self.cert.fingerprint
            ),
            destinations,
            cfg.network.beacon_interval_s,
        )
        self.beacon.start()
        self.detector.start()
        self.scheduler.start()
        self._started = True
        self.events.emit(
            "coordinator_started",
            ip=self.advertise_ip,
            tcp_port=self.port,
            policy=cfg.scheduler.policy,
            config_hash=cfg.config_hash(),
            fingerprint=self.cert.fingerprint,
        )
        log.info(
            "coordinator listening on %s:%d (fingerprint %s)",
            self.advertise_ip,
            self.port,
            self.cert.fingerprint,
        )

    async def stop(self) -> None:
        """Shut everything down in dependency order."""
        if not self._started:
            return
        self._started = False
        if self.beacon is not None:
            await self.beacon.stop()
        await self.scheduler.stop()
        await self.detector.stop()
        await self.dispatcher.drain()
        await self.server.stop()
        if self.local_worker is not None:
            await self.local_worker.stop()
        self.events.close()
        self.security_log.close()

    async def __aenter__(self) -> Coordinator:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    # ----------------------------------------------------------- workers

    def _on_change(self) -> None:
        self.scheduler.trigger()
        # Done when final can run and no verification is outstanding (spec Phase 10 item 6).
        if (
            self.job is not None
            and self.job.final_ready()
            and self.dispatcher.verifier.outstanding() == 0
        ):
            self._job_done.set()

    def local_active(self) -> bool:
        """True once the trusted local worker is ACTIVE."""
        return any(r.trusted for r in self.registry.active())

    async def wait_for_workers(self, n_remote: int, timeout_s: float) -> int:
        """Wait until ``n_remote`` remote workers (and the local one, if enabled) are ACTIVE.

        Returns the number of ACTIVE remote workers when the wait ended (it
        also ends at ``timeout_s``; the job then runs with whoever is there).
        """
        deadline = monotonic() + timeout_s
        while monotonic() < deadline:
            remote = len(self.registry.remote_active())
            local_ok = self.local_worker is None or self.local_active()
            if remote >= n_remote and local_ok:
                return remote
            await asyncio.sleep(WAIT_POLL_S)
        return len(self.registry.remote_active())

    # --------------------------------------------------------------- jobs

    def _write_json(self, name: str, data: Any) -> None:
        (self.output_dir / name).write_text(
            json.dumps(to_jsonable(redact(data)), indent=2, sort_keys=False) + "\n"
        )

    def snapshot(self, job: Job) -> dict[str, Any]:
        """Estimator state at job start (spec 11.2)."""
        return {
            "job_id": job.job_id,
            "policy": self.cfg.scheduler.policy,
            "local_bench_ms_per_unit": self.local_bench_ms_per_unit,
            "scheduler": dataclasses.asdict(self.cfg.scheduler),
            "energy": dataclasses.asdict(self.cfg.energy),
            "min_battery_pct": self.cfg.workers.min_battery_pct,
            "workers": [
                {
                    **r.snapshot(),
                    **self.netest.snapshot(r.worker_id),
                    "rep": self.reputation.rep(r.worker_id),
                    "lambda": self.reputation.failure_rate(r.worker_id),
                }
                for r in self.registry
            ],
            "compression": self.netest.compression_table(),
            "tasks": [
                {
                    "task_id": t.task_id,
                    "stage": t.stage,
                    "kind": t.kind,
                    "deps": t.deps,
                    "work_units": t.work_units,
                    "input_bytes_est": t.input_bytes_est,
                    "output_bytes_est": t.output_bytes_est,
                }
                for t in job.tasks.values()
            ],
        }

    def verification_summary(self, job: Job) -> dict[str, Any]:
        """Verification metrics for ``report.json`` (spec Phase 10 item 7)."""
        stats = self.dispatcher.verifier.stats
        checks = [a for a in job.attempts.values() if a.purpose in {"verify", "tiebreak"}]
        check_ms = sum(a.compute_ms or 0.0 for a in checks)
        total_ms = sum(a.compute_ms or 0.0 for a in job.attempts.values())
        names = {r.worker_id: r.name for r in self.registry}
        return {
            "verifications": len(checks),
            "mismatches": stats.mismatches,
            "matches": stats.matches,
            "tiebreaks": stats.tiebreaks,
            "free_replica_matches": stats.free_replica_matches,
            "skipped_no_verifier": stats.skipped_no_verifier,
            "repaired_tasks": stats.repaired_tasks,
            "overhead_pct": round(100.0 * check_ms / total_ms, 3) if total_ms else 0.0,
            "detection_latency_s": [
                {"worker": names.get(w, w), "latency_s": t}
                for w, t in sorted(stats.detection_latency_s().items())
            ],
            "quarantined": sorted(names.get(w, w) for w in stats.quarantined_t),
        }

    def _local_speeds(self) -> tuple[dict[str, float], float]:
        local = next((r for r in self.registry if r.trusted), None)
        if local is None:
            return {}, self.local_bench_ms_per_unit
        speeds = {k: e.value for k, e in local.speed.items() if e.value is not None}
        return speeds, local.bench_ms_per_unit

    async def run_job(
        self, job: Job | None = None, timeout_s: float | None = None
    ) -> dict[str, Any]:
        """Run one job to completion and return its report (also written to disk)."""
        if self.job is not None:
            raise JobFailed("a job is already running")
        loop = asyncio.get_running_loop()
        if job is None:
            job = await loop.run_in_executor(None, build_job, self.cfg)
        self.job = job
        job.policy = self.cfg.scheduler.policy
        job.policy_params = {"alpha": self.cfg.scheduler.alpha, "kappa": self.cfg.scheduler.kappa}
        self._job_done = asyncio.Event()
        self.events.reset_clock()
        self.assembler.start_reference(job)
        self._write_json("snapshot.json", self.snapshot(job))
        grid = job.grid
        self.events.emit(
            "job_started",
            job_id=job.job_id,
            job_type=job.job_type,
            policy=self.cfg.scheduler.policy,
            alpha=self.cfg.scheduler.alpha,
            n_tasks=len(job.tasks),
            rows=grid.rows if grid else None,
            cols=grid.cols if grid else None,
            halo=grid.halo if grid else None,
        )
        t0 = monotonic()
        job.start_t = t0
        self.estimators.energy.start_job()
        self.dispatcher.start_job(job)
        self.scheduler.start_job(job)
        finished = await wait_event(self._job_done, timeout_s if timeout_s is not None else 1e9)
        if not finished:
            self.scheduler.stop_job()
            self.job = None
            raise JobFailed(f"job did not finish within {timeout_s} s") from None
        makespan = monotonic() - t0
        job.end_t = monotonic()
        self.scheduler.stop_job()
        self.estimators.energy.end_job()
        for record in self.registry:  # nothing more is needed from in-flight copies
            for attempt in list(record.inflight.values()):
                self.dispatcher.cancel(attempt, "job finished")
        outcome = await self.assembler.finish(job)
        outcome["verification"] = self.verification_summary(job)
        speeds, fallback = self._local_speeds()
        report = build_report(
            job,
            self.cfg,
            self.registry,
            self.dispatcher.traffic,
            RunTiming(makespan, local_estimate_s(job, speeds, fallback)),
            outcome,
            self.cfg.experiment.seed,
            self.estimators,
        )
        self._write_json("report.json", report)
        self.last_job, self.last_report = job, report
        self.events.emit(
            "job_finished",
            job_id=job.job_id,
            makespan_s=round(makespan, 4),
            output_correct=outcome.get("output_correct"),
        )
        self.job = None
        return report


async def run_until_stopped(coordinator: Coordinator, stop: asyncio.Event) -> None:
    """Keep serving until ``stop`` is set (Ctrl+C in the CLI)."""
    with contextlib.suppress(asyncio.CancelledError):
        await stop.wait()
