"""Coordinator command line: ``python -m mdc.coordinator`` (spec Phase 4).

::

    python -m mdc.coordinator [--config FILE] [--set key=value ...] [--headless]
        [--run-job] [--exit-after-job] [--wait-workers N] [--wait-timeout-s 30]
        [--output-dir DIR] [--job-timeout-s S] [--insecure-dev]

* ``--run-job``: start the job once ``--wait-workers`` remote workers (the
  local worker is not counted) are ACTIVE, or when ``--wait-timeout-s`` expires.
* ``--exit-after-job``: shut down after the job (workers see the connection
  close and keep retrying — fine for experiments).
* ``--headless``: no dashboard. Otherwise the dashboard serves on
  ``dashboard.host:port`` (loopback by default; HTTPS with the coordinator
  certificate if the host is not loopback) and prints its login URL; the admin
  token is in ``state_dir/admin_token``.
* ``--insecure-dev``: allow ``security.tls: false``; only with a loopback bind.
* ``--diag``: print interfaces, port availability, a test beacon, the
  certificate fingerprint and the device count, then exit (never secrets).

Exit status: 0 = job finished with correct output (or served until Ctrl+C),
1 = job finished with incorrect output or failed, 2 = configuration error.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import sys
from collections.abc import Sequence
from pathlib import Path

from mdc.common.config import Config, ConfigError, load_config
from mdc.common.logging_setup import setup_logging
from mdc.common.tls import server_context
from mdc.coordinator.coordinator import Coordinator, JobFailed, run_until_stopped
from mdc.coordinator.diag import run_diag
from mdc.coordinator.policies import UnknownPolicyError
from mdc.dashboard.auth import admin_token_path, load_or_create_admin_token
from mdc.dashboard.backend import CoordinatorBackend
from mdc.dashboard.web import Dashboard, require_safe_binding

EXIT_OK, EXIT_JOB_FAILED, EXIT_USAGE = 0, 1, 2
RED, BOLD, RESET = "\033[1;31m", "\033[1m", "\033[0m"


def build_parser() -> argparse.ArgumentParser:
    """The coordinator's argument parser."""
    p = argparse.ArgumentParser(
        prog="python -m mdc.coordinator", description="TESAR-Live coordinator"
    )
    p.add_argument("--config", default=None)
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--headless", action="store_true", help="no dashboard")
    p.add_argument("--run-job", action="store_true")
    p.add_argument("--exit-after-job", action="store_true")
    p.add_argument("--wait-workers", type=int, default=0)
    p.add_argument("--wait-timeout-s", type=float, default=30.0)
    p.add_argument("--job-timeout-s", type=float, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--insecure-dev", action="store_true")
    p.add_argument("--diag", action="store_true", help="print network diagnostics and exit")
    return p


def print_banner(coordinator: Coordinator, cfg: Config) -> None:
    """Tell the operator how workers connect."""
    print(f"{BOLD}TESAR-Live coordinator{RESET}")
    print(
        f"  Workers can connect with: --coordinator {coordinator.advertise_ip}:{coordinator.port}"
    )
    print(f"  Certificate fingerprint:  {coordinator.cert.fingerprint}")
    print(
        f"  Policy: {cfg.scheduler.policy} (alpha={cfg.scheduler.alpha}), "
        f"output: {coordinator.output_dir}"
    )
    if not cfg.security.tls:
        print(f"{RED}*** INSECURE DEV MODE: TLS DISABLED (loopback only) ***{RESET}")
    sys.stdout.flush()


async def amain(args: argparse.Namespace, cfg: Config) -> int:
    """Async entry point."""
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(sig, stop.set)
    coordinator = Coordinator(cfg, insecure_dev=args.insecure_dev, output_dir=args.output_dir)
    await coordinator.start()
    dashboard: Dashboard | None = None
    backend: CoordinatorBackend | None = None
    try:
        print_banner(coordinator, cfg)
        if not args.headless and cfg.dashboard.enabled:
            dashboard, backend = await start_dashboard(coordinator, cfg)
        if not args.run_job:
            await run_until_stopped(coordinator, stop)
            return EXIT_OK
        remote = await coordinator.wait_for_workers(args.wait_workers, args.wait_timeout_s)
        print(
            f"  Starting job with {remote} remote worker(s) (+ local: {coordinator.local_active()})"
        )
        job_task = asyncio.create_task(coordinator.run_job(timeout_s=args.job_timeout_s))
        stop_task = asyncio.create_task(stop.wait())
        done, _ = await asyncio.wait({job_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        stop_task.cancel()
        if job_task not in done:
            job_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, JobFailed):
                await job_task
            print("Interrupted.")
            return EXIT_JOB_FAILED
        try:
            report = job_task.result()
        except JobFailed as exc:
            print(f"{RED}Job failed: {exc}{RESET}")
            return EXIT_JOB_FAILED
        mark = "correct ✓" if report["output_correct"] else "INCORRECT ✗"
        print(
            f"  Job {report['job_id']} finished in {report['makespan_s']:.2f} s "
            f"(speedup {report['speedup']}): output {mark}"
        )
        print(f"  Report: {Path(coordinator.output_dir) / 'report.json'}")
        if not args.exit_after_job:
            await run_until_stopped(coordinator, stop)
        return EXIT_OK if report["output_correct"] else EXIT_JOB_FAILED
    finally:
        if backend is not None:
            await backend.stop()
        if dashboard is not None:
            await dashboard.stop()
        await coordinator.stop()


async def start_dashboard(
    coordinator: Coordinator, cfg: Config
) -> tuple[Dashboard, CoordinatorBackend]:
    """Serve the dashboard and print where to log in (spec S.9 item 2)."""
    https = require_safe_binding(cfg)
    token = load_or_create_admin_token(cfg.security.state_dir)
    backend = CoordinatorBackend(coordinator)
    dashboard = Dashboard(cfg, backend, token, tls=https)
    ssl_ctx = (
        server_context(coordinator.cert.cert_path, coordinator.cert.key_path) if https else None
    )
    port = await dashboard.start(ssl_ctx)
    host = cfg.dashboard.host
    shown = f"[{host}]" if ":" in host else host
    print(f"  Dashboard: {'https' if https else 'http'}://{shown}:{port}/login")
    print(f"  Admin token file: {admin_token_path(cfg.security.state_dir)}")
    sys.stdout.flush()
    return dashboard, backend


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the exit status."""
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config, args.overrides, insecure_dev=args.insecure_dev)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if args.diag:
        return run_diag(cfg)
    if not args.headless and cfg.dashboard.enabled:
        try:
            require_safe_binding(cfg)  # refuse before anything starts (S.9 item 1)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_USAGE
    output_dir = Path(args.output_dir or cfg.job.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(cfg.logging.level, output_dir / "coordinator.log")
    try:
        return asyncio.run(amain(args, cfg))
    except UnknownPolicyError as exc:
        print(
            f"error: {exc} (set scheduler.policy, e.g. --set scheduler.policy=round_robin)",
            file=sys.stderr,
        )
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
