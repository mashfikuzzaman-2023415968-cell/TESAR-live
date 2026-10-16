"""Worker command line: ``python -m mdc.worker`` (spec Phase 3, S.7).

Credentials (never as a command-line argument — ``ps`` shows argv to every
user on the device):

* ``--join-file PATH`` — a join file with mode 0600 (looser files are refused);
* ``--join-code -``    — read the join code from standard input;
* ``MDC_JOIN_CODE``    — environment variable (used for the local worker).

Safety rules enforced before anything starts:

* fault-injection flags and ``--allow-demo-control`` require ``--lab-mode``,
  which prints a red warning banner;
* ``--local`` (the coordinator's trusted local worker) refuses all faults;
* the worker refuses to run as root unless ``--lab-mode --allow-root``;
* ``--insecure-dev`` (no TLS) only connects to a loopback coordinator.

``--diag`` prints platform, battery, benchmark, beacon and connectivity
information (never secrets) and exits.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import platform
import signal
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import mdc
from mdc.common.credentials import CredentialError, JoinCode, decode_join_code, read_join_file
from mdc.common.credentials import join_code_from_env as _join_from_env
from mdc.common.logging_setup import setup_logging
from mdc.common.safe_names import is_valid_name
from mdc.common.statedir import ensure_private_dir
from mdc.common.tls import PinMismatchError, open_pinned_connection
from mdc.worker.battery import make_battery
from mdc.worker.discovery_listener import NO_BEACON_HELP, listen_for_beacon
from mdc.worker.faults import FaultInjector, FaultSettings
from mdc.worker.shaper import Shaper
from mdc.worker.worker import Worker, WorkerOptions, detect_platform

log = logging.getLogger("mdc.worker")

EXIT_USAGE = 2
RED, RESET = "\033[1;31m", "\033[0m"
FAULT_FLAGS = (
    "drop_prob",
    "crash_prob",
    "crash_after_s",
    "malicious_prob",
    "slowdown",
    "flaky_mean_s",
    "collude_key",
    "net_delay_ms",
    "net_bw_kbps",
)
MAX_PORT = 65535
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class UsageError(Exception):
    """Invalid or unsafe combination of command-line options."""


def parse_hostport(text: str) -> tuple[str, int]:
    """``HOST:PORT`` (or ``[v6]:PORT``) -> ``(host, port)``."""
    host, sep, port_text = text.rpartition(":")
    if not sep or not host or not port_text.isdigit():
        raise UsageError(f"--coordinator must look like HOST:PORT, got {text!r}")
    port = int(port_text)
    if not 1 <= port <= MAX_PORT:
        raise UsageError("coordinator port must be in 1..65535")
    return host.strip("[]"), port


def build_parser() -> argparse.ArgumentParser:
    """The worker's argument parser."""
    p = argparse.ArgumentParser(prog="python -m mdc.worker", description="TESAR-Live worker")
    cred = p.add_argument_group("credentials (one of; or env MDC_JOIN_CODE)")
    cred.add_argument("--join-file", help="join file (mode 0600)")
    cred.add_argument("--join-code", help="'-' to read the join code from stdin")
    p.add_argument("--coordinator", help="HOST:PORT (skips beacon discovery)")
    p.add_argument("--name", help="display name (default: the enrolled name)")
    p.add_argument("--slots", type=int, default=1, help="concurrent tasks (1..8)")
    p.add_argument("--max-cpu-slots", type=int, default=None, help="cap on --slots")
    p.add_argument("--prefetch", type=int, default=1)
    p.add_argument("--device-class", choices=["phone", "laptop"], default=None)
    p.add_argument("--battery", choices=["auto", "termux", "sysfs", "mock"], default="auto")
    p.add_argument("--mock-battery-pct", type=float, default=80.0)
    p.add_argument("--mock-drain-pct-per-min", type=float, default=0.5)
    p.add_argument(
        "--min-battery", type=float, default=15.0, help="refuse tasks below this %% unless charging"
    )
    p.add_argument("--task-max-s", type=float, default=120.0, help="per-task CPU time limit")
    p.add_argument("--idle-timeout-s", type=float, default=30.0)
    p.add_argument("--battery-interval-s", type=float, default=2.0)
    p.add_argument("--discovery-port", type=int, default=50000)
    p.add_argument("--multicast-group", default="239.255.77.77")
    p.add_argument("--local", action="store_true", help="trusted local worker (coordinator use)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--lab-mode", action="store_true", help="allow fault injection (testing only)")
    p.add_argument("--allow-demo-control", action="store_true")
    p.add_argument("--allow-root", action="store_true")
    p.add_argument("--insecure-dev", action="store_true", help="no TLS; loopback only")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--log-dir", default="~/.mdc-worker/logs")
    p.add_argument("--diag", action="store_true", help="print diagnostics and exit")
    p.add_argument("--diag-listen-s", type=float, default=5.0)
    faults = p.add_argument_group("fault injection (requires --lab-mode)")
    faults.add_argument("--drop-prob", type=float, default=None)
    faults.add_argument("--crash-prob", type=float, default=None)
    faults.add_argument("--crash-after-s", type=float, default=None)
    faults.add_argument("--malicious-prob", type=float, default=None)
    faults.add_argument("--slowdown", type=float, default=None)
    faults.add_argument("--flaky-mean-s", type=float, default=None)
    faults.add_argument(
        "--collude-key", default=None, help="identical corruption across workers sharing the key"
    )
    faults.add_argument("--net-delay-ms", type=float, default=None)
    faults.add_argument("--net-bw-kbps", type=float, default=None)
    return p


def check_safety(args: argparse.Namespace) -> None:
    """Refuse unsafe option combinations (fail closed)."""
    used_faults = [f for f in FAULT_FLAGS if getattr(args, f) is not None]
    if used_faults and not args.lab_mode:
        flags = ", ".join("--" + f.replace("_", "-") for f in used_faults)
        raise UsageError(f"{flags} refused: fault injection requires --lab-mode")
    if args.allow_demo_control and not args.lab_mode:
        raise UsageError("--allow-demo-control requires --lab-mode")
    if args.allow_root and not args.lab_mode:
        raise UsageError("--allow-root requires --lab-mode")
    if args.local:
        if used_faults or args.allow_demo_control:
            raise UsageError("the trusted --local worker never accepts faults or CONTROL")
        if args.coordinator is None:
            raise UsageError("--local requires --coordinator 127.0.0.1:PORT")
        if parse_hostport(args.coordinator)[0] not in LOOPBACK_HOSTS:
            raise UsageError("the trusted --local worker only connects to 127.0.0.1")
    if hasattr(os, "geteuid") and os.geteuid() == 0 and not (args.lab_mode and args.allow_root):
        raise UsageError(
            "refusing to run as root: a worker should never have root privileges "
            "(use --lab-mode --allow-root only for throw-away test machines)"
        )
    if args.coordinator is not None:
        parse_hostport(args.coordinator)
    if args.name is not None and not is_valid_name(args.name):
        raise UsageError("--name must match ^[A-Za-z0-9_.-]{1,32}$")
    if not 1 <= args.slots <= 8:  # noqa: PLR2004
        raise UsageError("--slots must be in 1..8")
    if args.join_code is not None and args.join_code != "-":
        raise UsageError(
            "--join-code only accepts '-' (read from stdin): a secret on the command line "
            "is visible to every user in the process list. Use --join-file or MDC_JOIN_CODE."
        )
    if args.insecure_dev:
        if args.coordinator is None:
            raise UsageError("--insecure-dev requires --coordinator 127.0.0.1:PORT")
        host, _ = parse_hostport(args.coordinator)
        if host not in LOOPBACK_HOSTS:
            raise UsageError("--insecure-dev only connects to a loopback coordinator")


def load_join(args: argparse.Namespace) -> JoinCode:
    """Credentials from --join-file, --join-code - (stdin) or MDC_JOIN_CODE."""
    if args.join_file and args.join_code:
        raise UsageError("give either --join-file or --join-code -, not both")
    if args.join_file:
        return read_join_file(Path(args.join_file).expanduser())
    if args.join_code == "-":
        return decode_join_code(sys.stdin.readline(4096))
    code = _join_from_env()
    if code is None:
        raise UsageError("no credentials: use --join-file PATH, --join-code - or MDC_JOIN_CODE")
    return code


def build_options(args: argparse.Namespace) -> WorkerOptions:
    """Translate parsed arguments into :class:`WorkerOptions`."""
    slots = args.slots
    if args.max_cpu_slots is not None:
        slots = max(1, min(slots, args.max_cpu_slots))
    device_class = args.device_class or ("phone" if detect_platform() == "android" else "laptop")
    return WorkerOptions(
        coordinator=parse_hostport(args.coordinator) if args.coordinator else None,
        name=args.name,
        slots=slots,
        prefetch=max(0, args.prefetch),
        device_class=device_class,
        local=args.local,
        lab_mode=args.lab_mode,
        allow_demo_control=args.allow_demo_control,
        min_battery_pct=args.min_battery,
        task_max_s=args.task_max_s,
        idle_timeout_s=args.idle_timeout_s,
        battery_interval_s=args.battery_interval_s,
        discovery_port=args.discovery_port,
        multicast_group=args.multicast_group or None,
        insecure_dev=args.insecure_dev,
    )


def build_faults(args: argparse.Namespace) -> tuple[FaultInjector, Shaper]:
    """Fault injector and shaper from the (already safety-checked) arguments."""
    settings = FaultSettings(
        drop_prob=args.drop_prob or 0.0,
        crash_prob=args.crash_prob or 0.0,
        crash_after_s=args.crash_after_s,
        malicious_prob=args.malicious_prob or 0.0,
        slowdown=args.slowdown if args.slowdown is not None else 1.0,
        flaky_mean_s=args.flaky_mean_s,
        collude_key=args.collude_key,
    )
    try:
        injector = FaultInjector(settings, args.seed)
        shaper = Shaper(args.net_delay_ms, args.net_bw_kbps)
    except ValueError as exc:
        raise UsageError(str(exc)) from None
    return injector, shaper


async def run_diag(args: argparse.Namespace, join: JoinCode | None) -> int:
    """Print diagnostics (no secrets) and exit."""
    from mdc.tasks.benchmark import run_benchmark  # noqa: PLC0415

    print("TESAR-Live worker diagnostics")
    print(
        f"  version:   mdc {mdc.__version__}, Python {platform.python_version()}, "
        f"numpy {np.__version__}"
    )
    print(f"  platform:  {detect_platform()} ({platform.platform()}), cpus={os.cpu_count()}")
    if join is not None:
        print(f"  identity:  name={join.name} worker_id={join.worker_id}")
        print(f"  pinned fp: {join.fp}")
        print(f"  join addr: {join.host}:{join.port}")
    else:
        print("  identity:  (no credentials given)")
    battery = make_battery(args.battery, args.mock_battery_pct, args.mock_drain_pct_per_min)
    print(f"  battery:   {type(battery).__name__} -> {battery.read()}")
    print(f"  benchmark: {run_benchmark():.3f} ms per work unit")
    if join is not None:
        print(f"  beacon:    listening {args.diag_listen_s:.0f} s on UDP {args.discovery_port} ...")
        try:
            beacon = await listen_for_beacon(
                join.fp, args.discovery_port, args.multicast_group or None, args.diag_listen_s
            )
        except OSError as exc:
            print(f"             cannot listen: {exc}")
        else:
            print(
                f"             heard {beacon.host}:{beacon.port}"
                if beacon
                else f"             none. {NO_BEACON_HELP}"
            )
    if args.coordinator and join is not None:
        host, port = parse_hostport(args.coordinator)
        print(f"  connect:   TLS to {host}:{port} ...")
        try:
            _, writer = await open_pinned_connection(host, port, join.fp, 5.0)
        except PinMismatchError:
            print("             TLS ok but the certificate does NOT match the pinned fingerprint")
        except (OSError, asyncio.TimeoutError) as exc:
            print(
                f"             failed: {type(exc).__name__}: {exc} "
                "(firewall? wrong address? client isolation on the WiFi?)"
            )
        else:
            print("             ok: TLS handshake done and certificate pin matches")
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit status."""
    args = build_parser().parse_args(argv)
    try:
        check_safety(args)
        join: JoinCode | None
        if args.diag:
            try:
                join = load_join(args)
            except UsageError:
                join = None
            return asyncio.run(run_diag(args, join))
        join = load_join(args)
        options = build_options(args)
        faults, shaper = build_faults(args)
    except (UsageError, CredentialError, PermissionError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    log_dir = ensure_private_dir(args.log_dir)
    setup_logging(args.log_level, log_dir / ("local.log" if args.local else "worker.log"))
    if args.lab_mode:
        print(
            f"{RED}*** LAB MODE: fault injection / demo control enabled. "
            f"Never use on devices you rely on. ***{RESET}",
            file=sys.stderr,
        )
        log.warning("lab mode enabled: faults=%s", faults.settings)
    if args.insecure_dev:
        print(f"{RED}*** INSECURE DEV MODE: no TLS (loopback only) ***{RESET}", file=sys.stderr)
    battery = make_battery(args.battery, args.mock_battery_pct, args.mock_drain_pct_per_min)
    worker = Worker(join, options, battery, faults, shaper)
    return asyncio.run(_run_with_signals(worker))


async def _run_with_signals(worker: Worker) -> int:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, RuntimeError):  # Windows
            loop.add_signal_handler(sig, worker.stop)
    # Logged only now, so "starting" means "ready: SIGINT/SIGTERM stop me cleanly".
    log.info(
        "worker %s (%s) starting: slots=%d, battery=%s, bench=%.3f ms/unit",
        worker.name,
        worker.join.worker_id,
        worker.slots,
        worker.battery.source,
        worker.bench_ms_per_unit,
    )
    return await worker.run()


if __name__ == "__main__":
    sys.exit(main())
