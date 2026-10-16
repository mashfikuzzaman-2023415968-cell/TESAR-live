"""Launch a local heterogeneous cluster of workers from a profile (spec Phase 5 item 7).

Profile (``configs/cluster_profiles/*.yaml``)::

    coordinator: "127.0.0.1:50010"
    lab_mode: true          # required if any worker uses fault flags
    workers:
      - {name: "fast-phone", slots: 1, args: ["--device-class", "phone"]}
      - {name: "slow-phone", slots: 1, args: ["--slowdown", "3", "--net-bw-kbps", "2000"]}

For every worker this script:

1. enrolls it in the coordinator's state directory through the enrollment API
   (an already enrolled name gets a fresh secret instead — same ``worker_id``,
   so its history is kept);
2. writes its join file to ``<runs>/cluster_secrets/<name>.join`` (mode 0600 —
   it is a secret) and passes it with ``--join-file`` (never on argv);
3. starts ``python -m mdc.worker`` with its own log directory under
   ``<runs>/cluster_logs/<name>/`` and seed ``seed + index``.

PIDs are written to ``<runs>/cluster_pids.json`` (used by ``scripts/chaos.py``).
Ctrl+C (or SIGTERM) is forwarded to every worker; the script waits for them.

Usage::

    python scripts/local_cluster.py configs/cluster_profiles/hetero5.yaml \\
        [--state-dir ~/.mdc] [--coordinator HOST:PORT] [--seed 1] [--runs-dir runs] \\
        [--duration S]
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mdc.common.credentials import write_join_file  # noqa: E402
from mdc.common.safe_names import is_valid_name  # noqa: E402
from mdc.common.statedir import ensure_private_dir  # noqa: E402
from mdc.coordinator.certs import ensure_certificate  # noqa: E402
from mdc.coordinator.enrollment import DeviceRegistry  # noqa: E402

MAX_SLOTS = 8
STOP_TIMEOUT_S = 10.0


class ProfileError(ValueError):
    """The cluster profile is malformed."""


@dataclass
class WorkerSpec:
    """One worker of a profile."""

    name: str
    slots: int = 1
    args: list[str] = field(default_factory=list)


@dataclass
class Profile:
    """A parsed cluster profile."""

    coordinator: str
    lab_mode: bool
    workers: list[WorkerSpec]


def load_profile(path: str | Path) -> Profile:
    """Parse and validate a profile YAML file."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) - {"coordinator", "lab_mode", "workers"}:
        raise ProfileError("profile must have only: coordinator, lab_mode, workers")
    coordinator = data.get("coordinator", "127.0.0.1:50010")
    lab_mode = data.get("lab_mode", False)
    workers = data.get("workers")
    if not isinstance(coordinator, str) or ":" not in coordinator:
        raise ProfileError("coordinator must be 'HOST:PORT'")
    if not isinstance(lab_mode, bool):
        raise ProfileError("lab_mode must be true or false")
    if not isinstance(workers, list) or not workers:
        raise ProfileError("workers must be a non-empty list")
    specs: list[WorkerSpec] = []
    for entry in workers:
        if not isinstance(entry, dict) or set(entry) - {"name", "slots", "args"}:
            raise ProfileError(f"bad worker entry {entry!r}")
        name, slots, args = entry.get("name"), entry.get("slots", 1), entry.get("args", [])
        if not is_valid_name(name):
            raise ProfileError(f"invalid worker name {name!r}")
        if type(slots) is not int or not 1 <= slots <= MAX_SLOTS:
            raise ProfileError(f"{name}: slots must be 1..8")
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise ProfileError(f"{name}: args must be a list of strings (quote numbers)")
        specs.append(WorkerSpec(str(name), slots, list(args)))
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ProfileError("worker names must be unique")
    return Profile(coordinator, lab_mode, specs)


def worker_argv(
    spec: WorkerSpec,
    join_file: Path,
    coordinator: str,
    lab_mode: bool,
    seed: int,
    log_dir: Path,
    extra: Sequence[str] = (),
) -> list[str]:
    """Command line of one worker (no secrets: the join code is in the 0600 file)."""
    argv = [
        sys.executable,
        "-m",
        "mdc.worker",
        "--join-file",
        str(join_file),
        "--coordinator",
        coordinator,
        "--slots",
        str(spec.slots),
        "--seed",
        str(seed),
        "--log-dir",
        str(log_dir),
    ]
    if lab_mode:
        argv.append("--lab-mode")
    if "--battery" not in spec.args:
        argv += ["--battery", "mock"]  # simulated cluster: never read the laptop's battery
    return argv + spec.args + list(extra)


def prepare(profile: Profile, state_dir: Path, runs_dir: Path, coordinator: str) -> dict[str, Path]:
    """Enroll (or re-key) every worker and write its join file; return name -> join file."""
    ensure_private_dir(state_dir)
    cert = ensure_certificate(state_dir, 825)
    registry = DeviceRegistry(state_dir)
    host, _, port_text = coordinator.rpartition(":")
    port = int(port_text)
    secrets_dir = ensure_private_dir(runs_dir / "cluster_secrets")
    joins: dict[str, Path] = {}
    for spec in profile.workers:
        if registry.find_active(spec.name) is not None:
            code = registry.rotate_secret(spec.name, host, port, cert.fingerprint)
        else:
            code = registry.enroll(spec.name, host, port, cert.fingerprint)
        joins[spec.name] = write_join_file(secrets_dir / f"{spec.name}.join", code)
    return joins


def launch(
    profile: Profile,
    joins: dict[str, Path],
    runs_dir: Path,
    coordinator: str,
    seed: int,
    extra: Sequence[str] = (),
) -> dict[str, subprocess.Popen[bytes]]:
    """Start every worker process; write the PID file."""
    logs = runs_dir / "cluster_logs"
    procs: dict[str, subprocess.Popen[bytes]] = {}
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env.pop("MDC_JOIN_CODE", None)
    for index, spec in enumerate(profile.workers):
        log_dir = logs / spec.name
        log_dir.mkdir(parents=True, exist_ok=True)
        argv = worker_argv(
            spec, joins[spec.name], coordinator, profile.lab_mode, seed + index, log_dir, extra
        )
        with (log_dir / "stdout.log").open("wb") as out:
            procs[spec.name] = subprocess.Popen(  # argument list, no shell
                argv,
                stdout=out,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=ROOT,
            )
    pid_file = runs_dir / "cluster_pids.json"
    pid_file.write_text(json.dumps({n: p.pid for n, p in procs.items()}, indent=1))
    return procs


def stop_all(procs: dict[str, subprocess.Popen[bytes]]) -> dict[str, int | None]:
    """SIGTERM every worker, wait, SIGKILL stragglers; return exit codes."""
    for proc in procs.values():
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
    deadline = time.monotonic() + STOP_TIMEOUT_S
    for proc in procs.values():
        try:
            proc.wait(max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    return {name: proc.returncode for name, proc in procs.items()}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Launch a local worker cluster")
    parser.add_argument("profile")
    parser.add_argument("--state-dir", default="~/.mdc", help="coordinator state directory")
    parser.add_argument("--coordinator", default=None, help="override the profile's HOST:PORT")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="stop after this many seconds (default: until Ctrl+C)",
    )
    parser.add_argument(
        "--extra-args",
        default="",
        help='worker options added to every worker, e.g. "--drop-prob 0.2" (experiments)',
    )
    args = parser.parse_args(argv)
    extra = shlex.split(args.extra_args)
    if extra and not load_profile(args.profile).lab_mode:
        print("error: --extra-args needs a profile with lab_mode: true", file=sys.stderr)
        return 2
    try:
        profile = load_profile(args.profile)
    except (OSError, ProfileError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    coordinator = args.coordinator or profile.coordinator
    runs_dir = Path(args.runs_dir).expanduser()
    runs_dir.mkdir(parents=True, exist_ok=True)
    joins = prepare(profile, Path(args.state_dir).expanduser(), runs_dir, coordinator)
    procs = launch(profile, joins, runs_dir, coordinator, args.seed, extra)
    if coordinator.rpartition(":")[0] in {"127.0.0.1", "localhost", "::1"}:
        print(
            f"note: all {len(procs)} workers connect from one address; start the coordinator "
            f"with --set network.max_connections_per_ip={len(procs) + 2} (default 4)",
            flush=True,
        )
    print(
        f"started {len(procs)} workers -> {coordinator}: "
        + ", ".join(f"{n} (pid {p.pid})" for n, p in procs.items()),
        flush=True,
    )

    stopping = {"flag": False}

    def request_stop(signum: int, frame: Any) -> None:
        stopping["flag"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    end = None if args.duration is None else time.monotonic() + args.duration
    while not stopping["flag"] and (end is None or time.monotonic() < end):
        if all(p.poll() is not None for p in procs.values()):
            break
        time.sleep(0.2)
    codes = stop_all(procs)
    print("stopped: " + ", ".join(f"{n}={c}" for n, c in codes.items()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
