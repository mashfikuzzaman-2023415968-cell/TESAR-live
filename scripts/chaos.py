"""Disturb a running local cluster on a schedule (spec Phase 5 item 6).

Actions (``T`` in seconds)::

    --kill NAME@T        SIGKILL the worker (a phone whose app is killed)
    --stop NAME@T        SIGSTOP (a frozen phone: process alive, nothing answers)
    --cont NAME@T        SIGCONT (the frozen phone wakes up again)
    --random-kills K --window A-B
                         kill K distinct workers at random times in [A, B] (seeded)

Worker PIDs come from ``runs/cluster_pids.json`` (written by
``scripts/local_cluster.py``). Times are measured from when chaos starts, or —
with ``--events runs/latest/events.jsonl`` — from the coordinator's
``job_started`` event, so the disturbance lands inside the job.

Usage::

    python scripts/chaos.py --pids runs/cluster_pids.json --stop slow-phone@3 \\
        --cont slow-phone@12 --random-kills 1 --window 5-15 --seed 7
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ACTIONS = {"kill": signal.SIGKILL, "stop": signal.SIGSTOP, "cont": signal.SIGCONT}
POLL_S = 0.1


class ChaosError(ValueError):
    """Bad action specification."""


@dataclass(frozen=True, order=True)
class Action:
    """Send ``signal_name`` to worker ``name`` at time ``at`` (seconds)."""

    at: float
    name: str
    signal_name: str


def parse_action(kind: str, spec: str) -> Action:
    """``"slow-phone@3.5"`` -> Action(3.5, "slow-phone", kind)."""
    name, sep, at_text = spec.rpartition("@")
    if not sep or not name:
        raise ChaosError(f"--{kind} expects NAME@SECONDS, got {spec!r}")
    try:
        at = float(at_text)
    except ValueError:
        raise ChaosError(f"--{kind}: {at_text!r} is not a number") from None
    if at < 0:
        raise ChaosError(f"--{kind}: time must be >= 0")
    return Action(at, name, kind)


def parse_window(text: str) -> tuple[float, float]:
    """``"5-15"`` -> (5.0, 15.0)."""
    start, sep, end = text.partition("-")
    try:
        a, b = float(start), float(end)
    except ValueError:
        raise ChaosError(f"--window expects A-B, got {text!r}") from None
    if not sep or a < 0 or b < a:
        raise ChaosError(f"--window expects 0 <= A <= B, got {text!r}")
    return a, b


def random_kills(
    names: list[str], count: int, window: tuple[float, float], rng: random.Random
) -> list[Action]:
    """``count`` distinct workers killed at uniform random times in ``window``."""
    if not 0 <= count <= len(names):
        raise ChaosError(f"--random-kills must be in 0..{len(names)}")
    victims = rng.sample(sorted(names), count)
    return [Action(rng.uniform(*window), name, "kill") for name in victims]


def build_plan(args: argparse.Namespace, pids: dict[str, int]) -> list[Action]:
    """All actions, sorted by time; unknown worker names are an error."""
    plan = [parse_action(kind, spec) for kind in ACTIONS for spec in getattr(args, kind)]
    if args.random_kills:
        rng = random.Random(args.seed)  # noqa: S311 - experiment randomness, not security
        plan += random_kills(list(pids), args.random_kills, parse_window(args.window), rng)
    unknown = sorted({a.name for a in plan} - set(pids))
    if unknown:
        raise ChaosError(f"unknown worker(s) {unknown}; known: {sorted(pids)}")
    return sorted(plan)


def wait_for_job_start(events_file: Path, timeout_s: float) -> bool:
    """Block until ``events_file`` contains a ``job_started`` event."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if events_file.exists():
            for line in events_file.read_text(encoding="utf-8").splitlines():
                if '"type":"job_started"' in line.replace(" ", ""):
                    return True
        time.sleep(POLL_S)
    return False


def execute(
    plan: list[Action],
    pids: dict[str, int],
    send: Callable[[int, int], None] = os.kill,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> list[str]:
    """Run the plan; return a log line per action (missing processes are reported)."""
    start = clock()
    log = []
    for action in plan:
        delay = action.at - (clock() - start)
        if delay > 0:
            sleep(delay)
        try:
            send(pids[action.name], ACTIONS[action.signal_name])
            result = "ok"
        except ProcessLookupError:
            result = "process already gone"
        line = f"t={action.at:6.2f}s {action.signal_name:<4} {action.name} -> {result}"
        log.append(line)
        print(line, flush=True)
    return log


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Disturb a local worker cluster")
    parser.add_argument("--pids", default="runs/cluster_pids.json")
    parser.add_argument("--events", default=None, help="start the clock at job_started")
    parser.add_argument("--wait-s", type=float, default=300.0)
    for kind in ACTIONS:
        parser.add_argument(f"--{kind}", action="append", default=[], metavar="NAME@T")
    parser.add_argument("--random-kills", type=int, default=0)
    parser.add_argument("--window", default="0-0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        pids = {str(k): int(v) for k, v in json.loads(Path(args.pids).read_text()).items()}
        plan = build_plan(args, pids)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.events and not wait_for_job_start(Path(args.events), args.wait_s):
        print("error: no job_started event appeared", file=sys.stderr)
        return 1
    execute(plan, pids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
