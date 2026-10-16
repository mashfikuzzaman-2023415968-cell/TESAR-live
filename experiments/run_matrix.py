#!/usr/bin/env python3
"""Run an experiment: every matrix point x repetition on a local cluster (spec Phase 13).

Usage::

    python experiments/run_matrix.py configs/experiments/e2_alpha_sweep.yaml [--dry-run] [--only N]

Config format (``configs/experiments/*.yaml``)::

    name: e2_alpha_sweep                    # [a-z0-9_]+, results go to runs/<name>/
    cluster_profile: configs/cluster_profiles/hetero_faulty6.yaml
    repetitions: 5
    seed: 42                                # run i uses seed + i (coordinator and workers)
    timeout_s: 300                          # per job
    local_baseline: true                    # also run policy=local once per repetition
    base_overrides:                         # --set for every run
      - job.image_path=data/images/test_1024.png
    matrix:                                 # cartesian product
      scheduler.policy: [tesar]
      scheduler.alpha: [0.0, 0.5, 1.0]
      cluster.drop_prob: [0.0, 0.2]         # cluster.<flag>: worker option --<flag> for all workers

For each run: a fresh port and a fresh temporary state directory, the
coordinator (``--headless --run-job --exit-after-job``) and
``scripts/local_cluster.py`` with the same seed, outputs in
``runs/<name>/<point_id>/rep<i>/``. Runs whose ``report.json`` exists are
skipped (resume). One row per run is appended to ``runs/<name>/results.csv``.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
NAME_RE = re.compile(r"^[a-z0-9_]{1,64}$")
KEY_RE = re.compile(r"^[a-z_]+\.[a-z_]+$")
MAX_REPETITIONS = 100
CLUSTER_PREFIX = "cluster."
BASELINE_ID = "local_baseline"
REPORT_FIELDS = (
    "job_id",
    "policy",
    "alpha",
    "kappa",
    "adaptive_chunking",
    "verification_enabled",
    "replication",
    "seed",
    "image",
    "blur_iterations",
    "n_tiles",
    "n_workers_remote",
    "makespan_s",
    "local_estimate_s",
    "speedup",
    "attempts_total",
    "attempts_ok",
    "attempts_timeout",
    "attempts_lost",
    "attempts_error",
    "attempts_late",
    "wasted_work_s",
    "replicas",
    "verifications",
    "verification_mismatches",
    "verification_overhead_pct",
    "detection_latency_s",
    "wrong_results_reached_final",
    "quarantined",
    "output_correct",
    "differing_pixels",
    "messages_sent",
    "bytes_sent",
    "bytes_received",
    "avg_chunk_tiles",
)
RUN_FIELDS = ("experiment", "point_id", "rep", "run_seed", "status", "run_dir")


class ExperimentError(ValueError):
    """Invalid experiment configuration."""


@dataclass
class Experiment:
    """A parsed experiment config."""

    name: str
    cluster_profile: Path
    repetitions: int
    seed: int = 42
    timeout_s: float = 300.0
    local_baseline: bool = False
    base_overrides: list[str] = field(default_factory=list)
    matrix: dict[str, list[Any]] = field(default_factory=dict)

    @property
    def matrix_keys(self) -> list[str]:
        return list(self.matrix)


@dataclass(frozen=True)
class Run:
    """One planned run."""

    point_id: str
    rep: int
    params: dict[str, Any]
    seed: int

    def out_dir(self, runs_root: Path, name: str) -> Path:
        return runs_root / name / self.point_id / f"rep{self.rep}"


def load_experiment(path: Path) -> Experiment:
    """Parse and validate an experiment YAML (``yaml.safe_load`` only)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ExperimentError("an experiment config must be a mapping")
    allowed = {
        "name",
        "cluster_profile",
        "repetitions",
        "seed",
        "timeout_s",
        "local_baseline",
        "base_overrides",
        "matrix",
    }
    unknown = set(data) - allowed
    if unknown:
        raise ExperimentError(f"unknown keys: {sorted(unknown)}")
    name = data.get("name")
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ExperimentError("name must match [a-z0-9_]{1,64}")
    profile = data.get("cluster_profile")
    if not isinstance(profile, str):
        raise ExperimentError("cluster_profile is required")
    reps = data.get("repetitions", 1)
    if type(reps) is not int or not 1 <= reps <= MAX_REPETITIONS:
        raise ExperimentError("repetitions must be an int in 1..100")
    seed = data.get("seed", 42)
    timeout = data.get("timeout_s", 300)
    if type(seed) is not int or isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ExperimentError("seed must be an int and timeout_s a number")
    overrides = data.get("base_overrides", []) or []
    if not isinstance(overrides, list) or not all(
        isinstance(o, str) and "=" in o for o in overrides
    ):
        raise ExperimentError("base_overrides must be a list of key=value strings")
    matrix = data.get("matrix", {}) or {}
    if not isinstance(matrix, dict):
        raise ExperimentError("matrix must be a mapping")
    for key, values in matrix.items():
        if not isinstance(key, str) or not KEY_RE.match(key):
            raise ExperimentError(f"bad matrix key {key!r} (section.name)")
        if not isinstance(values, list) or not values:
            raise ExperimentError(f"matrix values of {key} must be a non-empty list")
    return Experiment(
        name=name,
        cluster_profile=(ROOT / profile) if not Path(profile).is_absolute() else Path(profile),
        repetitions=reps,
        seed=seed,
        timeout_s=float(timeout),
        local_baseline=bool(data.get("local_baseline", False)),
        base_overrides=list(overrides),
        matrix={k: list(v) for k, v in matrix.items()},
    )


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def point_id(index: int, params: dict[str, Any]) -> str:
    """A readable, filesystem-safe id like ``p03_alpha-0.5``."""
    parts = [f"{k.split('.')[-1]}-{_fmt(v)}" for k, v in params.items()]
    slug = re.sub(r"[^A-Za-z0-9.\-_]+", "_", "_".join(parts))[:80]
    return f"p{index:02d}" + (f"_{slug}" if slug else "")


def plan(exp: Experiment) -> list[Run]:
    """Every run in execution order (repetition-major, so partial results are balanced)."""
    keys = exp.matrix_keys
    points = [
        dict(zip(keys, combo, strict=True)) for combo in itertools.product(*exp.matrix.values())
    ]
    if not points:
        points = [{}]
    runs = []
    for rep in range(exp.repetitions):
        for index, params in enumerate(points):
            runs.append(Run(point_id(index, params), rep, params, exp.seed + rep))
        if exp.local_baseline:
            runs.append(Run(BASELINE_ID, rep, {"scheduler.policy": "local"}, exp.seed + rep))
    return runs


def split_params(params: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(coordinator ``--set`` overrides, worker extra args) for one run."""
    sets, extra = [], []
    for key, value in params.items():
        if key.startswith(CLUSTER_PREFIX):
            flag = "--" + key[len(CLUSTER_PREFIX) :].replace("_", "-")
            if value not in (None, False, 0, 0.0, ""):
                extra += [flag] if value is True else [flag, _fmt(value)]
        else:
            sets.append(f"{key}={_fmt(value)}")
    return sets, extra


def count_workers(profile: Path) -> int:
    data = yaml.safe_load(profile.read_text(encoding="utf-8"))
    workers = data.get("workers") if isinstance(data, dict) else None
    if not isinstance(workers, list) or not workers:
        raise ExperimentError(f"{profile} has no workers")
    return len(workers)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def execute(exp: Experiment, run: Run, out: Path, n_workers: int) -> str:
    """Run one job; return ``ok``, ``incorrect`` or ``failed``."""
    py = sys.executable
    sets, extra = split_params(run.params)
    port = free_port()
    with tempfile.TemporaryDirectory(prefix="mdc-exp-") as tmp:
        state = Path(tmp) / "state"
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        env.pop("MDC_JOIN_CODE", None)
        overrides = [
            f"security.state_dir={state}",
            f"network.tcp_port={port}",
            "network.coordinator_bind=127.0.0.1",
            "network.beacon_modes=[]",
            f"network.max_connections_per_ip={n_workers + 4}",
            f"experiment.seed={run.seed}",
            *exp.base_overrides,
            *sets,
        ]
        out.mkdir(parents=True, exist_ok=True)
        coord_cmd = [
            py,
            "-m",
            "mdc.coordinator",
            "--headless",
            "--run-job",
            "--exit-after-job",
            "--wait-workers",
            str(n_workers),
            "--wait-timeout-s",
            "60",
            "--job-timeout-s",
            str(exp.timeout_s),
            "--output-dir",
            str(out),
            *[a for kv in overrides for a in ("--set", kv)],
        ]
        cluster_cmd = [
            py,
            str(ROOT / "scripts" / "local_cluster.py"),
            str(exp.cluster_profile),
            "--coordinator",
            f"127.0.0.1:{port}",
            "--state-dir",
            str(state),
            "--runs-dir",
            str(Path(tmp) / "cluster"),
            "--seed",
            str(run.seed),
            "--duration",
            str(exp.timeout_s + 90),
            "--extra-args",
            shlex.join(extra),
        ]
        with (out / "coordinator.stdout").open("wb") as log:
            coord = subprocess.Popen(
                coord_cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        cluster = None
        try:
            time.sleep(1.0)  # the coordinator creates the certificate the cluster pins
            with (out / "cluster.stdout").open("wb") as log:
                cluster = subprocess.Popen(
                    cluster_cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
                )
            coord.wait(timeout=exp.timeout_s + 120)
        except subprocess.TimeoutExpired:
            pass
        finally:
            for proc in (coord, cluster):
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(20)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
    report = out / "report.json"
    if not report.exists():
        return "failed"
    return "ok" if json.loads(report.read_text()).get("output_correct") else "incorrect"


def csv_row(exp: Experiment, run: Run, status: str, out: Path) -> dict[str, Any]:
    """One results.csv row: run identity, matrix parameters, report metrics."""
    row: dict[str, Any] = {
        "experiment": exp.name,
        "point_id": run.point_id,
        "rep": run.rep,
        "run_seed": run.seed,
        "status": status,
        "run_dir": str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out),
    }
    for key in exp.matrix_keys:
        row[key] = _fmt(run.params[key]) if key in run.params else ""
    report_path = out / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for key in REPORT_FIELDS:
        value = report.get(key)
        row[key] = json.dumps(value) if isinstance(value, (list, dict)) else value
    return row


def columns(exp: Experiment) -> list[str]:
    return [*RUN_FIELDS, *exp.matrix_keys, *REPORT_FIELDS]


def append_row(path: Path, exp: Experiment, row: dict[str, Any]) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns(exp))
        if new:
            writer.writeheader()
        writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run an experiment matrix")
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="print the plan only")
    parser.add_argument("--only", type=int, default=None, help="run only the first N runs")
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    args = parser.parse_args(argv)
    try:
        exp = load_experiment(args.config)
        n_workers = count_workers(exp.cluster_profile)
    except (OSError, ExperimentError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    runs = plan(exp)
    if args.only is not None:
        runs = runs[: max(0, args.only)]
    results = args.runs_root / exp.name / "results.csv"
    print(f"{exp.name}: {len(runs)} run(s), {n_workers} workers, results -> {results}")
    for i, run in enumerate(runs, start=1):
        out = run.out_dir(args.runs_root, exp.name)
        sets, extra = split_params(run.params)
        label = f"[{i}/{len(runs)}] {run.point_id} rep{run.rep} seed {run.seed}"
        if args.dry_run:
            print(f"{label}: --set {' '.join(sets) or '-'} | workers {' '.join(extra) or '-'}")
            continue
        if (out / "report.json").exists():
            print(f"{label}: done earlier, skipped")
            continue
        t0 = time.monotonic()
        status = execute(exp, run, out, n_workers)
        append_row(results, exp, csv_row(exp, run, status, out))
        print(f"{label}: {status} in {time.monotonic() - t0:.1f} s", flush=True)
        time.sleep(1.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
