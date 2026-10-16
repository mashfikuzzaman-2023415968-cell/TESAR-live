"""Offline MILP optimum for small instances (spec Phase 9 item 4, experiment E7).

Not a live policy: given the estimator snapshot a coordinator writes at job
start (``runs/<run>/snapshot.json``), it computes the best possible static
assignment of the stage-1 tiles (one tile per task, no chunking, no
replication) and compares it with the greedy TESAR rule on the *same*
predictions. The gap shows how much the online greedy loses.

Model (all coefficients come from the snapshot)::

    c[t,d] = rtt_d + (in_t + out_t) / bw_d + work_t * ms_per_unit_d / 1000
    e[t,d] = p_compute(d) * compute + p_tx(d) * (rtt_d + bytes / bw_d)
    R[t,d] = rep_d * exp(-λ_d * c[t,d])            (0.9999 for the trusted local)

    variables  x[t,d] in {0, 1},  M >= 0
    Σ_d x[t,d] = 1                         for every task t
    Σ_t x[t,d] * c[t,d] <= M * slots_d     for every device d
    Σ_t x[t,d] * e[t,d] <= budget_d        for every device d with a finite budget
    minimize  alpha * M / M_ref + (1 - alpha) * kappa * Σ x[t,d] * (-ln R[t,d]) / N

``M_ref`` is the time to run every task on the local worker and ``N`` the number
of tasks. Because ``-ln R`` is additive (see ``tesar.py``), the objective is
linear. The greedy comparison assigns tasks in order to the device with the
smallest ``alpha * (queue + c) / T_loc + (1 - alpha) * kappa * (-ln R)``, which
is the live TESAR cost with chunks of one tile.

Usage::

    python -m mdc.coordinator.policies.milp_offline --snapshot runs/<run>/snapshot.json --alpha 0.5

Requires the optional ``pulp`` package and a solver (HiGHS via ``highspy``, or
CBC); keep instances small (default: the first 60 tiles, at most 8 devices).
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mdc.common.config import EnergyConfig
from mdc.coordinator.energy import UNLIMITED, EnergyModel
from mdc.coordinator.reputation import battery_eligible, task_reliability

MAX_SNAPSHOT_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_TASKS = 60
MAX_DEVICES = 8
MIN_RELIABILITY = 1e-6
EXIT_USAGE = 2


class MilpUnavailableError(RuntimeError):
    """pulp or a MILP solver is not installed."""


@dataclass(frozen=True)
class Device:
    """One worker as seen in the snapshot."""

    worker_id: str
    name: str
    trusted: bool
    device_class: str
    slots: int
    ms_per_unit: dict[str, float]
    bench_ms_per_unit: float
    rtt_s: float
    bw_bytes_per_s: float
    rep: float
    failure_rate: float
    budget_j: float

    def speed(self, kind: str) -> float:
        """ms per work unit for ``kind`` (measured, else benchmark)."""
        return self.ms_per_unit.get(kind, self.bench_ms_per_unit)


@dataclass(frozen=True)
class TaskSpec:
    """One stage-1 tile."""

    task_id: str
    kind: str
    work_units: float
    in_bytes: float
    out_bytes: float


@dataclass
class Instance:
    """Devices, tasks and the predicted cost matrices."""

    devices: list[Device]
    tasks: list[TaskSpec]
    alpha: float
    kappa: float
    c: list[list[float]]  # c[t][d] seconds
    e: list[list[float]]  # e[t][d] joules
    r: list[list[float]]  # R[t][d]
    t_loc: list[float]  # local compute time per task
    m_ref: float


# ------------------------------------------------------------------ loading


def _number(raw: Any, what: str, low: float = 0.0, high: float = 1e12) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
        raise ValueError(f"{what} must be a finite number")
    return min(high, max(low, float(raw)))


def _speeds(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    return {
        str(k): _number(v, "ms_per_unit", 1e-9)
        for k, v in raw.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    }


def load_snapshot(path: Path) -> dict[str, Any]:
    """Read a ``snapshot.json`` (size-limited, must be a JSON object)."""
    if path.stat().st_size > MAX_SNAPSHOT_BYTES:
        raise ValueError(f"{path} is larger than {MAX_SNAPSHOT_BYTES} bytes")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("a snapshot must be a JSON object")
    return data


def build_instance(
    snapshot: dict[str, Any],
    alpha: float | None = None,
    kappa: float | None = None,
    max_tasks: int = DEFAULT_MAX_TASKS,
    include_local: bool = True,
) -> Instance:
    """Turn a snapshot into an :class:`Instance` (raises ``ValueError`` on bad input)."""
    raw_sched = snapshot.get("scheduler")
    sched: dict[str, Any] = raw_sched if isinstance(raw_sched, dict) else {}
    a = _number(alpha if alpha is not None else sched.get("alpha", 0.5), "alpha", 0.0, 1.0)
    k = _number(kappa if kappa is not None else sched.get("kappa", 10.0), "kappa", 0.0, 1e6)
    raw_energy = snapshot.get("energy")
    energy_cfg = EnergyConfig()
    if isinstance(raw_energy, dict):
        known = {f for f in EnergyConfig.__dataclass_fields__ if f in raw_energy}
        energy_cfg = EnergyConfig(**{f: raw_energy[f] for f in known})
    min_battery = _number(snapshot.get("min_battery_pct", 20.0), "min_battery_pct", 0, 100)
    energy = EnergyModel(energy_cfg, min_battery)
    devices = _devices(snapshot, energy, min_battery, include_local)
    tasks = _tasks(snapshot, max_tasks)
    compression = _compression(snapshot)
    local = next((d for d in devices if d.trusted), None)
    fallback_bench = (
        local.bench_ms_per_unit
        if local is not None
        else _number(snapshot.get("local_bench_ms_per_unit", 1.0), "local bench", 1e-9)
    )
    c, e, r = [], [], []
    for task in tasks:
        ratio_in, ratio_out = compression.get(task.kind, (1.0, 1.0))
        wire = task.in_bytes * ratio_in + task.out_bytes * ratio_out
        c_row, e_row, r_row = [], [], []
        for d in devices:
            compute = task.work_units * d.speed(task.kind) / 1000.0
            transfer = d.rtt_s + wire / max(1.0, d.bw_bytes_per_s)
            c_row.append(compute + transfer)
            e_row.append(energy.task_energy_j(d.device_class, compute, transfer))
            r_row.append(
                task_reliability(d.rep, d.failure_rate, compute + transfer, d.trusted, True)
            )
        c.append(c_row)
        e.append(e_row)
        r.append(r_row)
    t_loc = [
        t.work_units * (local.speed(t.kind) if local else fallback_bench) / 1000.0 for t in tasks
    ]
    return Instance(devices, tasks, a, k, c, e, r, t_loc, max(sum(t_loc), 1e-9))


def _devices(
    snapshot: dict[str, Any], energy: EnergyModel, min_battery: float, include_local: bool
) -> list[Device]:
    raw = snapshot.get("workers")
    if not isinstance(raw, list):
        raise ValueError("snapshot has no 'workers' list")
    devices = []
    for w in raw:
        if not isinstance(w, dict):
            raise ValueError("every worker must be an object")
        trusted = w.get("trusted") is True
        if (trusted and not include_local) or w.get("state") not in {None, "ACTIVE"}:
            continue
        raw_battery = w.get("battery")
        battery: dict[str, Any] = raw_battery if isinstance(raw_battery, dict) else {}
        if not trusted and not battery_eligible(battery, min_battery):
            continue  # below the battery floor: never schedulable
        device_class = str(w.get("device_class", "laptop"))
        devices.append(
            Device(
                worker_id=str(w.get("worker_id", "")),
                name=str(w.get("name", "")),
                trusted=trusted,
                device_class=device_class,
                slots=int(_number(w.get("slots", 1), "slots", 1, 64)),
                ms_per_unit=_speeds(w.get("ms_per_unit")),
                bench_ms_per_unit=_number(w.get("bench_ms_per_unit", 1.0), "bench", 1e-9),
                rtt_s=_number(w.get("rtt_s", 0.0), "rtt_s", 0.0, 600.0),
                bw_bytes_per_s=_number(w.get("bw_bytes_per_s", 1e6), "bw", 1.0, 1e10),
                rep=_number(w.get("rep", 0.5), "rep", 0.0, 1.0),
                failure_rate=_number(w.get("lambda", 0.001), "lambda", 0.0, 1e6),
                budget_j=UNLIMITED if trusted else energy.budget_j(device_class, battery),
            )
        )
    if not devices:
        raise ValueError("no schedulable device in the snapshot")
    return devices[:MAX_DEVICES]


def _tasks(snapshot: dict[str, Any], max_tasks: int) -> list[TaskSpec]:
    raw = snapshot.get("tasks")
    if not isinstance(raw, list):
        raise ValueError("snapshot has no 'tasks' list")
    tasks = [
        TaskSpec(
            task_id=str(t.get("task_id", "")),
            kind=str(t.get("kind", "img_tile")),
            work_units=_number(t.get("work_units", 0.0), "work_units"),
            in_bytes=_number(t.get("input_bytes_est", 0.0), "input_bytes_est"),
            out_bytes=_number(t.get("output_bytes_est", 0.0), "output_bytes_est"),
        )
        for t in raw
        if isinstance(t, dict) and t.get("stage") == 1
    ]
    if not tasks:
        raise ValueError("snapshot has no stage-1 tasks")
    return tasks[: max(1, max_tasks)]


def _compression(snapshot: dict[str, Any]) -> dict[str, tuple[float, float]]:
    raw = snapshot.get("compression")
    out: dict[str, tuple[float, float]] = {}
    if isinstance(raw, dict):
        for kind, pair in raw.items():
            if isinstance(pair, (list, tuple)) and len(pair) == 2:  # noqa: PLR2004
                out[str(kind)] = (_number(pair[0], "ratio", 0, 2), _number(pair[1], "ratio", 0, 2))
    return out


# ----------------------------------------------------------------- objective


def objective(inst: Instance, assignment: list[int]) -> float:
    """``alpha * M / M_ref + (1 - alpha) * kappa * mean(-ln R)`` of a full assignment."""
    load = [0.0] * len(inst.devices)
    unreliability = 0.0
    for t, d in enumerate(assignment):
        load[d] += inst.c[t][d]
        unreliability += -math.log(max(inst.r[t][d], MIN_RELIABILITY))
    makespan = max(load[d] / inst.devices[d].slots for d in range(len(inst.devices)))
    n = len(inst.tasks)
    return inst.alpha * makespan / inst.m_ref + (1 - inst.alpha) * inst.kappa * unreliability / n


def makespan_of(inst: Instance, assignment: list[int]) -> float:
    """``max_d load_d / slots_d`` of an assignment."""
    load = [0.0] * len(inst.devices)
    for t, d in enumerate(assignment):
        load[d] += inst.c[t][d]
    return max(load[d] / inst.devices[d].slots for d in range(len(inst.devices)))


def greedy_assignment(inst: Instance) -> list[int]:
    """The live TESAR rule applied statically with one-tile chunks and queue waits."""
    load = [0.0] * len(inst.devices)
    spent = [0.0] * len(inst.devices)
    out = []
    for t in range(len(inst.tasks)):
        best: tuple[float, str, int] | None = None
        for d, dev in enumerate(inst.devices):
            if spent[d] + inst.e[t][d] > dev.budget_j:
                continue
            finish = load[d] / dev.slots + inst.c[t][d]
            cost = inst.alpha * finish / max(inst.t_loc[t], 1e-9) + (
                1 - inst.alpha
            ) * inst.kappa * -math.log(max(inst.r[t][d], MIN_RELIABILITY))
            if best is None or (cost, dev.worker_id) < best[:2]:
                best = (cost, dev.worker_id, d)
        if best is None:
            raise ValueError(f"no device can afford task {inst.tasks[t].task_id}")
        d = best[2]
        out.append(d)
        load[d] += inst.c[t][d]
        spent[d] += inst.e[t][d]
    return out


# ---------------------------------------------------------------------- MILP


def _pulp() -> Any:
    """The optional ``pulp`` module, typed as ``Any``: pulp 3 (Python < 3.12 in our
    lock) and pulp 4 have different APIs and type stubs; :func:`solve_milp` uses
    the small common subset through :func:`_var` and :func:`_optimal`."""
    try:
        return importlib.import_module("pulp")
    except ImportError as exc:
        raise MilpUnavailableError("pulp is not installed (pip install pulp highspy)") from exc


def _solver(pulp: Any, time_limit_s: float) -> Any:
    available = set(pulp.listSolvers(onlyAvailable=True))
    for name in ("HiGHS", "PULP_CBC_CMD", "COIN_CMD"):
        if name in available:
            return getattr(pulp, name)(msg=False, timeLimit=time_limit_s)
    raise MilpUnavailableError("pulp has no MILP solver available (pip install highspy)")


def _var(pulp: Any, prob: Any, name: str, binary: bool) -> Any:
    cat = "Binary" if binary else "Continuous"
    if hasattr(prob, "add_variable"):  # pulp >= 4
        return prob.add_variable(name, lowBound=0, cat=cat)
    return pulp.LpVariable(name, lowBound=0, cat=cat)


def _optimal(result: Any) -> bool:
    """pulp 3 returns a status code, pulp 4 a stats object with an enum status."""
    status = getattr(result, "status", result)
    return int(getattr(status, "value", status)) == 1


def solve_milp(inst: Instance, time_limit_s: float = 60.0) -> list[int]:
    """Optimal assignment of the model above (raises ``MilpUnavailableError``)."""
    pulp = _pulp()
    solver = _solver(pulp, time_limit_s)
    tasks, devs = range(len(inst.tasks)), range(len(inst.devices))
    prob = pulp.LpProblem("tesar_offline", pulp.LpMinimize)
    x = {(t, d): _var(pulp, prob, f"x_{t}_{d}", binary=True) for t in tasks for d in devs}
    m = _var(pulp, prob, "M", binary=False)
    cost = [[-math.log(max(inst.r[t][d], MIN_RELIABILITY)) for d in devs] for t in tasks]
    rel = pulp.lpSum(x[t, d] * cost[t][d] for t in tasks for d in devs)
    n = len(inst.tasks)
    prob += inst.alpha * m / inst.m_ref + (1 - inst.alpha) * inst.kappa * rel / n
    for t in tasks:
        prob += pulp.lpSum(x[t, d] for d in devs) == 1
    for d in devs:
        dev = inst.devices[d]
        prob += pulp.lpSum(x[t, d] * inst.c[t][d] for t in tasks) <= m * dev.slots
        if math.isfinite(dev.budget_j):
            prob += pulp.lpSum(x[t, d] * inst.e[t][d] for t in tasks) <= dev.budget_j
    if not _optimal(prob.solve(solver)):
        raise ValueError("the MILP was not solved to optimality")
    return [max(devs, key=lambda d: x[t, d].value() or 0.0) for t in tasks]


# ----------------------------------------------------------------------- CLI


def compare(inst: Instance, time_limit_s: float = 60.0) -> dict[str, Any]:
    """Greedy vs MILP on one instance."""
    greedy = greedy_assignment(inst)
    best = solve_milp(inst, time_limit_s)
    g_obj, m_obj = objective(inst, greedy), objective(inst, best)
    return {
        "tasks": len(inst.tasks),
        "devices": [d.name for d in inst.devices],
        "alpha": inst.alpha,
        "kappa": inst.kappa,
        "greedy_objective": g_obj,
        "milp_objective": m_obj,
        "gap_pct": 100.0 * (g_obj - m_obj) / m_obj if m_obj > 0 else 0.0,
        "greedy_makespan_s": makespan_of(inst, greedy),
        "milp_makespan_s": makespan_of(inst, best),
    }


def main(argv: list[str] | None = None) -> int:
    """``python -m mdc.coordinator.policies.milp_offline --snapshot FILE``."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    p.add_argument("--snapshot", type=Path, required=True, help="runs/<run>/snapshot.json")
    p.add_argument("--alpha", type=float, default=None, help="default: the run's alpha")
    p.add_argument("--kappa", type=float, default=None, help="default: the run's kappa")
    p.add_argument("--max-tasks", type=int, default=DEFAULT_MAX_TASKS)
    p.add_argument("--no-local", action="store_true", help="leave out the trusted local worker")
    p.add_argument("--time-limit-s", type=float, default=60.0)
    args = p.parse_args(argv)
    try:
        inst = build_instance(
            load_snapshot(args.snapshot),
            args.alpha,
            args.kappa,
            args.max_tasks,
            include_local=not args.no_local,
        )
        result = compare(inst, args.time_limit_s)
    except (OSError, ValueError, MilpUnavailableError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    print(
        f"instance: {result['tasks']} tiles x {len(result['devices'])} devices "
        f"({', '.join(result['devices'])}), alpha={result['alpha']}, kappa={result['kappa']}"
    )
    print(
        f"greedy TESAR objective: {result['greedy_objective']:.6f} "
        f"(makespan {result['greedy_makespan_s']:.3f} s)"
    )
    print(
        f"MILP optimum:           {result['milp_objective']:.6f} "
        f"(makespan {result['milp_makespan_s']:.3f} s)"
    )
    print(f"greedy gap: {result['gap_pct']:.2f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
