"""Scheduling policies with hand-built contexts (spec Phase 9 tests; fastest from Phase 6)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from mdc.common.config import load_config
from mdc.coordinator.energy import EnergyModel
from mdc.coordinator.job_builder import build_image_job_from_array
from mdc.coordinator.policies import milp_offline as M
from mdc.coordinator.policies.base import SchedulingContext, WorkerView
from mdc.coordinator.policies.fastest import FastestPolicy
from mdc.coordinator.policies.most_reliable import MostReliablePolicy
from mdc.coordinator.policies.tesar import TesarPolicy


def job(cols: int = 8):  # type: ignore[no-untyped-def]
    return build_image_job_from_array(np.zeros((32, cols * 32), dtype=np.uint8), 32, 1, 1, 64)


def w(wid: str, ms: float, free: int = 1, trusted: bool = False, **kw: float) -> WorkerView:
    return WorkerView(wid, wid, trusted, "phone", 1, free, bench_ms_per_unit=ms, **kw)


def ctx(j, workers, *overrides: str) -> SchedulingContext:  # type: ignore[no-untyped-def]
    return SchedulingContext(
        j.ready_tasks(), workers, j, load_config(overrides=list(overrides)), 0.0
    )


def test_fastest_picks_the_fast_device() -> None:
    j = job()
    out = FastestPolicy(load_config()).assign(ctx(j, [w("slow", 50_000), w("fast", 10)]))
    assert out[0].worker_id == "fast"
    assert out[0].predicted_s is not None and out[0].predicted_s > 0


def test_fastest_counts_queue_wait() -> None:
    j = job()
    busy = w("busy", 10, inflight_remaining_s=100.0)
    idle = w("idle", 20)
    out = FastestPolicy(load_config()).assign(ctx(j, [busy, idle]))
    assert out[0].worker_id == "idle"  # faster CPU, but 100 s of queued work


def test_fastest_counts_bandwidth() -> None:
    j = job()
    far = w("far", 1, bw_bytes_per_s=1_000.0)
    near = w("near", 50, bw_bytes_per_s=1e8)
    assert FastestPolicy(load_config()).assign(ctx(j, [far, near]))[0].worker_id == "near"


def test_fastest_fills_capacity_and_never_overlaps() -> None:
    j = job(cols=8)
    out = FastestPolicy(load_config(overrides=["chunking.max_chunk_tiles=2"])).assign(
        ctx(j, [w("a", 10, free=2), w("b", 10, free=2)], "chunking.max_chunk_tiles=2")
    )
    assert len(out) == 4
    ids = [t for a in out for t in a.task_ids]
    assert len(ids) == len(set(ids)) == 8
    assert all(len(a.task_ids) <= 2 for a in out)


def test_fastest_respects_allow_local_and_blocked() -> None:
    j = job()
    local = w("local", 1, trusted=True)
    remote = w("remote", 1000)
    no_local = load_config(overrides=["scheduler.allow_local=false"])
    assert {a.worker_id for a in FastestPolicy(no_local).assign(ctx(j, [local, remote]))} == {
        "remote"
    }
    c = ctx(j, [remote])
    c.blocked = {"s1_r00_c00"}
    out = FastestPolicy(load_config()).assign(c)
    assert all("s1_r00_c00" not in a.task_ids for a in out)
    assert FastestPolicy(load_config()).assign(ctx(j, [])) == []


def test_fastest_bigger_chunks_for_fast_stable_workers() -> None:
    j = job(cols=8)
    out = FastestPolicy(load_config()).assign(ctx(j, [w("fast", 1), w("slow", 200_000)]))
    sizes = {a.worker_id: len(a.task_ids) for a in out}
    assert sizes["fast"] > sizes.get("slow", 1)


# ------------------------------------------------- Phase 9: most_reliable, tesar, MILP

FIXED = ["chunking.adaptive=false", "chunking.fixed_chunk_tiles=1"]


def trio() -> list[WorkerView]:
    """A: fast but unreliable; B: slow but reliable; C: in between (see test below)."""
    common = {"bw_bytes_per_s": 1e9, "failure_rate_per_s": 1e-9}
    return [
        w("A", 1, rtt_s=0.0, rep=0.5, **common),
        w("B", 1, rtt_s=0.3, rep=0.999, **common),
        w("C", 1, rtt_s=0.006, rep=0.95, **common),
    ]


def tesar_choice(alpha: float, *extra: str) -> str:
    j = job()
    cfg = load_config(overrides=[*FIXED, f"scheduler.alpha={alpha}", "scheduler.replication=false"])
    c = ctx(j, trio(), *FIXED, *extra)
    c.local_bench_ms_per_unit = 1000.0  # T_loc = 3 ms per tile
    return TesarPolicy(cfg).assign(c)[0].worker_id


def test_most_reliable_picks_the_reliable_device() -> None:
    j = job()
    cfg = load_config(overrides=FIXED)
    assert FastestPolicy(cfg).assign(ctx(j, trio(), *FIXED))[0].worker_id == "A"
    out = MostReliablePolicy(cfg).assign(ctx(j, trio(), *FIXED))
    assert out[0].worker_id == "B"
    assert out[0].predicted_s is not None


def test_most_reliable_breaks_ties_by_predicted_time() -> None:
    j = job()
    cfg = load_config(overrides=FIXED)
    slow = w("slow", 1, rtt_s=0.5, rep=0.9, failure_rate_per_s=0.0)
    quick = w("quick", 1, rtt_s=0.0, rep=0.9, failure_rate_per_s=0.0)
    assert MostReliablePolicy(cfg).assign(ctx(j, [slow, quick], *FIXED))[0].worker_id == "quick"


def test_tesar_alpha_extremes_and_middle() -> None:
    j = job()
    cfg = load_config(overrides=FIXED)
    fastest = FastestPolicy(cfg).assign(ctx(j, trio(), *FIXED))[0].worker_id
    reliable = MostReliablePolicy(cfg).assign(ctx(j, trio(), *FIXED))[0].worker_id
    assert tesar_choice(1.0) == fastest == "A"
    assert tesar_choice(0.0) == reliable == "B"
    assert tesar_choice(0.5) == "C"  # neither the fastest nor the most reliable


def test_tesar_cost_formula() -> None:
    policy = TesarPolicy(load_config(overrides=["scheduler.alpha=0.25", "scheduler.kappa=4"]))
    expected = 0.25 * (2.0 / 4) / 0.5 + 0.75 * 4 * (-math.log(0.8) / 4)
    assert policy.cost(2.0, 0.8, 4, 0.5) == pytest.approx(expected)
    assert math.isfinite(policy.cost(1.0, 0.0, 1, 0.0))  # R and T_loc floors


@pytest.mark.parametrize("policy_cls", [FastestPolicy, MostReliablePolicy, TesarPolicy])
def test_infeasible_devices_are_never_chosen(policy_cls: type) -> None:
    j = job()
    cfg = load_config(overrides=FIXED)
    broke = w("broke", 1, rep=0.999, rtt_s=0.0, energy_left_j=0.0)
    low = w("low", 1, rep=0.999, rtt_s=0.0, eligible=False)
    ok = w("ok", 1000, rep=0.6, rtt_s=0.2)
    c = ctx(j, [broke, low, ok], *FIXED)
    c.energy = EnergyModel()
    out = policy_cls(cfg).assign(c)
    assert out and {a.worker_id for a in out} == {"ok"}
    c2 = ctx(j, [w("broke", 1, energy_left_j=0.0), w("low", 1, eligible=False)], *FIXED)
    c2.energy = EnergyModel()
    assert policy_cls(cfg).assign(c2) == []  # nobody can afford anything


def test_local_worker_is_exempt_from_energy_checks() -> None:
    j = job()
    cfg = load_config(overrides=FIXED)
    local = w("local", 1, trusted=True, energy_left_j=0.0, eligible=False)
    c = ctx(j, [local], *FIXED)
    c.energy = EnergyModel()
    assert TesarPolicy(cfg).assign(c)[0].worker_id == "local"


def replication_ctx(*overrides: str):  # type: ignore[no-untyped-def]
    j = job(cols=1)
    common = {"bw_bytes_per_s": 1e9, "failure_rate_per_s": 1e-9, "rtt_s": 0.0}
    workers = [
        w("p", 1, rep=0.6, **common),
        w("q", 1, rep=0.7, **common),
        w("s", 1, rep=0.8, **common),
        w("t", 1, rep=0.75, **common),
    ]
    cfg = load_config(overrides=[*FIXED, "scheduler.alpha=0.5", *overrides])
    return cfg, ctx(j, workers, *FIXED)


def test_replication_triggers_below_target_and_stops_at_max_replicas() -> None:
    cfg, c = replication_ctx("scheduler.reliability_target=0.99", "scheduler.max_replicas=2")
    out = TesarPolicy(cfg).assign(c)
    primary = [a for a in out if a.purpose == "primary"]
    replicas = [a for a in out if a.purpose == "replica"]
    assert len(primary) == 1 and len(replicas) == 2  # 0.99 not reached: capped at 2
    assert all(r.task_ids == primary[0].task_ids for r in replicas)
    devices = [a.worker_id for a in out]
    assert len(devices) == len(set(devices))  # never twice on one device


def test_replication_stops_once_the_target_is_met() -> None:
    cfg, c = replication_ctx("scheduler.reliability_target=0.9", "scheduler.max_replicas=8")
    out = TesarPolicy(cfg).assign(c)
    reps = {v.worker_id: v.rep for v in c.workers}
    chosen = [a.worker_id for a in out]
    failure = math.prod(1 - reps[d] for d in chosen)
    assert 1 - failure >= 0.9
    assert 1 - failure / (1 - reps[chosen[-1]]) < 0.9  # one replica fewer would not do


def test_no_replication_when_disabled_reliable_or_local() -> None:
    cfg, c = replication_ctx("scheduler.replication=false")
    assert [a.purpose for a in TesarPolicy(cfg).assign(c)] == ["primary"]
    cfg, c = replication_ctx("scheduler.reliability_target=0.5")  # 0.6+ already meets it
    assert [a.purpose for a in TesarPolicy(cfg).assign(c)] == ["primary"]
    j = job(cols=1)
    cfg = load_config(overrides=[*FIXED, "scheduler.alpha=0.0"])
    local_first = ctx(j, [w("local", 1, trusted=True), w("x", 1, rep=0.5)], *FIXED)
    assert [a.purpose for a in TesarPolicy(cfg).assign(local_first)] == ["primary"]


async def test_scheduler_never_uses_suspect_or_quarantined_workers() -> None:
    from mdc.coordinator.registry import WorkerState  # noqa: PLC0415
    from mdc.coordinator.scheduler import Scheduler  # noqa: PLC0415
    from tests.test_failure_detection import CFG, Harness  # noqa: PLC0415

    h = Harness(3)
    h.registry.set_state(h.registry.get("w1"), WorkerState.SUSPECT)  # type: ignore[arg-type]
    h.registry.set_state(h.registry.get("w2"), WorkerState.QUARANTINED)  # type: ignore[arg-type]
    sched = Scheduler(CFG, h.registry, h.dispatcher, TesarPolicy(CFG))
    sched.start_job(h.job)
    sched.schedule_once()
    users = {a.worker_id for a in h.job.attempts.values()}
    assert users == {"w0"}
    await h.dispatcher.drain()


DEVICES = [  # name, trusted, class, slots, bench, rtt, bw, rep, lambda, battery %
    ("local", True, "laptop", 1, 5.0, 0.0, 1e9, 0.5, 0.001, None),
    ("fast", False, "laptop", 2, 2.0, 0.01, 5e6, 0.9, 0.01, 90.0),
    ("flaky", False, "phone", 1, 1.0, 0.02, 2e6, 0.4, 0.5, 60.0),
    ("dead-battery", False, "phone", 1, 0.1, 0.0, 1e9, 0.99, 0.0, 5.0),
]


def snapshot(n_tasks: int = 8, **energy: float) -> dict:  # type: ignore[type-arg]
    keys = (
        "worker_id",
        "trusted",
        "device_class",
        "slots",
        "bench_ms_per_unit",
        "rtt_s",
        "bw_bytes_per_s",
        "rep",
        "lambda",
    )
    workers = []
    for row in DEVICES:
        dev = dict(zip(keys, row[:-1], strict=True))
        pct = row[-1]
        battery = {} if pct is None else {"pct": pct, "charging": False}
        workers.append(
            {
                **dev,
                "name": dev["worker_id"],
                "ms_per_unit": {},
                "state": "ACTIVE",
                "battery": battery,
            }
        )
    return {
        "scheduler": {"alpha": 0.5, "kappa": 10.0},
        "energy": energy,
        "min_battery_pct": 20.0,
        "workers": workers,
        "compression": {"img_tile": [0.8, 0.9]},
        "tasks": [
            {
                "task_id": f"s1_{i}",
                "stage": 1,
                "kind": "img_tile",
                "work_units": 0.3,
                "input_bytes_est": 20000,
                "output_bytes_est": 16000,
            }
            for i in range(n_tasks)
        ]
        + [{"task_id": "final", "stage": 3, "kind": "final", "work_units": 0.0}],
    }


@pytest.mark.parametrize("alpha", [0.0, 0.3, 0.5, 0.8, 1.0])
def test_milp_objective_never_worse_than_greedy(alpha: float) -> None:
    inst = M.build_instance(snapshot(), alpha=alpha)
    assert [d.name for d in inst.devices] == ["local", "fast", "flaky"]  # battery floor
    greedy = M.objective(inst, M.greedy_assignment(inst))
    best = M.objective(inst, M.solve_milp(inst))
    assert best <= greedy + 1e-7


def test_milp_respects_energy_budgets() -> None:
    inst = M.build_instance(snapshot(max_job_energy_pct=0.0000001), alpha=1.0)
    assignment = M.solve_milp(inst)
    for d, dev in enumerate(inst.devices):
        used = sum(inst.e[t][d] for t, chosen in enumerate(assignment) if chosen == d)
        assert used <= dev.budget_j + 1e-9
    assert {inst.devices[d].name for d in assignment} == {"local"}  # remote budgets ~0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.update(workers="nope"),
        lambda s: s.update(tasks=[]),
        lambda s: s["workers"][1].update(rtt_s=float("nan")),
        lambda s: s["workers"][1].update(rep="high"),
        lambda s: s.update(workers=[s["workers"][3]]),  # nothing schedulable
    ],
)
def test_milp_rejects_hostile_snapshots(mutate) -> None:  # type: ignore[no-untyped-def]
    snap = snapshot()
    mutate(snap)
    with pytest.raises(ValueError):
        M.build_instance(snap)


def test_milp_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(snapshot(6)))
    assert M.main(["--snapshot", str(path), "--alpha", "0.4", "--max-tasks", "4"]) == 0
    out = capsys.readouterr().out
    assert "4 tiles x 3 devices" in out and "MILP optimum" in out and "greedy gap" in out
    assert M.main(["--snapshot", str(tmp_path / "missing.json")]) == 2
    path.write_text("[1, 2]")
    assert M.main(["--snapshot", str(path)]) == 2
