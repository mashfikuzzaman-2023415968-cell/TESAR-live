"""Energy model, budgets, ledger, feasibility and measured energy (spec Phase 8)."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mdc.common.config import EnergyConfig, load_config
from mdc.coordinator.energy import (
    MAX_CURRENT_MA,
    MAX_SAMPLE_GAP_FACTOR,
    UNLIMITED,
    WH_TO_J,
    EnergyModel,
)
from mdc.coordinator.policies.base import Assignment, chunk_compute_s
from mdc.coordinator.policies.fastest import FastestPolicy
from mdc.coordinator.policies.round_robin import RoundRobinPolicy
from mdc.coordinator.scheduler import Scheduler
from tests.test_failure_detection import CFG, Harness, result_msg

CFG_E = EnergyConfig()


def test_task_energy_formula() -> None:
    m = EnergyModel()
    assert m.task_energy_j("phone", 2.0, 1.0) == pytest.approx(
        CFG_E.p_compute_w_phone * 2 + CFG_E.p_tx_w_phone * 1
    )
    assert m.task_energy_j("laptop", 1.0, 0.5) == pytest.approx(
        CFG_E.p_compute_w_laptop + CFG_E.p_tx_w_laptop * 0.5
    )
    assert m.task_energy_j("desktop", 1.0, 0.0) == CFG_E.p_compute_w_laptop  # unknown -> laptop
    assert m.task_energy_j("phone", -5.0, -1.0) == 0.0


def test_budget_computation() -> None:
    m = EnergyModel()
    expected = CFG_E.battery_capacity_wh_phone * WH_TO_J * 0.5 * CFG_E.max_job_energy_pct / 100
    assert m.budget_j("phone", {"pct": 50.0, "charging": False}) == pytest.approx(expected)
    assert m.budget_j("phone", {"pct": 50.0, "charging": True}) == UNLIMITED
    assert m.budget_j("laptop", {}) == UNLIMITED  # no battery information: desktop on mains
    assert m.budget_j("phone", {"pct": 500.0}) == m.budget_j("phone", {"pct": 100.0})
    assert m.budget_j("phone", {"pct": float("nan")}) == UNLIMITED
    assert EnergyModel(EnergyConfig(enabled=False)).budget_j("phone", {"pct": 1.0}) == UNLIMITED


def test_feasibility_budget_battery_charging_and_local() -> None:
    m = EnergyModel(min_battery_pct=20.0)
    low = {"pct": 15.0, "charging": False}
    ok = {"pct": 80.0, "charging": False}
    m.open_budget("w", "phone", ok, trusted=False)
    budget = m.ledger("w").budget_j
    assert m.feasible("w", ok, False, budget * 0.5)
    assert not m.feasible("w", ok, False, budget * 1.5)  # over budget
    assert not m.feasible("w", low, False, 0.001)  # battery floor
    assert m.feasible("w", {"pct": 15.0, "charging": True}, False, 0.001)  # charging bypasses
    assert m.feasible("w", low, True, 1e12)  # the trusted local worker is exempt
    assert EnergyModel(EnergyConfig(enabled=False)).feasible("w", low, False, 1e12)


def test_ledger_commit_and_settle() -> None:
    m = EnergyModel()
    m.open_budget("w", "phone", {"pct": 100.0}, trusted=False)
    budget = m.ledger("w").budget_j
    m.commit("w", 10.0)
    m.commit("w", 5.0)
    assert m.ledger("w").remaining_j == pytest.approx(budget - 15.0)
    m.settle("w", 10.0, 7.0)  # finished: measured energy replaces the prediction
    m.settle("w", 5.0, None)  # failed: charged its prediction
    entry = m.ledger("w")
    assert entry.committed_j == pytest.approx(0.0)
    assert entry.spent_j == pytest.approx(12.0)
    m.settle("w", 99.0, -3.0)  # hostile/negative values never go below zero
    assert entry.committed_j == 0.0 and entry.spent_j == pytest.approx(12.0)


def test_budget_fixed_once_per_job_and_reset_by_start_job() -> None:
    m = EnergyModel()
    first = m.open_budget("w", "phone", {"pct": 80.0}, False).budget_j
    assert m.open_budget("w", "phone", {"pct": 10.0}, False).budget_j == first  # fixed
    m.commit("w", 3.0)
    m.start_job()
    entry = m.ledger("w")
    assert entry.committed_j == 0.0 and not entry.budget_set
    assert m.open_budget("w", "phone", {"pct": 10.0}, False).budget_j < first
    assert m.open_budget("local", "laptop", {"pct": 1.0}, True).budget_j == UNLIMITED


def test_measured_energy_integration() -> None:
    m = EnergyModel(EnergyConfig(sample_interval_s=2.0))
    sample = {"current_ma": -500.0, "voltage_v": 4.0, "charging": False}
    assert m.on_battery_sample("w", sample, 10.0) == 0.0  # first sample: no interval yet
    assert m.on_battery_sample("w", sample, 12.0) == pytest.approx(0.5 * 4.0 * 2.0)  # idle
    m.start_job()
    assert m.on_battery_sample("w", sample, 13.0) == pytest.approx(0.5 * 4.0 * 1.0)
    entry = m.ledger("w")
    assert entry.measured_j == pytest.approx(2.0)
    assert entry.idle_baseline_w == pytest.approx(2.0)  # 4 J over 2 s
    gap = m.on_battery_sample("w", sample, 1013.0)  # a long gap is capped
    assert gap == pytest.approx(0.5 * 4.0 * MAX_SAMPLE_GAP_FACTOR * 2.0)
    m.end_job()
    m.on_battery_sample("w", sample, 1014.0)
    assert entry.measured_j == pytest.approx(2.0 + gap)  # after the job: not job energy


def test_measured_energy_skips_charging_and_missing_values() -> None:
    m = EnergyModel()
    m.start_job()
    m.on_battery_sample("w", {}, 1.0)
    for sample in (
        {"current_ma": 500.0, "voltage_v": 4.0, "charging": True},
        {"current_ma": None, "voltage_v": 4.0},
        {"current_ma": 500.0, "voltage_v": None},
        {"current_ma": True, "voltage_v": 4.0},
        {"current_ma": "500", "voltage_v": 4.0},
    ):
        assert m.on_battery_sample("w", sample, m.ledger("w").last_sample_t + 1.0) == 0.0  # type: ignore[operator]
    assert m.on_battery_sample("w", {"current_ma": 1.0, "voltage_v": 4.0}, 0.5) == 0.0  # clock


@pytest.mark.parametrize(
    "current, voltage",
    [
        (float("nan"), 4.0),
        (float("inf"), 4.0),
        (500.0, float("nan")),
        (500.0, -4.0),
        (1e12, 4.0),
        (500.0, 1e9),
    ],
)
def test_hostile_battery_values_are_bounded(current: float, voltage: float) -> None:
    m = EnergyModel(EnergyConfig(sample_interval_s=1.0))
    m.start_job()
    m.on_battery_sample("w", {"current_ma": current, "voltage_v": voltage}, 0.0)
    joules = m.on_battery_sample("w", {"current_ma": current, "voltage_v": voltage}, 1.0)
    assert math.isfinite(joules) and 0.0 <= joules <= MAX_CURRENT_MA / 1000 * 30.0 * 1.0


@settings(max_examples=200, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.floats(allow_nan=True, allow_infinity=True),
            st.floats(allow_nan=True, allow_infinity=True),
            st.floats(0.0, 100.0),
        ),
        max_size=30,
    )
)
def test_prop_measured_energy_finite_and_nonnegative(
    samples: list[tuple[float, float, float]],
) -> None:
    m = EnergyModel()
    m.start_job()
    t = 0.0
    for current, voltage, dt in samples:
        t += dt
        m.on_battery_sample("w", {"current_ma": current, "voltage_v": voltage}, t)
    measured = m.ledger("w").measured_j
    assert math.isfinite(measured) and measured >= 0.0


# ------------------------------------------------------------- scheduler level


def battery(h: Harness, wid: str, pct: float, charging: bool = False) -> None:
    rec = h.registry.get(wid)
    assert rec is not None
    rec.battery = {"pct": pct, "charging": charging}
    rec.device_class = "phone"


async def test_low_battery_worker_gets_no_tasks_and_is_logged_once() -> None:
    h = Harness(2)
    battery(h, "w0", 15.0)
    battery(h, "w1", 90.0)
    sched = Scheduler(CFG, h.registry, h.dispatcher, RoundRobinPolicy(CFG))
    sched.start_job(h.job)
    for _ in range(3):
        sched.schedule_once()
    w0 = h.registry.get("w0")
    assert w0 is not None and not w0.inflight
    assert h.registry.get("w1").inflight  # type: ignore[union-attr]
    logged = [e for e in h.events.recent() if e["type"] == "worker_ineligible_battery"]
    assert len(logged) == 1 and logged[0]["worker_id"] == "w0"
    await h.dispatcher.drain()


async def test_charging_bypasses_the_battery_floor() -> None:
    h = Harness(1)
    battery(h, "w0", 5.0, charging=True)
    sched = Scheduler(CFG, h.registry, h.dispatcher, RoundRobinPolicy(CFG))
    sched.start_job(h.job)
    assert sched.schedule_once() == 1
    await h.dispatcher.drain()


async def test_exhausted_budget_stops_assignments_and_is_logged() -> None:
    cfg = load_config(overrides=["workers.prefetch=0", "energy.max_job_energy_pct=0.00000001"])
    h = Harness(1, local=True)
    battery(h, "w0", 50.0)  # budget: 15 Wh * 3600 * 0.5 * 1e-10 = 2.7e-6 J, below one tile
    local = h.registry.get("local")
    assert local is not None
    local.battery = {"pct": 1.0, "charging": False}  # exempt regardless
    sched = Scheduler(cfg, h.registry, h.dispatcher, FastestPolicy(cfg))
    sched.energy.cfg = cfg.energy  # the harness dispatcher's (shared) model used CFG
    sched.start_job(h.job)
    sched.schedule_once()
    assert not h.registry.get("w0").inflight  # type: ignore[union-attr]
    assert local.inflight  # the local worker took the work
    assert "energy_budget_exhausted" in h.types()
    await h.dispatcher.drain()


async def test_scheduler_enforces_feasibility_even_for_a_careless_policy() -> None:
    class Careless:
        """Ignores energy: always sends three tiles of row 1 to w0."""

        name = "careless"

        def assign(self, ctx):  # type: ignore[no-untyped-def]
            return [Assignment(("s1_r01_c00", "s1_r01_c01", "s1_r01_c02"), "w0")]

    h = Harness(1)
    battery(h, "w0", 100.0)
    sched = Scheduler(CFG, h.registry, h.dispatcher, Careless())  # type: ignore[arg-type]
    sched.start_job(h.job)
    rec = h.registry.get("w0")
    assert rec is not None
    # Budget for about one tile: the gate (next tile only) lets w0 through, but the
    # three-tile chunk does not fit, so the scheduler must refuse it.
    one = h.dispatcher.energy.task_energy_j(
        "phone", chunk_compute_s(h.job, ("s1_r00_c00",), rec.ms_per_unit), 0.0
    )
    probe = h.dispatcher.send(Assignment(("s1_r00_c00",), "w0"))
    assert probe is not None
    one = max(one, probe.predicted_energy_j)
    h.dispatcher.cancel(probe, "measuring only")
    entry = sched.energy.open_budget("w0", "phone", rec.battery, False)
    entry.budget_j = entry.spent_j + one * 1.5
    assert sched.schedule_once() == 0
    assert not rec.inflight
    assert "energy_budget_exhausted" in h.types()
    await h.dispatcher.drain()


async def test_dispatcher_commits_and_settles_energy() -> None:
    h = Harness(1)
    battery(h, "w0", 100.0)
    attempt = h.dispatcher.send(Assignment(("s1_r00_c00",), "w0"))
    assert attempt is not None and attempt.predicted_energy_j > 0
    entry = h.dispatcher.energy.ledger("w0")
    assert entry.committed_j == pytest.approx(attempt.predicted_energy_j)
    sent = [e for e in h.events.recent() if e["type"] == "attempt_sent"]
    assert sent[-1]["predicted_energy_j"] == pytest.approx(attempt.predicted_energy_j, abs=1e-4)
    h.dispatcher.on_result(h.channels["w0"], result_msg(attempt.attempt_id, ["s1_r00_c00"], h))
    assert entry.committed_j == pytest.approx(0.0)
    assert entry.spent_j > 0
    rec = h.registry.get("w0")
    assert rec is not None and rec.energy_spent_j == entry.spent_j
    lost = h.dispatcher.send(Assignment(("s1_r00_c01",), "w0"))
    assert lost is not None
    h.dispatcher.mark_lost(lost, "test")  # failed: charged the prediction
    assert entry.spent_j == pytest.approx(rec.energy_spent_j)
    assert entry.committed_j == pytest.approx(0.0)
    await h.dispatcher.drain()


async def test_pong_battery_samples_feed_measured_energy() -> None:
    h = Harness(1)
    h.dispatcher.energy.start_job()
    pong = {
        "seq": 0,
        "t_c": 0.0,
        "battery": {"pct": 80.0, "charging": False, "current_ma": -1000.0, "voltage_v": 4.0},
    }
    h.detector.on_pong("w0", pong)
    h.clock.now += 1.0
    h.detector.on_pong("w0", pong)
    rec = h.registry.get("w0")
    assert rec is not None and rec.energy_measured_j == pytest.approx(4.0)
    await h.dispatcher.drain()


def test_report_has_energy_fields() -> None:
    from mdc.coordinator.estimators import Estimators  # noqa: PLC0415
    from mdc.coordinator.metrics import RunTiming, build_report  # noqa: PLC0415

    h = Harness(1)
    battery(h, "w0", 60.0)
    est = h.dispatcher.estimators
    est.energy.open_budget("w0", "phone", {"pct": 60.0}, False)
    report = build_report(
        h.job, CFG, h.registry, h.dispatcher.traffic, RunTiming(1.0, 2.0), {}, 1, est
    )
    w0 = report["per_worker"]["w0"]
    assert {
        "energy_pred_j",
        "energy_meas_j",
        "energy_budget_j",
        "idle_baseline_w",
        "charging",
    } <= set(w0)
    assert w0["energy_budget_j"] > 0 and w0["charging"] is False
    assert isinstance(est, Estimators)
