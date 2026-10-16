"""Beta reputation, failure rate λ and task reliability R (spec Phase 7)."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mdc.common.config import ReputationConfig, load_config
from mdc.coordinator.reputation import (
    FAILURE_KINDS,
    MIN_ATTEMPTS_FOR_RATE,
    TRUSTED_RELIABILITY,
    Reputation,
    battery_eligible,
    task_reliability,
)

EVENTS = st.lists(st.sampled_from(["success", *sorted(FAILURE_KINDS)]), max_size=60)


def apply(rep: Reputation, events: list[str], wid: str = "w") -> None:
    for event in events:
        if event == "success":
            rep.record_success(wid)
        else:
            rep.record_failure(wid, event)


def test_new_worker_has_neutral_reputation() -> None:
    rep = Reputation()
    assert rep.rep("new") == pytest.approx(0.5)
    snap = rep.snapshot("new")
    assert snap["s"] == snap["f"] == 1.0
    assert snap["lambda"] == ReputationConfig().default_failure_rate_per_s


def test_successes_raise_reputation() -> None:
    rep = Reputation()
    values = []
    for _ in range(10):
        rep.record_success("w")
        values.append(rep.rep("w"))
    assert values == sorted(values) and values[-1] > 0.85


def test_wrong_result_hurts_much_more_than_timeout() -> None:
    a, b = Reputation(), Reputation()
    for r in (a, b):
        apply(r, ["success"] * 5)
    start = a.rep("w")
    a.record_failure("w", "timeout")
    b.record_failure("w", "wrong_result")
    drop_timeout, drop_wrong = start - a.rep("w"), start - b.rep("w")
    assert drop_timeout > 0 and drop_wrong > 3 * drop_timeout


def test_decay_makes_old_failures_matter_less() -> None:
    no_decay = Reputation(ReputationConfig(decay=1.0))
    decay = Reputation(ReputationConfig(decay=0.8))
    for r in (no_decay, decay):
        apply(r, ["wrong_result", "wrong_result"] + ["success"] * 15)
    assert decay.rep("w") > no_decay.rep("w")


def test_counts_by_type_and_unknown_kind() -> None:
    rep = Reputation()
    apply(rep, ["success", "timeout", "timeout", "disconnect", "error", "wrong_result"])
    snap = rep.snapshot("w")
    assert snap["successes"] == 1
    assert snap["failures"] == {"timeout": 2, "disconnect": 1, "error": 1, "wrong_result": 1}
    with pytest.raises(ValueError, match="unknown failure kind"):
        rep.record_failure("w", "bogus")


def test_failure_rate_default_until_three_attempts() -> None:
    cfg = ReputationConfig()
    rep = Reputation(cfg)
    for _ in range(MIN_ATTEMPTS_FOR_RATE - 1):
        rep.record_attempt("w", True, 0.5)
        assert rep.failure_rate("w") == cfg.default_failure_rate_per_s
    rep.record_attempt("w", True, 0.5)
    assert rep.failure_rate("w") > 1.0  # every attempt failed


def test_failure_rate_increases_with_failures() -> None:
    rates = []
    for n_failed in range(6):
        rep = Reputation()
        for i in range(10):
            rep.record_attempt("w", i < n_failed, 1.0)
        rates.append(rep.failure_rate("w"))
    assert rates == sorted(rates) and rates[0] < rates[-1]
    assert rates[0] == pytest.approx(1e-6)  # never failed: the floor


def test_failure_rate_matches_poisson_model() -> None:
    """With failure fraction p over attempts of length T: exp(-λT) = 1 - p."""
    rep = Reputation(ReputationConfig(failure_rate_ewma_alpha=1.0))  # last sample only
    for _ in range(MIN_ATTEMPTS_FOR_RATE):
        rep.record_attempt("w", True, 2.0)
    lam = rep.failure_rate("w")
    assert math.exp(-lam * 2.0) == pytest.approx(1 - 0.999)


def test_reliability_decreases_with_predicted_time() -> None:
    rep = Reputation()
    for i in range(10):
        rep.record_attempt("w", i % 3 == 0, 1.0)
    values = [rep.reliability("w", t) for t in (0.0, 0.5, 1.0, 5.0, 50.0)]
    assert values == sorted(values, reverse=True)
    assert values[0] == pytest.approx(rep.rep("w"))


def test_reliability_special_cases() -> None:
    rep = Reputation()
    assert rep.reliability("w", 3.0, trusted=True) == TRUSTED_RELIABILITY
    assert rep.reliability("w", 3.0, eligible=False) == 0.0
    assert task_reliability(0.8, 0.0, 100.0, False, True) == pytest.approx(0.8)


def test_battery_eligibility() -> None:
    assert battery_eligible({}, 20)  # desktop without a battery
    assert battery_eligible({"pct": None}, 20)
    assert battery_eligible({"pct": 50.0, "charging": False}, 20)
    assert not battery_eligible({"pct": 15.0, "charging": False}, 20)
    assert battery_eligible({"pct": 15.0, "charging": True}, 20)  # charging bypasses the floor


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, -float("inf")])
def test_hostile_predicted_times_are_harmless(bad: float) -> None:
    rep = Reputation()
    for _ in range(5):
        rep.record_attempt("w", True, bad)
    lam = rep.failure_rate("w")
    assert math.isfinite(lam) and lam > 0
    r = rep.reliability("w", bad if math.isfinite(bad) else 1.0)
    assert 0.0 <= r <= 1.0


def test_reputation_uses_config_weights() -> None:
    cfg = load_config(overrides=["reputation.weight_timeout=0", "reputation.decay=1"])
    rep = Reputation(cfg.reputation)
    for _ in range(5):
        rep.record_failure("w", "timeout")
    assert rep.rep("w") == pytest.approx(0.5)  # weight 0: timeouts are forgiven


@settings(max_examples=200, deadline=None)
@given(EVENTS, st.floats(0.5, 1.0))
def test_prop_rep_always_strictly_between_0_and_1(events: list[str], decay: float) -> None:
    rep = Reputation(ReputationConfig(decay=decay))
    apply(rep, events)
    assert 0.0 < rep.rep("w") < 1.0


@settings(max_examples=200, deadline=None)
@given(EVENTS, st.floats(0.5, 1.0))
def test_prop_success_never_lowers_and_failure_never_raises(
    events: list[str], decay: float
) -> None:
    rep = Reputation(ReputationConfig(decay=decay))
    apply(rep, events)
    before = rep.rep("w")
    rep.record_success("w")
    assert rep.rep("w") >= before - 1e-12
    for kind in sorted(FAILURE_KINDS):
        before = rep.rep("w")
        rep.record_failure("w", kind)
        assert rep.rep("w") <= before + 1e-12


@settings(max_examples=200, deadline=None)
@given(
    st.lists(st.tuples(st.booleans(), st.floats(-10, 1e4, allow_nan=False)), max_size=40),
    st.floats(0, 1e4),
)
def test_prop_lambda_positive_and_reliability_in_unit_interval(
    attempts: list[tuple[bool, float]], predicted: float
) -> None:
    rep = Reputation()
    for failed, duration in attempts:
        rep.record_attempt("w", failed, duration)
    lam = rep.failure_rate("w")
    assert math.isfinite(lam) and lam > 0
    assert 0.0 <= rep.reliability("w", predicted) <= 1.0
