"""Rate limiting and auth-failure bans (spec S.6.3) — unit level with a fake clock.

The end-to-end ban against a live coordinator is in ``test_sec_resource_limits.py``.
"""

from __future__ import annotations

import pytest

from mdc.common.ratelimit import FailureTracker, SlidingCounter, TokenBucket

pytestmark = pytest.mark.security


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_token_bucket_burst_and_refill() -> None:
    clock = FakeClock()
    bucket = TokenBucket(rate=2.0, capacity=3, clock=clock)
    assert [bucket.allow() for _ in range(4)] == [True, True, True, False]
    clock.now = 0.5  # +1 token
    assert bucket.allow() and not bucket.allow()
    clock.now = 100.0
    assert bucket.tokens == 3.0  # never above capacity
    assert bucket.allow(3.0) and not bucket.allow(0.5)


def test_token_bucket_clock_going_backwards_is_harmless() -> None:
    clock = FakeClock()
    clock.now = 10.0
    bucket = TokenBucket(1.0, 1, clock=clock)
    bucket.allow()
    clock.now = 5.0
    assert not bucket.allow()


@pytest.mark.parametrize(("rate", "capacity"), [(0, 1), (-1, 1), (1, 0.5)])
def test_token_bucket_validation(rate: float, capacity: float) -> None:
    with pytest.raises(ValueError):
        TokenBucket(rate, capacity)


def test_token_bucket_rejects_non_positive_cost() -> None:
    with pytest.raises(ValueError):
        TokenBucket(1, 1).allow(0)


def test_ban_after_limit_within_window_and_recovery() -> None:
    clock = FakeClock()
    tracker = FailureTracker(limit=3, window_s=60, ban_s=300, clock=clock)
    assert tracker.record_failure("1.2.3.4") is False
    assert tracker.record_failure("1.2.3.4") is False
    assert not tracker.is_banned("1.2.3.4")
    assert tracker.record_failure("1.2.3.4") is True  # third failure starts the ban
    assert tracker.is_banned("1.2.3.4")
    assert tracker.ban_remaining_s("1.2.3.4") == 300
    assert not tracker.is_banned("5.6.7.8")  # per IP
    clock.now = 299.9
    assert tracker.is_banned("1.2.3.4")
    assert tracker.record_failure("1.2.3.4") is False  # failures during a ban do not extend it
    clock.now = 300.0
    assert not tracker.is_banned("1.2.3.4")  # recovered
    assert tracker.ban_remaining_s("1.2.3.4") == 0.0
    assert tracker.ban_remaining_s("never-seen") == 0.0


def test_failures_outside_window_are_forgotten() -> None:
    clock = FakeClock()
    tracker = FailureTracker(limit=3, window_s=10, ban_s=60, clock=clock)
    tracker.record_failure("ip")
    clock.now = 6
    tracker.record_failure("ip")
    clock.now = 12  # the first failure has left the window
    assert tracker.failures("ip") == 1
    assert tracker.record_failure("ip") is False
    assert tracker.failures("ip") == 2
    assert tracker.failures("unknown") == 0


def test_success_clears_history() -> None:
    tracker = FailureTracker(limit=2, window_s=60, ban_s=60, clock=FakeClock())
    tracker.record_failure("ip")
    tracker.record_success("ip")
    tracker.record_success("other")  # unknown key: no error
    assert tracker.record_failure("ip") is False


def test_zero_ban_disables_bans() -> None:
    tracker = FailureTracker(limit=1, window_s=60, ban_s=0, clock=FakeClock())
    assert tracker.record_failure("ip") is False
    assert not tracker.is_banned("ip")


def test_memory_is_bounded_lru() -> None:
    clock = FakeClock()
    tracker = FailureTracker(limit=2, window_s=60, ban_s=60, max_entries=1024, clock=clock)
    tracker.record_failure("keep")
    for i in range(5000):  # an attacker spraying from many addresses
        tracker.record_failure(f"10.0.{i // 256}.{i % 256}")
        if i % 100 == 0:
            tracker.record_failure("keep")  # recently used: stays remembered
    assert len(tracker) == 1024
    assert tracker.is_banned("keep")


@pytest.mark.parametrize(
    "kwargs",
    [{"limit": 0}, {"window_s": 0}, {"ban_s": -1}, {"max_entries": 0}],
)
def test_failure_tracker_validation(kwargs: dict) -> None:
    args = {"limit": 1, "window_s": 1, "ban_s": 1, **kwargs}
    with pytest.raises(ValueError):
        FailureTracker(**args)


def test_sliding_counter() -> None:
    clock = FakeClock()
    counter = SlidingCounter(60, clock=clock)
    assert [counter.add() for _ in range(3)] == [1, 2, 3]
    clock.now = 61
    assert counter.add() == 1
    with pytest.raises(ValueError):
        SlidingCounter(0)
