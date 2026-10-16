"""Reputation and failure rate: how reliable is each worker? (spec Phase 7).

**Beta reputation** — per ``worker_id`` (the enrolled identity, so history
survives reconnects within one coordinator run)::

    s, f start at prior_success, prior_failure          (1, 1 -> rep 0.5)
    on every event:   s *= decay ; f *= decay            (old behaviour fades)
    success:          s += 1
    failure of kind k: f += weight_k                     (timeout, disconnect,
                                                          error, wrong_result)
    rep = s / (s + f)                                    (expected P(success))

``s`` and ``f`` are the parameters of a Beta distribution over the worker's
success probability; ``rep`` is its mean. A wrong result weighs 5x a timeout
by default: lying is worse than being slow.

**Failure rate** λ (failures per second of assigned work). Per finished
attempt we update two EWMAs (``failure_rate_ewma_alpha``): ``p`` = fraction of
attempts that failed and ``T`` = predicted duration of an attempt. If failures
arrive as a Poisson process with rate λ, an attempt of length ``T`` fails with
probability ``1 - exp(-λT)``, hence::

    λ = -ln(1 - p) / T          (p capped at 0.999, T at least 50 ms)

with ``default_failure_rate_per_s`` until a worker has finished 3 attempts.
(The spec suggests ``failures / max(1, busy_time)``; DECISIONS.md explains why
this form fits sub-second tasks and the ``exp(-λt)`` model used below.)

**Task reliability** (used by policies)::

    R(task, d) = rep(d) * exp(-λ(d) * predicted_s(task, d))
    R = 0 if d's battery is below the floor and not charging (ineligible)
    R = 0.9999 for the trusted local worker
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from mdc.common.config import ReputationConfig
from mdc.common.ewma import EWMA

FAILURE_KINDS = frozenset({"timeout", "disconnect", "error", "wrong_result"})
TRUSTED_RELIABILITY = 0.9999
MIN_ATTEMPTS_FOR_RATE = 3
MAX_FAILURE_FRACTION = 0.999
MIN_ATTEMPT_S = 0.05
MIN_RATE = 1e-6


def task_reliability(
    rep: float, failure_rate: float, predicted_s: float, trusted: bool, eligible: bool
) -> float:
    """``R = rep * exp(-λ * predicted_s)``; 0.9999 if trusted, 0 if ineligible."""
    if trusted:
        return TRUSTED_RELIABILITY
    if not eligible:
        return 0.0
    return rep * math.exp(-max(0.0, failure_rate) * max(0.0, predicted_s))


def battery_eligible(battery: dict[str, Any], min_pct: float) -> bool:
    """A worker may get work unless it reports a battery below ``min_pct`` while
    not charging. An unknown level (desktop, no sysfs) counts as eligible."""
    if battery.get("charging") is True:
        return True
    pct = battery.get("pct")
    return pct is None or float(pct) >= min_pct


@dataclass
class _Record:
    s: float
    f: float
    successes: int = 0
    failures: dict[str, int] = field(default_factory=lambda: dict.fromkeys(FAILURE_KINDS, 0))
    attempts: int = 0
    fail_fraction: EWMA | None = None
    attempt_s: EWMA | None = None


class Reputation:
    """Beta reputation and failure-rate estimates for every worker."""

    def __init__(self, config: ReputationConfig | None = None) -> None:
        self.cfg = config if config is not None else ReputationConfig()
        self._records: dict[str, _Record] = {}

    def _entry(self, worker_id: str) -> _Record:
        record = self._records.get(worker_id)
        if record is None:
            record = _Record(self.cfg.prior_success, self.cfg.prior_failure)
            self._records[worker_id] = record
        return record

    def _decay(self, record: _Record) -> None:
        record.s *= self.cfg.decay
        record.f *= self.cfg.decay

    # -------------------------------------------------------------- events

    def record_success(self, worker_id: str) -> None:
        """An accepted (possibly not yet verified) result."""
        record = self._entry(worker_id)
        self._decay(record)
        record.s += 1.0
        record.successes += 1

    def record_failure(self, worker_id: str, kind: str) -> None:
        """A failure of ``kind`` (see :data:`FAILURE_KINDS`)."""
        if kind not in FAILURE_KINDS:
            raise ValueError(f"unknown failure kind {kind!r}")
        record = self._entry(worker_id)
        self._decay(record)
        record.f += float(getattr(self.cfg, f"weight_{kind}"))
        record.failures[kind] += 1

    def record_attempt(self, worker_id: str, failed: bool, predicted_s: float) -> None:
        """One finished attempt, for the failure-rate estimate."""
        record = self._entry(worker_id)
        alpha = self.cfg.failure_rate_ewma_alpha
        if record.fail_fraction is None or record.attempt_s is None:
            record.fail_fraction, record.attempt_s = EWMA(alpha), EWMA(alpha)
        duration = predicted_s if math.isfinite(predicted_s) and predicted_s > 0 else 0.0
        record.fail_fraction.update(1.0 if failed else 0.0)
        record.attempt_s.update(duration)
        record.attempts += 1

    # ------------------------------------------------------------- queries

    def rep(self, worker_id: str) -> float:
        """Expected probability of success, always strictly between 0 and 1."""
        record = self._records.get(worker_id)
        if record is None:
            return self.cfg.prior_success / (self.cfg.prior_success + self.cfg.prior_failure)
        return record.s / (record.s + record.f)

    def failure_rate(self, worker_id: str) -> float:
        """λ in failures per second of assigned work."""
        record = self._records.get(worker_id)
        if (
            record is None
            or record.attempts < MIN_ATTEMPTS_FOR_RATE
            or record.fail_fraction is None
            or record.attempt_s is None
        ):
            return self.cfg.default_failure_rate_per_s
        p = min(MAX_FAILURE_FRACTION, record.fail_fraction.value_or(0.0))
        duration = max(MIN_ATTEMPT_S, record.attempt_s.value_or(MIN_ATTEMPT_S))
        return max(MIN_RATE, -math.log1p(-p) / duration)

    def reliability(
        self, worker_id: str, predicted_s: float, trusted: bool = False, eligible: bool = True
    ) -> float:
        """``R(task, d) = rep * exp(-λ * predicted_s)`` with the special cases above."""
        return task_reliability(
            self.rep(worker_id), self.failure_rate(worker_id), predicted_s, trusted, eligible
        )

    def snapshot(self, worker_id: str) -> dict[str, Any]:
        """``rep``, ``lambda``, ``s``, ``f`` and counts by type."""
        record = self._records.get(worker_id)
        s = record.s if record else self.cfg.prior_success
        f = record.f if record else self.cfg.prior_failure
        return {
            "rep": self.rep(worker_id),
            "lambda": self.failure_rate(worker_id),
            "s": s,
            "f": f,
            "successes": record.successes if record else 0,
            "failures": dict(record.failures) if record else dict.fromkeys(FAILURE_KINDS, 0),
            "attempts": record.attempts if record else 0,
        }
