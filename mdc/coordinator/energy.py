"""Energy model, budgets and ledger: do not drain the volunteers' phones (spec Phase 8).

**Model** per device class (``phone`` or ``laptop``; anything else uses the
laptop constants)::

    E_task(d) [J] = p_compute_w(class) * compute_s + p_tx_w(class) * transfer_s
    capacity_j(d) = battery_capacity_wh(class) * 3600
    budget_j(d)   = capacity_j * (battery_pct / 100) * (max_job_energy_pct / 100)

The constants in ``EnergyConfig`` are **rough placeholders**: real phones vary
a lot (SoC, screen, radio, temperature). Calibrate them from measured runs
(Phase 12, ``mdc.cli calibrate-energy``) and record the values in DECISIONS.md.

**Budget** is fixed when a worker is first seen during a job (job start for
workers already connected). A device whose level is unknown (desktop, no
battery) or that is charging has no budget limit: it is not spending battery.

**Ledger** per worker: sending an attempt adds its predicted energy to
``energy_committed_j``. When it finishes, the commitment is released and
``energy_spent_j`` grows by the energy computed from the *measured* times
(``compute_ms`` from the worker, clamped to the coordinator-measured total;
transfer = the rest). Failed attempts (timeout, lost, error) are charged
their predicted energy: the device most likely did the work, and charging
it keeps the budget conservative.

**Feasibility** (used by policies and enforced by the scheduler)::

    eligible(d, task)  iff  battery_eligible(d)          # >= min_battery_pct or charging
                       and  spent + committed + E_task <= budget
    the trusted local worker is exempt (it is the laptop running the coordinator)

**Measured energy**: when a PONG reports ``current_ma`` and ``voltage_v`` and
the device is not charging, ``|current_ma| / 1000 * voltage_v * dt`` is
integrated between consecutive samples (``dt`` on the coordinator's clock,
capped so a long gap never adds a huge guess). Samples before the job starts
build the **idle baseline** (average power while waiting for workers); it is
reported, not subtracted, so the analysis can decide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from mdc.common.config import EnergyConfig
from mdc.coordinator.reputation import battery_eligible

WH_TO_J = 3600.0
PERCENT = 100.0
MAX_CURRENT_MA = 20_000.0  # 20 A: anything above is a hostile or broken report
MAX_VOLTAGE_V = 30.0
MAX_SAMPLE_GAP_FACTOR = 3.0  # dt is capped at 3 x the expected sample interval
UNLIMITED = math.inf


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _finite_nonneg(value: Any) -> float | None:
    if not _is_number(value):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


@dataclass
class EnergyLedger:
    """Energy bookkeeping of one worker during one job."""

    budget_j: float = UNLIMITED
    committed_j: float = 0.0
    spent_j: float = 0.0
    measured_j: float = 0.0
    idle_j: float = 0.0
    idle_s: float = 0.0
    last_sample_t: float | None = None
    budget_set: bool = False
    exhausted_logged: bool = False
    battery_logged: bool = False

    @property
    def remaining_j(self) -> float:
        """Budget left for new work (``inf`` if unlimited)."""
        return self.budget_j - self.spent_j - self.committed_j

    @property
    def idle_baseline_w(self) -> float | None:
        """Average measured power before the job started (``None`` without samples)."""
        return self.idle_j / self.idle_s if self.idle_s > 0 else None


class EnergyModel:
    """The energy formulas plus one :class:`EnergyLedger` per worker."""

    def __init__(self, config: EnergyConfig | None = None, min_battery_pct: float = 20.0) -> None:
        self.cfg = config if config is not None else EnergyConfig()
        self.min_battery_pct = min_battery_pct
        self.ledgers: dict[str, EnergyLedger] = {}
        self.job_running = False

    # ------------------------------------------------------------- formulas

    def p_compute_w(self, device_class: str) -> float:
        """Compute power in watts (phone constants for phones, laptop ones otherwise)."""
        return (
            self.cfg.p_compute_w_phone if device_class == "phone" else self.cfg.p_compute_w_laptop
        )

    def p_tx_w(self, device_class: str) -> float:
        """Radio/transfer power in watts."""
        return self.cfg.p_tx_w_phone if device_class == "phone" else self.cfg.p_tx_w_laptop

    def capacity_j(self, device_class: str) -> float:
        """Nominal battery capacity in joules."""
        wh = (
            self.cfg.battery_capacity_wh_phone
            if device_class == "phone"
            else self.cfg.battery_capacity_wh_laptop
        )
        return wh * WH_TO_J

    def task_energy_j(self, device_class: str, compute_s: float, transfer_s: float) -> float:
        """``p_compute * compute_s + p_tx * transfer_s`` (negative times count as 0)."""
        compute_j = self.p_compute_w(device_class) * max(0.0, compute_s)
        return compute_j + self.p_tx_w(device_class) * max(0.0, transfer_s)

    def budget_j(self, device_class: str, battery: dict[str, Any]) -> float:
        """Energy this job may take from the device's battery (``inf``: no limit)."""
        if not self.cfg.enabled or battery.get("charging") is True:
            return UNLIMITED
        pct = _finite_nonneg(battery.get("pct"))
        if pct is None:
            return UNLIMITED
        pct = min(pct, PERCENT)
        return (
            self.capacity_j(device_class)
            * (pct / PERCENT)
            * (self.cfg.max_job_energy_pct / PERCENT)
        )

    # ---------------------------------------------------------------- ledger

    def ledger(self, worker_id: str) -> EnergyLedger:
        """The worker's ledger (created on first use)."""
        entry = self.ledgers.get(worker_id)
        if entry is None:
            entry = EnergyLedger()
            self.ledgers[worker_id] = entry
        return entry

    def start_job(self) -> None:
        """A job starts: fresh ledgers (budgets are fixed on first use), keep idle data."""
        self.job_running = True
        for entry in self.ledgers.values():
            entry.budget_j, entry.budget_set = UNLIMITED, False
            entry.committed_j = entry.spent_j = entry.measured_j = 0.0
            entry.exhausted_logged = entry.battery_logged = False

    def end_job(self) -> None:
        """The job finished: later samples no longer count as job energy."""
        self.job_running = False

    def open_budget(
        self, worker_id: str, device_class: str, battery: dict[str, Any], trusted: bool
    ) -> EnergyLedger:
        """Fix a worker's budget the first time it is considered during a job."""
        entry = self.ledger(worker_id)
        if not entry.budget_set:
            entry.budget_set = True
            entry.budget_j = UNLIMITED if trusted else self.budget_j(device_class, battery)
        return entry

    def commit(self, worker_id: str, predicted_j: float) -> None:
        """An attempt was sent: reserve its predicted energy."""
        self.ledger(worker_id).committed_j += max(0.0, predicted_j)

    def settle(self, worker_id: str, predicted_j: float, actual_j: float | None) -> None:
        """Release a commitment; charge ``actual_j`` (or the prediction if unknown)."""
        entry = self.ledger(worker_id)
        entry.committed_j = max(0.0, entry.committed_j - max(0.0, predicted_j))
        charge = predicted_j if actual_j is None else actual_j
        entry.spent_j += max(0.0, charge)

    # ----------------------------------------------------------- feasibility

    def feasible(
        self,
        worker_id: str,
        battery: dict[str, Any],
        trusted: bool,
        task_j: float,
    ) -> bool:
        """May ``worker_id`` take a task needing ``task_j`` joules?"""
        if trusted or not self.cfg.enabled:
            return True
        if not battery_eligible(battery, self.min_battery_pct):
            return False
        return self.ledger(worker_id).remaining_j >= task_j

    # -------------------------------------------------------------- measured

    def on_battery_sample(self, worker_id: str, battery: dict[str, Any], now: float) -> float:
        """Integrate one battery sample; return the joules added (0 if none)."""
        entry = self.ledger(worker_id)
        previous, entry.last_sample_t = entry.last_sample_t, now
        if previous is None or now <= previous:
            return 0.0
        raw_current = battery.get("current_ma")
        current = (
            _finite_nonneg(abs(raw_current))
            if isinstance(raw_current, (int, float)) and not isinstance(raw_current, bool)
            else None
        )  # discharging currents are reported negative on some devices
        voltage = _finite_nonneg(battery.get("voltage_v"))
        if battery.get("charging") is True or current is None or voltage is None:
            return 0.0
        current, voltage = min(current, MAX_CURRENT_MA), min(voltage, MAX_VOLTAGE_V)
        dt = min(now - previous, MAX_SAMPLE_GAP_FACTOR * self.cfg.sample_interval_s)
        joules = current / 1000.0 * voltage * dt
        if self.job_running:
            entry.measured_j += joules
        else:
            entry.idle_j += joules
            entry.idle_s += dt
        return joules
