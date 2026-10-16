"""The coordinator's live models of its workers, shared by every component.

One :class:`Estimators` exists per coordinator run. The dispatcher feeds it
(results, failures, transfers, energy), the scheduler and policies read it,
and the report summarizes it.
"""

from __future__ import annotations

from dataclasses import dataclass

from mdc.common.config import Config
from mdc.coordinator.energy import EnergyModel
from mdc.coordinator.netestimator import NetEstimator
from mdc.coordinator.reputation import Reputation


@dataclass
class Estimators:
    """Network (Phase 6), reputation (Phase 7) and energy (Phase 8) models."""

    netest: NetEstimator
    reputation: Reputation
    energy: EnergyModel

    @classmethod
    def from_config(cls, config: Config) -> Estimators:
        """Fresh models with the configured parameters."""
        return cls(
            NetEstimator(config.netest),
            Reputation(config.reputation),
            EnergyModel(config.energy, config.workers.min_battery_pct),
        )
