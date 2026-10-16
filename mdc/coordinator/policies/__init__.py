"""Scheduling policy registry.

``make_policy(config)`` returns a fresh policy instance for
``config.scheduler.policy``: ``local`` and ``round_robin`` (baselines),
``fastest`` (pulled forward to Phase 6), ``most_reliable`` and ``tesar``
(Phase 9). ``milp_offline`` is an offline analysis tool, not a live policy.
"""

from __future__ import annotations

from mdc.common.config import Config
from mdc.coordinator.policies.base import Policy, PolicyFactory
from mdc.coordinator.policies.fastest import FastestPolicy
from mdc.coordinator.policies.local_only import LocalOnlyPolicy
from mdc.coordinator.policies.most_reliable import MostReliablePolicy
from mdc.coordinator.policies.round_robin import RoundRobinPolicy
from mdc.coordinator.policies.tesar import TesarPolicy

POLICIES: dict[str, PolicyFactory] = {
    "fastest": FastestPolicy,
    "local": LocalOnlyPolicy,
    "most_reliable": MostReliablePolicy,
    "round_robin": RoundRobinPolicy,
    "tesar": TesarPolicy,
}


class UnknownPolicyError(ValueError):
    """The configured policy does not exist."""


def make_policy(config: Config) -> Policy:
    """Instantiate the configured policy."""
    name = config.scheduler.policy
    factory = POLICIES.get(name)
    if factory is None:
        raise UnknownPolicyError(
            f"scheduling policy {name!r} is not available; choose one of {sorted(POLICIES)}"
        )
    return factory(config)
