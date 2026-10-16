"""Application-level network shaping (spec Phase 3 item 6).

Laptop-only clusters run over loopback, where the network is effectively
infinite. To emulate a slow phone link, the worker sleeps before sending a
frame of ``n`` bytes::

    delay = net_delay_ms / 1000 + n * 8 / (net_bw_kbps * 1000)

Only bulk sends (TASK_RESULT, PROBE_ACK) are shaped — never PONG — so the
emulated bandwidth affects transfers without making a healthy worker look
dead to the failure detector. Consequence (documented for Phase 6): shaping
shows up in measured *bandwidth*, much less in ping RTT.

This only models a slow pipe; real radio behaviour (loss bursts, contention,
power-saving wake-up delays) is not modelled — see docs/EXPERIMENTS.md.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Shaper:
    """Delay/bandwidth emulation for bulk sends. ``None`` disables a component."""

    delay_ms: float | None = None
    bw_kbps: float | None = None

    def __post_init__(self) -> None:
        if self.delay_ms is not None and self.delay_ms < 0:
            raise ValueError("net delay must be >= 0 ms")
        if self.bw_kbps is not None and self.bw_kbps <= 0:
            raise ValueError("net bandwidth must be > 0 kbit/s")

    @property
    def enabled(self) -> bool:
        """True if any shaping is configured."""
        return self.delay_ms is not None or self.bw_kbps is not None

    def delay_s(self, n_bytes: int) -> float:
        """Seconds to wait before sending ``n_bytes``."""
        if n_bytes < 0:
            raise ValueError("byte count must be >= 0")
        delay = (self.delay_ms or 0.0) / 1000.0
        if self.bw_kbps is not None:
            delay += n_bytes * 8 / (self.bw_kbps * 1000.0)
        return delay
