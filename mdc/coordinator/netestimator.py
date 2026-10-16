"""Per-worker RTT and bandwidth estimation (spec Phase 6 item 1, new feature A).

**RTT** — every PONG gives a round-trip sample ``now - t_c`` measured on the
coordinator's own clock. The estimate is an EWMA (``netest.rtt_ewma_alpha``);
the minimum ever seen is kept too (the best case, least disturbed by load).
The handshake gives the first sample (CHALLENGE sent → HELLO received is one
round trip plus a few microseconds of HMAC), so a fresh worker has an RTT
before its first PONG.

**Bandwidth**, from two kinds of samples, smoothed by one EWMA
(``netest.bw_ewma_alpha``), ``netest.default_bw_bytes_per_s`` until the first:

* *active probe* at join: ``PROBE`` with ``probe_bytes`` random bytes; on
  ``PROBE_ACK``: ``probe_bytes / max(1 ms, elapsed - rtt_min)``. Limitation:
  this measures mainly the coordinator → worker direction (the ACK is tiny),
  and app-level shaping on the worker only delays its own sends — so for
  emulated slow links the probe overestimates; passive samples correct it.
* *passive* from every completed attempt::

      net_time      = total - (compute_ms + queue_ms) / 1000
      transfer_time = max(1 ms, net_time - rtt)
      sample        = (bytes_in + bytes_out) / transfer_time

  ignored when ``transfer_time < 5 ms`` or fewer than 16 KiB moved (too noisy
  to mean anything); clamped to ``[min_bw, 1e9]`` B/s.

**Compression ratios** — payloads are zlib-compressed, so transfer time
depends on *compressed* size. An EWMA of ``wire_bytes / raw_bytes`` is kept
per task kind and direction (starting at 1.0) and used to estimate the
bytes a future chunk will put on the wire.

All inputs come (directly or indirectly) from peers, so every sample is
checked: non-finite or negative values are dropped, results are clamped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from mdc.common.config import NetestConfig
from mdc.common.ewma import EWMA

MIN_TRANSFER_S = 1e-3
NOISY_TRANSFER_S = 5e-3
NOISY_BYTES = 16 * 1024
MAX_BW = 1e9
MAX_RTT_S = 600.0
COMPRESSION_ALPHA = 0.3


def _finite_non_negative(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    as_float = float(value)
    return as_float if math.isfinite(as_float) and as_float >= 0 else None


@dataclass
class LinkEstimate:
    """Network estimates of one worker."""

    rtt: EWMA
    bw: EWMA
    rtt_min: float | None = None
    probe_samples: int = 0
    passive_samples: int = 0
    ignored_samples: int = 0


@dataclass
class NetEstimator:
    """RTT, bandwidth and compression estimates for all workers."""

    config: NetestConfig
    links: dict[str, LinkEstimate] = field(default_factory=dict)
    compression: dict[tuple[str, str], EWMA] = field(default_factory=dict)

    def _link(self, worker_id: str) -> LinkEstimate:
        link = self.links.get(worker_id)
        if link is None:
            link = LinkEstimate(EWMA(self.config.rtt_ewma_alpha), EWMA(self.config.bw_ewma_alpha))
            self.links[worker_id] = link
        return link

    # ---------------------------------------------------------------- RTT

    def on_rtt(self, worker_id: str, rtt_s: object) -> bool:
        """Add an RTT sample (seconds). Returns False if the sample was rejected."""
        value = _finite_non_negative(rtt_s)
        link = self._link(worker_id)
        if value is None or value > MAX_RTT_S:
            link.ignored_samples += 1
            return False
        link.rtt.update(value)
        link.rtt_min = value if link.rtt_min is None else min(link.rtt_min, value)
        return True

    def rtt(self, worker_id: str) -> float:
        """Smoothed RTT in seconds (0 if never measured)."""
        link = self.links.get(worker_id)
        return link.rtt.value_or(0.0) if link is not None else 0.0

    def rtt_min(self, worker_id: str) -> float:
        """Smallest RTT seen (0 if never measured)."""
        link = self.links.get(worker_id)
        return link.rtt_min if link is not None and link.rtt_min is not None else 0.0

    # ----------------------------------------------------------- bandwidth

    def _clamp_bw(self, sample: float) -> float:
        return max(self.config.min_bw_bytes_per_s, min(MAX_BW, sample))

    def bw(self, worker_id: str) -> float:
        """Smoothed bandwidth in bytes/s (the configured default before any sample)."""
        link = self.links.get(worker_id)
        default = self.config.default_bw_bytes_per_s
        return link.bw.value_or(default) if link is not None else default

    def on_probe_ack(self, worker_id: str, probe_bytes: int, elapsed_s: object) -> float | None:
        """Active probe result; returns the bandwidth sample used (or None if rejected)."""
        elapsed = _finite_non_negative(elapsed_s)
        link = self._link(worker_id)
        if elapsed is None or probe_bytes <= 0 or elapsed > MAX_RTT_S:
            link.ignored_samples += 1
            return None
        sample = self._clamp_bw(
            probe_bytes / max(MIN_TRANSFER_S, elapsed - self.rtt_min(worker_id))
        )
        link.bw.update(sample)
        link.probe_samples += 1
        return sample

    def on_transfer(
        self,
        worker_id: str,
        total_s: object,
        compute_ms: object,
        queue_ms: object,
        bytes_in: int,
        bytes_out: int,
    ) -> float | None:
        """Passive sample from one completed attempt (spec formula); None if ignored."""
        total = _finite_non_negative(total_s)
        compute = _finite_non_negative(compute_ms)
        queue = _finite_non_negative(queue_ms)
        link = self._link(worker_id)
        moved = max(0, bytes_in) + max(0, bytes_out)
        if total is None or compute is None or queue is None:
            link.ignored_samples += 1
            return None
        net_time = total - (compute + queue) / 1000.0
        transfer_time = max(MIN_TRANSFER_S, net_time - self.rtt(worker_id))
        if transfer_time < NOISY_TRANSFER_S or moved < NOISY_BYTES:
            link.ignored_samples += 1
            return None
        sample = self._clamp_bw(moved / transfer_time)
        link.bw.update(sample)
        link.passive_samples += 1
        return sample

    def predict_transfer_s(self, worker_id: str, bytes_in: float, bytes_out: float) -> float:
        """``rtt + (bytes_in + bytes_out) / bw``."""
        return self.rtt(worker_id) + max(0.0, bytes_in + bytes_out) / self.bw(worker_id)

    # --------------------------------------------------------- compression

    def on_compression(self, kind: str, direction: str, raw_bytes: int, wire_bytes: int) -> None:
        """Observe ``wire / raw`` for ``kind`` and ``direction`` ("in" or "out")."""
        if raw_bytes <= 0 or wire_bytes < 0:
            return
        ratio = min(2.0, wire_bytes / raw_bytes)  # zlib can only expand slightly
        self.compression.setdefault((kind, direction), EWMA(COMPRESSION_ALPHA)).update(ratio)

    def compression_ratio(self, kind: str, direction: str) -> float:
        """Estimated wire/raw ratio (1.0 before any observation)."""
        estimate = self.compression.get((kind, direction))
        return estimate.value_or(1.0) if estimate is not None else 1.0

    def compression_table(self) -> dict[str, tuple[float, float]]:
        """``kind -> (ratio_in, ratio_out)`` for scheduling contexts."""
        kinds = {kind for kind, _ in self.compression}
        return {
            k: (self.compression_ratio(k, "in"), self.compression_ratio(k, "out")) for k in kinds
        }

    # ------------------------------------------------------------ snapshot

    def snapshot(self, worker_id: str) -> dict[str, Any]:
        """Estimates of one worker (dashboard, snapshot.json, report)."""
        link = self.links.get(worker_id)
        return {
            "rtt_s": self.rtt(worker_id),
            "rtt_min_s": self.rtt_min(worker_id),
            "bw_bytes_per_s": self.bw(worker_id),
            "probe_samples": link.probe_samples if link else 0,
            "passive_samples": link.passive_samples if link else 0,
            "ignored_samples": link.ignored_samples if link else 0,
        }
