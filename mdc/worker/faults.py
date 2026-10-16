"""Lab-mode fault injection (spec Phase 3 fault flags, S.7.5).

Faults make a laptop-only cluster behave like real phones misbehaving, so the
coordinator's failure handling can be tested and measured:

=====================  ==========================================================
``--drop-prob P``      accept a task but never answer (attempt times out)
``--crash-prob P``     exit the process immediately on receiving a task
``--crash-after-s S``  exit S seconds after start
``--malicious-prob P`` return a *plausible* corrupted result (Phase 10 must catch it)
``--collude-key K``    corrupt deterministically per input, so workers sharing K
                       return *identical* wrong results (a colluding group)
``--slowdown F``       after computing, sleep (F - 1) * compute time; the reported
                       benchmark is F times slower too
``--flaky-mean-s M``   disconnect after Exp(M) seconds, reconnect after U(2, 10) s
=====================  ==========================================================

(``--net-delay-ms`` / ``--net-bw-kbps`` live in ``shaper.py``.)

Every fault flag requires ``--lab-mode``; a worker never misbehaves by accident.
Randomness comes from a seeded ``random.Random`` named ``rng`` so experiments
are reproducible — never from it for anything security-related.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

MIN_CORRUPT_FRACTION = 0.05
MAX_CORRUPT_FRACTION = 0.20
MIN_CORRUPT_DELTA = 20
MAX_CORRUPT_DELTA = 80
FLAKY_RECONNECT_S = (2.0, 10.0)


@dataclass
class FaultSettings:
    """Current fault configuration (CONTROL messages may change it live)."""

    drop_prob: float = 0.0
    crash_prob: float = 0.0
    crash_after_s: float | None = None
    malicious_prob: float = 0.0
    slowdown: float = 1.0
    flaky_mean_s: float | None = None
    collude_key: str | None = None

    def any_enabled(self) -> bool:
        """True if at least one fault is active."""
        return (
            self.drop_prob > 0
            or self.crash_prob > 0
            or self.crash_after_s is not None
            or self.malicious_prob > 0
            or self.slowdown != 1.0
            or self.flaky_mean_s is not None
            or self.collude_key is not None
        )

    def validate(self) -> None:
        """Raise ``ValueError`` for out-of-range settings."""
        for name in ("drop_prob", "crash_prob", "malicious_prob"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.slowdown < 1.0:
            raise ValueError("slowdown must be >= 1")
        if self.crash_after_s is not None and self.crash_after_s < 0:
            raise ValueError("crash_after_s must be >= 0")
        if self.flaky_mean_s is not None and self.flaky_mean_s <= 0:
            raise ValueError("flaky_mean_s must be > 0")


class FaultInjector:
    """Decides, per task, which fault (if any) happens. Seeded and reproducible."""

    def __init__(self, settings: FaultSettings, seed: int | None = None) -> None:
        settings.validate()
        self.settings = settings
        self.rng = random.Random(seed)  # noqa: S311 - experiment randomness, not security

    def should_crash(self) -> bool:
        """Crash on receiving this task?"""
        return self.settings.crash_prob > 0 and self.rng.random() < self.settings.crash_prob

    def should_drop(self) -> bool:
        """Silently drop this task?"""
        return self.settings.drop_prob > 0 and self.rng.random() < self.settings.drop_prob

    def should_corrupt(self) -> bool:
        """Return a wrong result for this task?"""
        p = self.settings.malicious_prob
        return p > 0 and self.rng.random() < p

    def corruption_rng(self, task_input: bytes) -> random.Random:
        """Randomness for one corruption: the injector's own, or — for a colluding
        group — one derived from the key and the task input, so every member makes
        the same change to the same input."""
        key = self.settings.collude_key
        if key is None:
            return self.rng
        digest = hashlib.sha256(task_input).hexdigest()
        return random.Random(f"{key}:{digest}")  # noqa: S311 - lab fault, not security

    def slowdown_sleep_s(self, compute_s: float) -> float:
        """Extra sleep after computing: ``(F - 1) * compute_s``."""
        return max(0.0, (self.settings.slowdown - 1.0) * compute_s)

    def flaky_session_s(self) -> float | None:
        """Seconds until the next simulated disconnect (exponential), or ``None``."""
        mean = self.settings.flaky_mean_s
        return None if mean is None else self.rng.expovariate(1.0 / mean)

    def flaky_reconnect_s(self) -> float:
        """Seconds to stay away after a simulated disconnect (uniform 2-10 s)."""
        return self.rng.uniform(*FLAKY_RECONNECT_S)

    def apply_control(self, action: str, value: float | None) -> None:
        """Apply a CONTROL ``set_*`` action (already validated and lab-mode gated)."""
        if action == "set_malicious" and value is not None:
            self.settings.malicious_prob = value
        elif action == "set_drop" and value is not None:
            self.settings.drop_prob = value
        elif action == "set_slowdown" and value is not None:
            self.settings.slowdown = value
        self.settings.validate()


def corrupt_array(array: npt.NDArray[np.uint8], rng: random.Random) -> npt.NDArray[np.uint8]:
    """A *plausible* wrong image: shift a random 5-20 % rectangle by +20..80 (mod 256).

    Same shape and dtype, mostly-correct pixels — the kind of error only
    re-execution can catch (not a crash, not garbage). The caller computes the
    payload hash *after* corruption, so transport integrity still passes.
    """
    out = np.array(array, dtype=np.uint8, copy=True)
    height, width = out.shape
    area = height * width
    target = rng.uniform(MIN_CORRUPT_FRACTION, MAX_CORRUPT_FRACTION) * area
    aspect = rng.uniform(0.5, 2.0)
    rect_h = max(1, min(height, round((target * aspect) ** 0.5)))
    rect_w = max(1, min(width, round(target / rect_h)))
    y0 = rng.randrange(0, height - rect_h + 1)
    x0 = rng.randrange(0, width - rect_w + 1)
    delta = rng.randint(MIN_CORRUPT_DELTA, MAX_CORRUPT_DELTA)
    region = out[y0 : y0 + rect_h, x0 : x0 + rect_w].astype(np.int32)
    out[y0 : y0 + rect_h, x0 : x0 + rect_w] = ((region + delta) % 256).astype(np.uint8)
    return out


def corrupt_json(kind: str, result: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """A plausible wrong JSON result that still passes the schema validators.

    * ``row_stats``: ``edge_pixels`` changed by 1-5 % (at least 1), kept in [0, pixels].
    * ``synthetic``: one hex digit of the digest changed.
    """
    out = dict(result)
    if kind == "row_stats":
        pixels = int(out["pixels"])
        edge = int(out["edge_pixels"])
        change = max(1, round(edge * rng.uniform(0.01, 0.05)))
        candidate = edge + change if rng.random() < 0.5 else edge - change  # noqa: PLR2004
        if not 0 <= candidate <= pixels:
            candidate = edge - change if candidate > pixels else edge + change
        out["edge_pixels"] = max(0, min(pixels, candidate))
        if out["edge_pixels"] == edge and pixels > 0:  # 1-pixel strips: flip 0 <-> 1
            out["edge_pixels"] = 1 if edge == 0 else edge - 1
    elif kind == "synthetic":
        digest = str(out["digest"])
        pos = rng.randrange(len(digest))
        new_digit = format((int(digest[pos], 16) + rng.randint(1, 15)) % 16, "x")
        out["digest"] = digest[:pos] + new_digit + digest[pos + 1 :]
    return out
