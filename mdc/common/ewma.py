"""Exponentially weighted moving average (EWMA).

Used everywhere an estimate must follow a noisy, slowly changing quantity:
RTT, bandwidth, per-kind compute speed, failure rates.

    value_new = alpha * sample + (1 - alpha) * value_old

A large ``alpha`` reacts quickly but is noisy; a small ``alpha`` is smooth but
slow. The first sample initialises the average directly (unless an
``initial`` value was given), so a single early sample is not dragged towards
an arbitrary starting point.
"""

from __future__ import annotations

import math


class EWMA:
    """Exponentially weighted moving average of finite float samples.

    Args:
        alpha: smoothing factor in (0, 1].
        initial: optional starting estimate used until the first sample
            arrives (``count`` stays 0 until then).
    """

    def __init__(self, alpha: float, initial: float | None = None) -> None:
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be a number")
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha!r}")
        if initial is not None and not math.isfinite(initial):
            raise ValueError(f"initial must be finite, got {initial!r}")
        self.alpha = float(alpha)
        self._value: float | None = None if initial is None else float(initial)
        self._count = 0

    def update(self, x: float) -> float:
        """Add one sample and return the new average.

        Non-finite samples (NaN, ±inf) are rejected with ``ValueError``: one
        NaN would poison the average forever, and peers may send hostile
        values. Callers that handle untrusted numbers clamp them first.
        """
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise TypeError("sample must be a number")
        sample = float(x)
        if not math.isfinite(sample):
            raise ValueError(f"EWMA sample must be finite, got {x!r}")
        if self._count == 0 and self._value is None:
            self._value = sample
        else:
            previous = sample if self._value is None else self._value
            self._value = self.alpha * sample + (1.0 - self.alpha) * previous
        self._count += 1
        return self._value

    @property
    def value(self) -> float | None:
        """Current estimate (``None`` if no sample and no initial value)."""
        return self._value

    @property
    def count(self) -> int:
        """Number of samples added so far."""
        return self._count

    def value_or(self, default: float) -> float:
        """Current estimate, or ``default`` if there is none yet."""
        return default if self._value is None else self._value

    def __repr__(self) -> str:
        return f"EWMA(alpha={self.alpha}, value={self._value}, count={self._count})"
