"""Energy calibration from measured runs (spec Phase 12 item 6).

``python -m mdc.cli calibrate-energy --runs runs/<dir>`` reads every
``report.json`` below the directory and, per device class, fits::

    measured_J - idle_W * makespan_s  ≈  p_compute_W * compute_s + p_tx_W * transfer_s

by non-negative least squares (two unknowns: solved exactly by checking the
unconstrained solution and the two one-variable fits). Only workers with a
measured energy (battery current reported, not charging) are used. It prints
suggested ``energy.p_compute_w_<class>`` / ``energy.p_tx_w_<class>`` values and
changes nothing: you decide and record the values in DECISIONS.md.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MIN_SAMPLES = 2


@dataclass(frozen=True)
class Sample:
    """One worker in one run."""

    device_class: str
    compute_s: float
    transfer_s: float
    energy_j: float


@dataclass(frozen=True)
class Fit:
    """Suggested constants for one device class."""

    device_class: str
    samples: int
    p_compute_w: float
    p_tx_w: float
    rms_error_j: float


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def load_samples(runs_dir: Path) -> list[Sample]:
    """Samples from every readable ``report.json`` under ``runs_dir``."""
    samples = []
    for path in sorted(runs_dir.rglob("report.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        makespan = _finite(report.get("makespan_s")) or 0.0
        for worker in (report.get("per_worker") or {}).values():
            if not isinstance(worker, dict) or worker.get("trusted"):
                continue
            measured = _finite(worker.get("energy_meas_j"))
            compute = _finite(worker.get("compute_s"))
            transfer = _finite(worker.get("transfer_s"))
            if measured is None or compute is None or transfer is None:
                continue
            idle = _finite(worker.get("idle_baseline_w")) or 0.0
            samples.append(
                Sample(
                    str(worker.get("device_class", "phone")),
                    compute,
                    transfer,
                    max(0.0, measured - idle * makespan),
                )
            )
    return samples


def nnls2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative least squares for two columns (exact by enumeration)."""
    candidates = []
    solution, *_ = np.linalg.lstsq(x, y, rcond=None)
    if (solution >= 0).all():
        candidates.append(solution)
    for col in range(2):
        column = x[:, col]
        denom = float(column @ column)
        coef = max(0.0, float(column @ y) / denom) if denom > 0 else 0.0
        candidate = np.zeros(2)
        candidate[col] = coef
        candidates.append(candidate)
    return min(candidates, key=lambda c: float(np.sum((x @ c - y) ** 2)))


def fit(samples: list[Sample]) -> list[Fit]:
    """One fit per device class with at least :data:`MIN_SAMPLES` samples."""
    fits = []
    for device_class in sorted({s.device_class for s in samples}):
        rows = [s for s in samples if s.device_class == device_class]
        if len(rows) < MIN_SAMPLES:
            continue
        x = np.array([[s.compute_s, s.transfer_s] for s in rows])
        y = np.array([s.energy_j for s in rows])
        coef = nnls2(x, y)
        rms = float(np.sqrt(np.mean((x @ coef - y) ** 2)))
        fits.append(Fit(device_class, len(rows), float(coef[0]), float(coef[1]), rms))
    return fits


def report_text(fits: list[Fit], samples: int) -> str:
    """Human-readable suggestions."""
    if not fits:
        return (
            f"{samples} usable sample(s): need at least {MIN_SAMPLES} per device class with "
            "measured energy (battery current reported, device unplugged)."
        )
    lines = [f"Suggestions from {samples} sample(s) (nothing was changed):"]
    for f in fits:
        lines.append(
            f"  {f.device_class}: --set energy.p_compute_w_{f.device_class}={f.p_compute_w:.3f} "
            f"--set energy.p_tx_w_{f.device_class}={f.p_tx_w:.3f}   "
            f"({f.samples} samples, RMS error {f.rms_error_j:.2f} J)"
        )
    lines.append("Record the values you adopt in DECISIONS.md.")
    return "\n".join(lines)
