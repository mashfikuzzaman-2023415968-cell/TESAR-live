"""Run summary metrics: ``report.json`` and one ``results.csv`` row (spec 11.3).

Fields that later phases fill in (reputation, energy, verification, replicas)
are present from the start with neutral values, so the report schema is
stable for the experiment scripts.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

from mdc.common.config import Config
from mdc.coordinator.dispatcher import TrafficCounters
from mdc.coordinator.energy import EnergyModel
from mdc.coordinator.estimators import Estimators
from mdc.coordinator.job import Job
from mdc.coordinator.registry import Registry
from mdc.tasks.base import compute_seconds


def local_estimate_s(job: Job, local_ms_per_unit: dict[str, float], fallback: float) -> float:
    """Predicted time to run every schedulable task on the local device, one at a time."""
    return sum(
        compute_seconds(t.work_units, local_ms_per_unit.get(t.kind, fallback))
        for t in job.tasks.values()
        if t.stage < 3  # noqa: PLR2004 - stage 3 always runs on the coordinator
    )


def wasted_work_s(job: Job) -> float:
    """Compute time of attempts whose results were not used (excluding verification)."""
    total_ms = 0.0
    for attempt in job.attempts.values():
        if attempt.purpose == "verify" or attempt.status in {"ok", "late_accepted"}:
            continue
        if attempt.compute_ms is not None:
            total_ms += attempt.compute_ms
    return total_ms / 1000.0


@dataclass(frozen=True)
class RunTiming:
    """Measured makespan and the local-only estimate (for the speedup)."""

    makespan_s: float
    local_estimate_s: float


def busy_seconds(job: Job) -> dict[str, tuple[float, float]]:
    """Per worker: (compute seconds, transfer seconds) of every answered attempt.

    Compute is the worker's clamped ``compute_ms``; transfer is the rest of the
    coordinator-measured round trip minus queueing.
    """
    out: dict[str, tuple[float, float]] = {}
    for a in job.attempts.values():
        if a.status not in {"ok", "late_accepted"} or a.compute_ms is None or a.sent_t is None:
            continue
        compute = a.compute_ms / 1000.0
        total = (a.result_t or a.sent_t) - a.sent_t
        transfer = max(0.0, total - compute - (a.queue_ms or 0.0) / 1000.0)
        c, t = out.get(a.worker_id, (0.0, 0.0))
        out[a.worker_id] = (c + compute, t + transfer)
    return out


def _budget(energy: EnergyModel | None, worker_id: str) -> float | None:
    """The job energy budget in J (``None``: unlimited or unknown)."""
    if energy is None or worker_id not in energy.ledgers:
        return None
    budget = energy.ledgers[worker_id].budget_j
    return round(budget, 3) if math.isfinite(budget) else None


def _idle_w(energy: EnergyModel | None, worker_id: str) -> float | None:
    """Average measured power while waiting for the job (for baseline subtraction)."""
    if energy is None or worker_id not in energy.ledgers:
        return None
    idle = energy.ledgers[worker_id].idle_baseline_w
    return round(idle, 4) if idle is not None else None


def build_report(
    job: Job,
    cfg: Config,
    registry: Registry,
    traffic: TrafficCounters,
    timing: RunTiming,
    outcome: dict[str, Any],
    seed: int,
    estimators: Estimators | None = None,
) -> dict[str, Any]:
    """All metrics of one run (keys per spec 11.3)."""
    verification: dict[str, Any] = outcome.get("verification") or {}
    netest = estimators.netest if estimators is not None else None
    reputation = estimators.reputation if estimators is not None else None
    energy = estimators.energy if estimators is not None else None
    makespan_s, local_estimate = timing.makespan_s, timing.local_estimate_s
    # Work attempts only; verification attempts are counted in the verification fields.
    sent = [a for a in job.attempts.values() if a.purpose in {"primary", "replica"}]
    # Count from each attempt's *final* state: an attempt can time out and later have
    # its late result accepted, so counting status changes as they happen would
    # double-count it (see DECISIONS.md, Phase 5).
    statuses = Counter(a.status for a in sent)
    chunk_sizes = [len(a.task_ids) for a in sent if a.purpose in {"primary", "replica"}]
    busy = busy_seconds(job)
    per_worker: dict[str, Any] = {}
    for record in registry:
        battery_end = record.battery.get("pct") if record.battery else None
        per_worker[record.name] = {
            "worker_id": record.worker_id,
            "trusted": record.trusted,
            "device_class": record.device_class,
            "tasks_ok": record.tiles_done,  # accepted task results (tiles and row stats)
            "attempts_ok": record.completed,
            "failed": record.failed,
            "wrong": record.wrong,
            "strikes": record.strikes,
            "verified_ok": record.verified_ok,
            "state": record.state.value,
            "rep_final": (
                round(reputation.rep(record.worker_id), 4) if reputation is not None else None
            ),
            "lambda_final": (
                round(reputation.failure_rate(record.worker_id), 6)
                if reputation is not None
                else None
            ),
            "energy_pred_j": round(record.energy_spent_j, 4) if record.energy_spent_j else None,
            "energy_meas_j": (
                round(record.energy_measured_j, 4) if record.energy_measured_j else None
            ),
            "energy_budget_j": _budget(energy, record.worker_id),
            "idle_baseline_w": _idle_w(energy, record.worker_id),
            # Measured time split, for energy calibration (mdc.cli calibrate-energy).
            "compute_s": round(busy.get(record.worker_id, (0.0, 0.0))[0], 4),
            "transfer_s": round(busy.get(record.worker_id, (0.0, 0.0))[1], 4),
            "battery_start": record.battery_start,
            "battery_end": battery_end,
            "charging": record.battery.get("charging") if record.battery else None,
            "avg_rtt_ms": (
                round(netest.rtt(record.worker_id) * 1000, 3) if netest is not None else None
            ),
            "bw_kbps": (
                round(netest.bw(record.worker_id) * 8 / 1000, 1) if netest is not None else None
            ),
            "ms_per_unit": {k: e.value for k, e in record.speed.items()},
        }
    return {
        "job_id": job.job_id,
        "job_type": job.job_type,
        "policy": cfg.scheduler.policy,
        "alpha": cfg.scheduler.alpha,
        "kappa": cfg.scheduler.kappa,
        "adaptive_chunking": cfg.chunking.adaptive,
        "verification_enabled": cfg.verification.enabled,
        "replication": cfg.scheduler.replication,
        "reliability_target": cfg.scheduler.reliability_target,
        "max_replicas": cfg.scheduler.max_replicas,
        "allow_local": cfg.scheduler.allow_local,
        "seed": seed,
        "image": job.image_path,
        "blur_iterations": job.params.get("blur_iterations"),
        "n_tiles": sum(1 for t in job.tasks.values() if t.stage == 1),
        "n_workers_remote": sum(1 for r in registry if not r.trusted),
        "makespan_s": round(makespan_s, 4),
        "local_estimate_s": round(local_estimate, 4),
        "speedup": round(local_estimate / makespan_s, 4) if makespan_s > 0 else None,
        "attempts_total": len(sent),
        "attempts_ok": statuses["ok"] + statuses["late_accepted"],  # results that were used
        "attempts_timeout": sum(1 for a in sent if a.timed_out),
        "attempts_lost": statuses["lost"],
        "attempts_error": statuses["error"] + statuses["rejected"],
        "attempts_late": statuses["late"] + statuses["late_accepted"],
        "attempts_cancelled": statuses["cancelled"],
        "wasted_work_s": round(wasted_work_s(job), 4),
        "replicas": sum(1 for a in sent if a.purpose == "replica"),
        "verifications": verification.get(
            "verifications", sum(1 for a in sent if a.purpose in {"verify", "tiebreak"})
        ),
        "verification_mismatches": verification.get("mismatches", 0),
        "verification_overhead_pct": verification.get("overhead_pct", 0.0),
        "detection_latency_s": verification.get("detection_latency_s", []),
        "quarantined": verification.get("quarantined", []),
        "repaired_tasks": verification.get("repaired_tasks", 0),
        "tiebreaks": verification.get("tiebreaks", 0),
        "free_replica_matches": verification.get("free_replica_matches", 0),
        "wrong_results_reached_final": outcome.get("wrong_results_reached_final"),
        "output_correct": outcome.get("output_correct"),
        "differing_pixels": outcome.get("differing_pixels"),
        "messages_sent": traffic.messages_sent,
        "bytes_sent": traffic.bytes_sent,
        "bytes_received": traffic.bytes_received,
        "avg_chunk_tiles": (round(sum(chunk_sizes) / len(chunk_sizes), 3) if chunk_sizes else None),
        "per_worker": per_worker,
    }
