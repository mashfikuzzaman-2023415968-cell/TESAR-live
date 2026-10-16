#!/usr/bin/env python3
"""Figures with mean +/- 95 % confidence intervals from an experiment (spec Phase 13).

Usage: ``python experiments/plots.py runs/<experiment>``

Reads ``results.csv`` (one row per run, written by ``run_matrix.py``) and writes
PNG + PDF figures to ``runs/<experiment>/figures/`` plus ``summary.csv`` (mean,
95 % CI half-width and n per group). Confidence intervals use the t
distribution: ``t_{0.975, n-1} * s / sqrt(n)``; with one run the half-width is
undefined (shown without an error bar). Failed runs (no report) are excluded
from the metric means and listed in the summary.

Figures by experiment (name prefix):

* e1 — ``e1_makespan_by_policy``, ``e1_failures_by_policy`` (stacked timeout/lost/error)
* e2 — ``e2_tradeoff_curve`` (alpha vs makespan and wasted work), ``e2_pareto``
  (speedup vs reliability proxy 1 - failed/attempts, labelled by alpha)
* e3 — ``e3_failure_sweep`` (makespan and wasted work vs drop probability, per policy)
* e4 — ``e4_chunking`` (makespan and messages, adaptive vs fixed, per policy)
* e5 — ``e5_malicious`` (detection latency, correct-output rate, overhead vs base_prob)
* e7 — ``e7_milp_gap`` (box plot of greedy-vs-MILP objective gap per alpha), ``e7_gaps.csv``
* anything else — ``makespan_by_point``
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
Row = dict[str, Any]
CONFIDENCE = 0.95
ALPHA = "\N{GREEK SMALL LETTER ALPHA}"


# ------------------------------------------------------------------ data


def convert(value: str) -> Any:
    """CSV text -> bool / int / float / None / str."""
    if value in {"", "None", "null"}:
        return None
    if value in {"True", "true"}:
        return True
    if value in {"False", "false"}:
        return False
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() and "." not in value and "e" not in value else number


def load_results(exp_dir: Path) -> list[Row]:
    """All rows of ``results.csv`` with typed values."""
    with (exp_dir / "results.csv").open(newline="", encoding="utf-8") as fh:
        return [{k: convert(v) for k, v in row.items()} for row in csv.DictReader(fh)]


def mean_ci(
    values: Iterable[float | None], confidence: float = CONFIDENCE
) -> tuple[float, float, int]:
    """(mean, CI half-width, n) of the finite values; half-width is NaN for n < 2."""
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(xs)
    if n == 0:
        return math.nan, math.nan, 0
    mean = sum(xs) / n
    if n < 2:  # noqa: PLR2004
        return mean, math.nan, n
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))
    t = float(stats.t.ppf((1 + confidence) / 2, n - 1))
    return mean, t * sd / math.sqrt(n), n


def ok_rows(rows: list[Row]) -> list[Row]:
    return [r for r in rows if r.get("status") != "failed"]


def group(rows: list[Row], key: str) -> dict[Any, list[Row]]:
    out: dict[Any, list[Row]] = defaultdict(list)
    for row in rows:
        if row.get(key) is not None:  # e.g. local-baseline rows have no matrix value
            out[row.get(key)].append(row)
    return dict(sorted(out.items(), key=lambda kv: (str(type(kv[0])), kv[0])))


def failed_attempts(row: Row) -> float:
    return float(
        sum(row.get(k) or 0 for k in ("attempts_timeout", "attempts_lost", "attempts_error"))
    )


def reliability_proxy(row: Row) -> float | None:
    total = row.get("attempts_total") or 0
    return None if not total else 1.0 - failed_attempts(row) / float(total)


def detection_latency(row: Row) -> float | None:
    raw = row.get("detection_latency_s")
    try:
        items = json.loads(raw) if isinstance(raw, str) else (raw or [])
    except ValueError:
        return None
    lat = [float(d["latency_s"]) for d in items if isinstance(d, dict) and "latency_s" in d]
    return sum(lat) / len(lat) if lat else None


def correct(row: Row) -> float:
    return 1.0 if row.get("output_correct") is True else 0.0


def measured_speedup(rows: list[Row]) -> Callable[[Row], float | None]:
    """Speedup against the measured local baseline of the same repetition, else the estimate."""
    local = {
        r["rep"]: r["makespan_s"]
        for r in rows
        if r.get("point_id") == "local_baseline" and r.get("makespan_s")
    }

    def speedup(row: Row) -> float | None:
        base = local.get(row.get("rep"))
        if base and row.get("makespan_s"):
            return float(base) / float(row["makespan_s"])
        return row.get("speedup")

    return speedup


# ---------------------------------------------------------------- drawing


def save(fig: Any, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.png", out_dir / f"{name}.pdf"]
    fig.tight_layout()
    for path in paths:
        fig.savefig(path, dpi=150)
    plt.close(fig)
    return paths


def _err(half: float) -> float:
    return 0.0 if math.isnan(half) else half


def bar(
    ax: Any, labels: list[str], groups: list[list[Row]], metric: Callable[[Row], Any], ylabel: str
) -> None:
    stats_ = [mean_ci(metric(r) for r in g) for g in groups]
    ax.bar(
        labels,
        [s[0] for s in stats_],
        yerr=[_err(s[1]) for s in stats_],
        capsize=4,
        color="#4c78a8",
    )
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)


def line(
    ax: Any,
    xs: list[float],
    groups: list[list[Row]],
    metric: Callable[[Row], Any],
    label: str,
    **kw: Any,
) -> None:
    stats_ = [mean_ci(metric(r) for r in g) for g in groups]
    ax.errorbar(
        xs,
        [s[0] for s in stats_],
        yerr=[_err(s[1]) for s in stats_],
        capsize=4,
        marker="o",
        label=label,
        **kw,
    )


def makespan(row: Row) -> Any:
    return row.get("makespan_s")


def wasted(row: Row) -> Any:
    return row.get("wasted_work_s")


# -------------------------------------------------------------- figures


def fig_e1(rows: list[Row], out: Path) -> list[Path]:
    by = group(rows, "policy")
    labels, groups = [str(k) for k in by], list(by.values())
    fig, ax = plt.subplots(figsize=(6, 4))
    bar(ax, labels, groups, makespan, "makespan (s)")
    ax.set_title("E1: makespan by policy (mean ± 95% CI)")
    paths = save(fig, out, "e1_makespan_by_policy")
    fig, ax = plt.subplots(figsize=(6, 4))
    bottom = [0.0] * len(groups)
    for key, color in (
        ("attempts_timeout", "#f58518"),
        ("attempts_lost", "#e45756"),
        ("attempts_error", "#72b7b2"),
    ):
        means = [mean_ci(r.get(key) for r in g)[0] for g in groups]
        means = [0.0 if math.isnan(m) else m for m in means]
        ax.bar(labels, means, bottom=bottom, label=key.removeprefix("attempts_"), color=color)
        bottom = [b + m for b, m in zip(bottom, means, strict=True)]
    ax.set_ylabel("failed attempts per run (mean)")
    ax.set_title("E1: failures by policy")
    ax.legend()
    ax.tick_params(axis="x", rotation=20)
    return paths + save(fig, out, "e1_failures_by_policy")


def fig_e2(rows: list[Row], out: Path) -> list[Path]:
    speedup = measured_speedup(rows)
    sweep = [
        r for r in rows if r.get("point_id") != "local_baseline" and r.get("alpha") is not None
    ]
    by = group(sweep, "alpha")
    xs, groups = [float(k) for k in by], list(by.values())
    fig, ax = plt.subplots(figsize=(6.5, 4))
    line(ax, xs, groups, makespan, "makespan", color="#4c78a8")
    ax.set_xlabel(f"{ALPHA} (1 = speed only, 0 = reliability only)")
    ax.set_ylabel("makespan (s)", color="#4c78a8")
    ax2 = ax.twinx()
    line(ax2, xs, groups, wasted, "wasted work", color="#e45756")
    line(ax2, xs, groups, failed_attempts, "failed attempts", color="#f58518", linestyle="--")
    ax2.set_ylabel("wasted work (s) / failed attempts")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper right")
    ax.set_title("E2: speedup-reliability tradeoff (mean ± 95% CI)")
    paths = save(fig, out, "e2_tradeoff_curve")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for alpha, g in zip(xs, groups, strict=True):
        sx, hx, _ = mean_ci(speedup(r) for r in g)
        ry, hy, _ = mean_ci(reliability_proxy(r) for r in g)
        ax.errorbar(sx, ry, xerr=_err(hx), yerr=_err(hy), marker="o", capsize=3, color="#4c78a8")
        ax.annotate(f"{ALPHA}={alpha:g}", (sx, ry), textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("speedup vs local")
    ax.set_ylabel("reliability proxy (1 - failed/attempts)")
    ax.set_title("E2: speedup vs reliability")
    return paths + save(fig, out, "e2_pareto")


def fig_e3(rows: list[Row], out: Path) -> list[Path]:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for policy, prow in group(rows, "policy").items():
        by = group(prow, "cluster.drop_prob")
        xs, groups = [float(k or 0) for k in by], list(by.values())
        line(a1, xs, groups, makespan, str(policy))
        line(a2, xs, groups, wasted, str(policy))
    a1.set_xlabel("drop probability per result")
    a1.set_ylabel("makespan (s)")
    a2.set_xlabel("drop probability per result")
    a2.set_ylabel("wasted work (s)")
    a1.legend()
    fig.suptitle("E3: failure intensity (mean ± 95% CI)")
    return save(fig, out, "e3_failure_sweep")


def fig_e4(rows: list[Row], out: Path) -> list[Path]:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    labels, groups = [], []
    for (policy, adaptive), g in sorted(
        _pairs(rows, "policy", "chunking.adaptive").items(), key=str
    ):
        labels.append(f"{policy}\n{'adaptive' if adaptive else 'fixed'}")
        groups.append(g)
    bar(a1, labels, groups, makespan, "makespan (s)")
    bar(a2, labels, groups, lambda r: r.get("messages_sent"), "messages sent")
    fig.suptitle("E4: adaptive vs fixed chunking (mean ± 95% CI)")
    return save(fig, out, "e4_chunking")


def _pairs(rows: list[Row], k1: str, k2: str) -> dict[tuple[Any, Any], list[Row]]:
    out: dict[tuple[Any, Any], list[Row]] = defaultdict(list)
    for r in rows:
        out[(r.get(k1), r.get(k2))].append(r)
    return dict(out)


def fig_e5(rows: list[Row], out: Path, all_rows: list[Row]) -> list[Path]:
    by = group(rows, "verification.base_prob")
    xs, groups = [float(k) for k in by], list(by.values())
    all_by = group(all_rows, "verification.base_prob")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    line(axes[0], xs, groups, detection_latency, "detection latency")
    axes[0].set_ylabel("detection latency (s)")
    line(axes[1], xs, [all_by[k] for k in by], correct, "correct output")  # failures count as wrong
    axes[1].set_ylabel("correct-output rate")
    axes[1].set_ylim(-0.05, 1.05)
    line(axes[2], xs, groups, lambda r: r.get("verification_overhead_pct"), "overhead")
    axes[2].set_ylabel("verification overhead (% of compute)")
    for ax in axes:
        ax.set_xlabel("verification.base_prob")
    fig.suptitle("E5: malicious workers (mean ± 95% CI)")
    return save(fig, out, "e5_malicious")


def fig_e7(rows: list[Row], out: Path, max_tasks: int = 24) -> list[Path]:
    from mdc.coordinator.policies import milp_offline as milp  # noqa: PLC0415

    gaps: dict[float, list[float]] = defaultdict(list)
    table = []
    for row in rows:
        run_dir = Path(str(row.get("run_dir")))
        snap = (run_dir if run_dir.is_absolute() else ROOT / run_dir) / "snapshot.json"
        if not snap.exists():
            continue
        alpha = float(row.get("alpha") or 0.5)
        try:
            result = milp.compare(
                milp.build_instance(milp.load_snapshot(snap), alpha=alpha, max_tasks=max_tasks)
            )
        except (ValueError, milp.MilpUnavailableError) as exc:
            print(f"  skipped {snap}: {exc}")
            continue
        gaps[alpha].append(result["gap_pct"])
        table.append(
            {
                "run_dir": str(run_dir),
                "alpha": alpha,
                **{k: result[k] for k in ("greedy_objective", "milp_objective", "gap_pct")},
            }
        )
    if table:
        with (out / "e7_gaps.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    fig, ax = plt.subplots(figsize=(6, 4))
    keys = sorted(gaps)
    if keys:
        ax.boxplot([gaps[k] for k in keys], tick_labels=[f"{ALPHA}={k:g}" for k in keys])
    ax.set_ylabel("greedy objective gap over MILP optimum (%)")
    ax.set_title(f"E7: online greedy vs offline MILP (first {max_tasks} tiles)")
    return save(fig, out, "e7_milp_gap")


def fig_generic(rows: list[Row], out: Path) -> list[Path]:
    by = group(rows, "point_id")
    fig, ax = plt.subplots(figsize=(max(6, len(by) * 0.8), 4))
    bar(ax, [str(k) for k in by], list(by.values()), makespan, "makespan (s)")
    return save(fig, out, "makespan_by_point")


# --------------------------------------------------------------- summary


SUMMARY_METRICS = (
    "makespan_s",
    "speedup",
    "wasted_work_s",
    "attempts_total",
    "replicas",
    "verifications",
    "verification_overhead_pct",
    "messages_sent",
)


def write_summary(rows: list[Row], exp_dir: Path) -> Path:
    path = exp_dir / "figures" / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "point_id",
                "runs",
                "failed",
                "correct_rate",
                *[f"{m}_{s}" for m in SUMMARY_METRICS for s in ("mean", "ci95")],
            ]
        )
        for point, g in group(rows, "point_id").items():
            good = ok_rows(g)
            cells: list[Any] = [
                point,
                len(g),
                len(g) - len(good),
                round(sum(correct(r) for r in g) / len(g), 3),
            ]
            for metric in SUMMARY_METRICS:
                mean, half, _ = mean_ci(r.get(metric) for r in good)
                cells += [
                    round(mean, 4) if not math.isnan(mean) else "",
                    round(half, 4) if not math.isnan(half) else "",
                ]
            writer.writerow(cells)
    return path


def make_figures(exp_dir: Path) -> list[Path]:
    """All figures for one experiment directory."""
    all_rows = load_results(exp_dir)
    rows = ok_rows(all_rows)
    out = exp_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    name = exp_dir.name
    paths = [write_summary(all_rows, exp_dir)]
    if name.startswith("e1"):
        paths += fig_e1(rows, out)
    elif name.startswith("e2"):
        paths += fig_e2(rows, out)
    elif name.startswith("e3"):
        paths += fig_e3(rows, out)
    elif name.startswith("e4"):
        paths += fig_e4(rows, out)
    elif name.startswith("e5"):
        paths += fig_e5(rows, out, all_rows)
    elif name.startswith("e7"):
        paths += fig_e7(rows, out)
    else:
        paths += fig_generic(rows, out)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot an experiment's results")
    parser.add_argument("experiment_dir", type=Path)
    args = parser.parse_args(argv)
    if not (args.experiment_dir / "results.csv").exists():
        print(f"error: no results.csv in {args.experiment_dir}", file=sys.stderr)
        return 2
    for path in make_figures(args.experiment_dir):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
