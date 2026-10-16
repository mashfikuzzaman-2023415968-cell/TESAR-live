"""Experiment framework (spec Phase 13, G.3 row 13): config parsing, dry run, CSV schema,
confidence intervals, figures."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import pytest

from tests.helpers.procs import REPO

sys.path.insert(0, str(REPO / "experiments"))
import plots
import run_matrix as rm

CONFIGS = sorted((REPO / "configs" / "experiments").glob("*.yaml"))


def write_cfg(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "exp.yaml"
    path.write_text(text)
    return path


BASE = """
name: demo
cluster_profile: configs/cluster_profiles/hetero5.yaml
repetitions: 2
seed: 10
local_baseline: true
base_overrides: [job.blur_iterations=2]
matrix:
  scheduler.policy: [fastest, tesar]
  cluster.drop_prob: [0.0, 0.2]
"""


@pytest.mark.parametrize("path", CONFIGS, ids=[p.stem for p in CONFIGS])
def test_shipped_configs_parse(path: Path) -> None:
    exp = rm.load_experiment(path)
    assert exp.name == path.stem and exp.cluster_profile.exists()
    assert rm.count_workers(exp.cluster_profile) >= 5
    runs = rm.plan(exp)
    assert len(runs) <= 40  # each experiment stays laptop-sized
    header = [ln for ln in path.read_text().splitlines() if ln.startswith("#")]
    assert any("minutes" in ln for ln in header)  # runtime documented at the top


def test_plan_point_ids_seeds_and_baseline(tmp_path: Path) -> None:
    exp = rm.load_experiment(write_cfg(tmp_path, BASE))
    runs = rm.plan(exp)
    assert len(runs) == 2 * (4 + 1)
    assert [r.rep for r in runs] == [0] * 5 + [1] * 5  # repetition-major
    assert {r.seed for r in runs if r.rep == 1} == {11}
    assert runs[0].point_id == "p00_policy-fastest_drop_prob-0.0"
    assert runs[4].point_id == rm.BASELINE_ID and runs[4].params == {"scheduler.policy": "local"}
    assert len({r.point_id for r in runs}) == 5
    assert runs[1].out_dir(tmp_path, "demo") == tmp_path / "demo" / runs[1].point_id / "rep0"


def test_split_params_maps_cluster_keys_to_worker_flags() -> None:
    sets, extra = rm.split_params(
        {"scheduler.alpha": 0.5, "chunking.adaptive": False, "cluster.drop_prob": 0.2}
    )
    assert sets == ["scheduler.alpha=0.5", "chunking.adaptive=false"]
    assert extra == ["--drop-prob", "0.2"]
    assert rm.split_params({"cluster.drop_prob": 0.0}) == ([], [])  # zero: flag omitted
    assert rm.split_params({"cluster.lab_mode": True})[1] == ["--lab-mode"]


@pytest.mark.parametrize(
    "bad",
    [
        "[1, 2]",
        BASE + "extra: 1\n",
        BASE.replace("name: demo", "name: Bad Name"),
        BASE.replace("repetitions: 2", "repetitions: 0"),
        BASE.replace("scheduler.policy: [fastest, tesar]", "policy: [fastest]"),
        BASE.replace("[fastest, tesar]", "[]"),
        BASE.replace("[job.blur_iterations=2]", "[noequals]"),
        BASE.replace("cluster_profile: configs/cluster_profiles/hetero5.yaml\n", ""),
        BASE.replace("seed: 10", "seed: ten"),
        BASE.replace("matrix:", "matrix: 5\nx:").split("\n  scheduler")[0],
    ],
)
def test_bad_configs_rejected(tmp_path: Path, bad: str) -> None:
    with pytest.raises(rm.ExperimentError):
        rm.load_experiment(write_cfg(tmp_path, bad))


def test_dry_run_prints_the_plan_and_runs_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = write_cfg(tmp_path, BASE)
    assert rm.main([str(cfg), "--dry-run", "--runs-root", str(tmp_path / "runs")]) == 0
    out = capsys.readouterr().out
    assert "demo: 10 run(s), 5 workers" in out and "workers --drop-prob 0.2" in out
    assert not (tmp_path / "runs" / "demo").exists()
    assert rm.main([str(cfg), "--dry-run", "--only", "3"]) == 0
    assert capsys.readouterr().out.count("[") == 3
    assert rm.main([str(tmp_path / "missing.yaml")]) == 2


def test_csv_schema(tmp_path: Path) -> None:
    exp = rm.load_experiment(write_cfg(tmp_path, BASE))
    run = rm.plan(exp)[1]
    out = run.out_dir(tmp_path, exp.name)
    out.mkdir(parents=True)
    report = dict.fromkeys(rm.REPORT_FIELDS, 1)
    report["detection_latency_s"] = [{"worker": "bad", "latency_s": 0.4}]
    report["per_worker"] = {"x": {}}
    (out / "report.json").write_text(json.dumps(report))
    results = tmp_path / "results.csv"
    rm.append_row(results, exp, rm.csv_row(exp, run, "ok", out))
    rm.append_row(results, exp, rm.csv_row(exp, rm.plan(exp)[0], "failed", tmp_path / "none"))
    with results.open() as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        assert reader.fieldnames == rm.columns(exp)
    assert len(rows) == 2  # one row per run
    assert rows[0]["scheduler.policy"] == "fastest" and rows[0]["cluster.drop_prob"] == "0.2"
    assert json.loads(rows[0]["detection_latency_s"])[0]["latency_s"] == 0.4
    assert "per_worker" not in rows[0] and rows[1]["status"] == "failed"
    assert rows[1]["makespan_s"] == ""


def test_confidence_interval_against_hand_computed_values() -> None:
    # x = 2, 4, 6, 8: mean 5, s = sqrt(20/3) = 2.58199, t(0.975, 3) = 3.18245
    # half-width = 3.18245 * 2.58199 / sqrt(4) = 4.10850
    mean, half, n = plots.mean_ci([2, 4, 6, 8, None, float("nan")])
    assert (mean, n) == (5.0, 4) and half == pytest.approx(4.10850, abs=1e-4)
    one = plots.mean_ci([3.0])
    assert one[0] == 3.0 and math.isnan(one[1])
    assert math.isnan(plots.mean_ci([])[0])
    assert plots.mean_ci([1, 1, 1])[1] == 0.0


def test_convert_and_metric_helpers() -> None:
    assert [plots.convert(v) for v in ("", "True", "false", "3", "2.5", "1e3", "abc")] == [
        None,
        True,
        False,
        3,
        2.5,
        1000.0,
        "abc",
    ]
    row = {"attempts_total": 10, "attempts_timeout": 1, "attempts_lost": 1, "attempts_error": 0}
    assert plots.reliability_proxy(row) == 0.8 and plots.reliability_proxy({}) is None
    lat = '[{"worker": "a", "latency_s": 0.2}, {"worker": "b", "latency_s": 0.4}]'
    assert plots.detection_latency({"detection_latency_s": lat}) == pytest.approx(0.3)
    assert plots.detection_latency({"detection_latency_s": "[]"}) is None
    assert plots.detection_latency({"detection_latency_s": "{bad"}) is None


def fake_results(
    exp_dir: Path, keys: dict[str, list], extra_rows: list[dict] | None = None
) -> None:  # type: ignore[type-arg]
    exp_dir.mkdir(parents=True)
    rows = []
    import itertools  # noqa: PLC0415

    for rep in range(3):
        for i, combo in enumerate(itertools.product(*keys.values())):
            row = dict(zip(keys, combo, strict=True))
            row.update(
                point_id=f"p{i:02d}",
                rep=rep,
                status="ok",
                makespan_s=1.0 + i + rep * 0.1,
                speedup=2.0,
                wasted_work_s=0.1 * i,
                attempts_total=20,
                attempts_ok=18,
                attempts_timeout=1,
                attempts_lost=1,
                attempts_error=0,
                messages_sent=100 + i,
                verification_overhead_pct=5.0 * i,
                output_correct=True,
                detection_latency_s='[{"worker": "bad", "latency_s": 0.5}]',
                run_dir="x",
            )
            rows.append(row)
    rows += extra_rows or []
    fields = sorted({k for r in rows for k in r})
    with (exp_dir / "results.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize(
    ("name", "keys", "figures"),
    [
        (
            "e1_baselines",
            {"policy": ["local", "tesar"]},
            ["e1_makespan_by_policy", "e1_failures_by_policy"],
        ),
        ("e2_alpha_sweep", {"alpha": [0.0, 0.5, 1.0]}, ["e2_tradeoff_curve", "e2_pareto"]),
        (
            "e3_failure_sweep",
            {"policy": ["tesar"], "cluster.drop_prob": [0.0, 0.2]},
            ["e3_failure_sweep"],
        ),
        (
            "e4_chunking",
            {"policy": ["fastest"], "chunking.adaptive": [True, False]},
            ["e4_chunking"],
        ),
        ("e5_malicious", {"verification.base_prob": [0.05, 1.0]}, ["e5_malicious"]),
        ("other", {"x": [1]}, ["makespan_by_point"]),
    ],
)
def test_figures_are_written(tmp_path: Path, name: str, keys: dict, figures: list[str]) -> None:  # type: ignore[type-arg]
    baseline = [
        {
            "point_id": "local_baseline",
            "rep": r,
            "status": "ok",
            "makespan_s": 4.0,
            "policy": "local",
        }
        for r in range(3)
    ]
    failed = [{"point_id": "p00", "rep": 9, "status": "failed"}]
    fake_results(tmp_path / name, keys, baseline + failed)
    assert plots.main([str(tmp_path / name)]) == 0
    out = tmp_path / name / "figures"
    for fig in figures:
        assert (out / f"{fig}.png").stat().st_size > 1000 and (out / f"{fig}.pdf").exists()
    summary = list(csv.DictReader((out / "summary.csv").read_text().splitlines()))
    assert any(r["failed"] == "1" for r in summary)


def test_e7_solves_the_milp_on_saved_snapshots(tmp_path: Path) -> None:
    from tests.test_policies import snapshot  # noqa: PLC0415

    exp = tmp_path / "e7_milp_small"
    rows = []
    for i, alpha in enumerate((0.25, 0.75)):
        run_dir = tmp_path / f"run{i}"
        run_dir.mkdir()
        (run_dir / "snapshot.json").write_text(json.dumps(snapshot(8)))
        rows.append(
            {
                "point_id": f"p{i}",
                "rep": 0,
                "status": "ok",
                "alpha": alpha,
                "run_dir": str(run_dir),
                "makespan_s": 1.0,
            }
        )
    rows.append(
        {
            "point_id": "p9",
            "rep": 0,
            "status": "ok",
            "alpha": 0.5,
            "run_dir": str(tmp_path / "no-snapshot"),
            "makespan_s": 1.0,
        }
    )
    fake_results(exp, {}, rows)
    plots.make_figures(exp)
    gaps = list(csv.DictReader((exp / "figures" / "e7_gaps.csv").read_text().splitlines()))
    assert len(gaps) == 2 and all(float(g["gap_pct"]) >= -1e-6 for g in gaps)
    assert (exp / "figures" / "e7_milp_gap.png").exists()
    assert plots.main([str(tmp_path / "empty")]) == 2


@pytest.mark.e2e
@pytest.mark.timeout(300)
def test_run_matrix_runs_real_jobs_and_resumes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = write_cfg(
        tmp_path,
        BASE.replace("repetitions: 2", "repetitions: 1")
        .replace("job.blur_iterations=2", "job.image_path=data/images/test_512.png")
        .replace("[0.0, 0.2]", "[0.0]")
        .replace("[fastest, tesar]", "[tesar]"),
    )
    root = tmp_path / "runs"
    assert rm.main([str(cfg), "--runs-root", str(root)]) == 0
    rows = list(csv.DictReader((root / "demo" / "results.csv").read_text().splitlines()))
    assert [r["status"] for r in rows] == ["ok", "ok"]  # the point + the local baseline
    assert {r["policy"] for r in rows} == {"tesar", "local"}
    assert rm.main([str(cfg), "--runs-root", str(root)]) == 0  # resume: nothing re-run
    assert "skipped" in capsys.readouterr().out
    assert len(list(csv.DictReader((root / "demo" / "results.csv").read_text().splitlines()))) == 2
    assert plots.main([str(root / "demo")]) == 0
