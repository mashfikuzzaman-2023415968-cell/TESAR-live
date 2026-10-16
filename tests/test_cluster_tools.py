"""Tests for scripts/local_cluster.py and scripts/chaos.py."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import signal
import stat
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

from mdc.common.credentials import read_join_file
from mdc.coordinator.enrollment import DeviceRegistry

REPO = Path(__file__).resolve().parents[1]


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LC = _load("local_cluster")
CH = _load("chaos")


# ---------------------------------------------------------- local_cluster


@pytest.mark.parametrize("name", ["homogeneous4", "hetero5", "hetero_faulty6", "hetero_malicious6"])
def test_shipped_profiles_are_valid(name: str) -> None:
    profile = LC.load_profile(REPO / "configs" / "cluster_profiles" / f"{name}.yaml")
    fault_flags = {
        "--drop-prob",
        "--crash-prob",
        "--crash-after-s",
        "--malicious-prob",
        "--slowdown",
        "--flaky-mean-s",
        "--net-delay-ms",
        "--net-bw-kbps",
    }
    uses_faults = any(set(w.args) & fault_flags for w in profile.workers)
    assert profile.lab_mode == uses_faults  # lab mode exactly when needed


def test_malicious_profile_has_the_two_dishonest_workers() -> None:
    profile = LC.load_profile(REPO / "configs/cluster_profiles/hetero_malicious6.yaml")
    probs = sorted(
        w.args[w.args.index("--malicious-prob") + 1]
        for w in profile.workers
        if "--malicious-prob" in w.args
    )
    assert probs == ["0.05", "0.3"]


@pytest.mark.parametrize(
    "content",
    [
        "[1, 2]",
        "workers: []",
        "extra: 1\nworkers: [{name: a}]",
        "coordinator: nohost\nworkers: [{name: a}]",
        "lab_mode: yes-please\nworkers: [{name: a}]",
        "workers: [{name: 'bad name'}]",
        "workers: [{name: a, slots: 9}]",
        "workers: [{name: a, args: [--slowdown, 3]}]",  # unquoted number
        "workers: [{name: a}, {name: a}]",
        "workers: [{name: a, colour: red}]",
    ],
)
def test_bad_profiles_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "p.yaml"
    path.write_text(content)
    with pytest.raises(LC.ProfileError):
        LC.load_profile(path)


def test_worker_argv_has_no_secret_and_defaults_to_mock_battery(tmp_path: Path) -> None:
    spec = LC.WorkerSpec("w", 2, ["--slowdown", "3"])
    argv = LC.worker_argv(spec, tmp_path / "w.join", "127.0.0.1:1", True, 7, tmp_path)
    assert argv[argv.index("--join-file") + 1] == str(tmp_path / "w.join")
    assert "--lab-mode" in argv and argv[argv.index("--seed") + 1] == "7"
    assert argv[argv.index("--battery") + 1] == "mock"
    assert not any(a.startswith("MDC1.") for a in argv)
    own = LC.WorkerSpec("v", 1, ["--battery", "sysfs"])
    argv = LC.worker_argv(own, tmp_path / "v.join", "h:1", False, 0, tmp_path)
    assert argv.count("--battery") == 1 and "--lab-mode" not in argv


def test_prepare_enrolls_then_rekeys(tmp_path: Path, tmp_state_dir: Path) -> None:
    profile = LC.load_profile(REPO / "configs/cluster_profiles/homogeneous4.yaml")
    joins = LC.prepare(profile, tmp_state_dir, tmp_path / "runs", "127.0.0.1:50999")
    assert sorted(joins) == ["phone-1", "phone-2", "phone-3", "phone-4"]
    first = read_join_file(joins["phone-1"])
    assert (first.host, first.port) == ("127.0.0.1", 50999)
    assert stat.S_IMODE(joins["phone-1"].stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "runs/cluster_secrets").stat().st_mode) == 0o700
    again = LC.prepare(profile, tmp_state_dir, tmp_path / "runs", "127.0.0.1:50999")
    second = read_join_file(again["phone-1"])
    assert second.worker_id == first.worker_id and second.secret != first.secret
    assert len(DeviceRegistry(tmp_state_dir).devices()) == 4


def test_cluster_launch_and_stop(tmp_path: Path, tmp_state_dir: Path) -> None:
    profile = LC.Profile("127.0.0.1:1", False, [LC.WorkerSpec("solo", 1, [])])
    runs = tmp_path / "runs"
    joins = LC.prepare(profile, tmp_state_dir, runs, "127.0.0.1:1")
    procs = LC.launch(profile, joins, runs, "127.0.0.1:1", seed=3)
    log = runs / "cluster_logs/solo/worker.log"
    try:
        pids = json.loads((runs / "cluster_pids.json").read_text())
        assert pids == {"solo": procs["solo"].pid}
        deadline = time.monotonic() + 30
        while "starting" not in (log.read_text() if log.exists() else ""):
            assert time.monotonic() < deadline, "worker did not start"
            time.sleep(0.05)
    finally:
        codes = LC.stop_all(procs)
    assert codes == {"solo": 0}  # SIGTERM after start-up -> clean worker exit
    assert (runs / "cluster_logs/solo/stdout.log").exists()


def test_local_cluster_cli_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert LC.main([str(tmp_path / "missing.yaml")]) == 2
    assert "error" in capsys.readouterr().err


# ------------------------------------------------------------------ chaos


def test_parse_actions_and_window() -> None:
    assert CH.parse_action("kill", "slow-phone@3.5") == CH.Action(3.5, "slow-phone", "kill")
    assert CH.parse_action("stop", "a@b@2") == CH.Action(2.0, "a@b", "stop")
    for bad in ["nobody", "@3", "x@-1", "x@soon"]:
        with pytest.raises(CH.ChaosError):
            CH.parse_action("kill", bad)
    assert CH.parse_window("5-15") == (5.0, 15.0)
    for bad in ["5", "a-b", "9-3", "-1-2"]:
        with pytest.raises(CH.ChaosError):
            CH.parse_window(bad)


def test_random_kills_seeded_and_bounded() -> None:
    names = ["a", "b", "c", "d"]
    one = CH.random_kills(names, 2, (5, 10), random.Random(1))  # noqa: S311
    two = CH.random_kills(names, 2, (5, 10), random.Random(1))  # noqa: S311
    assert one == two and len({a.name for a in one}) == 2
    assert all(5 <= a.at <= 10 and a.signal_name == "kill" for a in one)
    with pytest.raises(CH.ChaosError):
        CH.random_kills(names, 5, (0, 1), random.Random(0))  # noqa: S311


def test_build_plan_sorted_and_validated() -> None:
    args = argparse.Namespace(
        kill=["a@5"], stop=["b@1"], cont=["b@3"], random_kills=0, window="0-0", seed=0
    )
    plan = CH.build_plan(args, {"a": 10, "b": 11})
    assert [(p.at, p.signal_name) for p in plan] == [(1.0, "stop"), (3.0, "cont"), (5.0, "kill")]
    args.kill = ["ghost@1"]
    with pytest.raises(CH.ChaosError, match="unknown"):
        CH.build_plan(args, {"a": 10})


def test_execute_sends_signals_on_schedule() -> None:
    sent: list[tuple[int, int]] = []
    now = [0.0]

    def send(pid: int, sig: int) -> None:
        if pid == 99:
            raise ProcessLookupError
        sent.append((pid, sig))

    def sleep(seconds: float) -> None:
        now[0] += seconds

    plan = [
        CH.Action(1.0, "a", "stop"),
        CH.Action(2.5, "a", "cont"),
        CH.Action(3.0, "gone", "kill"),
    ]
    log = CH.execute(plan, {"a": 10, "gone": 99}, send=send, clock=lambda: now[0], sleep=sleep)
    assert sent == [(10, signal.SIGSTOP), (10, signal.SIGCONT)]
    assert now[0] == pytest.approx(3.0)
    assert "already gone" in log[-1]


def test_wait_for_job_start(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    assert CH.wait_for_job_start(events, 0.2) is False
    events.write_text('{"t":0,"type":"coordinator_started"}\n{"t":1, "type": "job_started"}\n')
    assert CH.wait_for_job_start(events, 1.0) is True


def test_chaos_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pids = tmp_path / "pids.json"
    assert CH.main(["--pids", str(pids)]) == 2  # missing file
    pids.write_text(json.dumps({"a": 999999}))
    assert CH.main(["--pids", str(pids), "--kill", "a@0"]) == 0
    assert "already gone" in capsys.readouterr().out
    assert (
        CH.main(["--pids", str(pids), "--events", str(tmp_path / "none.jsonl"), "--wait-s", "0.1"])
        == 1
    )
