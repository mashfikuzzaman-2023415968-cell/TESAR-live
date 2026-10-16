"""End-to-end: coordinator + real worker processes run a real image job (spec Phase 4).

512 x 512 test image, r = 1, iters = 2, base tile 128 (16 tiles + 4 row tasks).
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from mdc.common.config import load_config
from mdc.coordinator.coordinator import Coordinator
from mdc.coordinator.security_log import read_security_log
from tests.helpers.procs import REPO, child_env

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(180)]

JOB = [
    f"job.image_path={REPO / 'data/images/test_512.png'}",
    "job.blur_radius=1",
    "job.blur_iterations=2",
    "job.base_tile=128",
]


async def run_with_workers(
    coordinator: Callable[..., Awaitable[Coordinator]],
    enrolled: Callable[..., Path],
    worker_proc: Callable[..., object],
    policy: str,
) -> dict:
    coord = await coordinator(f"scheduler.policy={policy}", *JOB, local_worker=True)
    workers = []
    for i in range(3):
        join = enrolled(f"e2e-{policy.replace('_', '-')}-{i}")
        workers.append(worker_proc(join, "--coordinator", f"127.0.0.1:{coord.port}"))
    assert await coord.wait_for_workers(3, 60) == 3, [w.log_text() for w in workers]
    assert coord.local_active()
    report = await coord.run_job(timeout_s=120)
    for w in workers:
        assert w.proc.poll() is None, w.log_text()  # no worker crashed
    return report


async def test_e2e_round_robin(coordinator, enrolled, worker_proc) -> None:  # type: ignore[no-untyped-def]
    report = await run_with_workers(coordinator, enrolled, worker_proc, "round_robin")
    assert report["output_correct"] is True and report["differing_pixels"] == 0
    remote = {n: w for n, w in report["per_worker"].items() if not w["trusted"]}
    assert len(remote) == 3
    assert all(w["tasks_ok"] >= 1 for w in remote.values()), remote
    assert report["attempts_ok"] == report["attempts_total"] == 20


async def test_e2e_local(coordinator, enrolled, worker_proc) -> None:  # type: ignore[no-untyped-def]
    report = await run_with_workers(coordinator, enrolled, worker_proc, "local")
    assert report["output_correct"] is True
    remote = [w for w in report["per_worker"].values() if not w["trusted"]]
    assert all(w["tasks_ok"] == 0 for w in remote)
    assert report["per_worker"]["local"]["tasks_ok"] == 20


def test_e2e_coordinator_cli(enrolled, worker_proc, tmp_path: Path, tmp_state_dir: Path) -> None:  # type: ignore[no-untyped-def]
    """``python -m mdc.coordinator --headless --run-job --exit-after-job --wait-workers 2``."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    joins = [enrolled(f"cli-{i}", port=port) for i in range(2)]
    out = tmp_path / "run"
    coord = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mdc.coordinator",
            "--headless",
            "--run-job",
            "--exit-after-job",
            "--wait-workers",
            "2",
            "--wait-timeout-s",
            "60",
            "--output-dir",
            str(out),
            "--set",
            f"security.state_dir={tmp_state_dir}",
            "--set",
            f"network.tcp_port={port}",
            "--set",
            "network.coordinator_bind=127.0.0.1",
            "--set",
            "network.beacon_modes=[]",
            "--set",
            "scheduler.policy=round_robin",
            *[a for j in JOB for a in ("--set", j)],
        ],
        cwd=tmp_path,
        env=child_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        for join in joins:
            worker_proc(join, "--coordinator", f"127.0.0.1:{port}")
        output, _ = coord.communicate(timeout=150)
    finally:
        if coord.poll() is None:
            coord.kill()
            coord.communicate()
    assert coord.returncode == 0, output
    assert "Workers can connect with" in output and "correct ✓" in output
    report = json.loads((out / "report.json").read_text())
    assert report["output_correct"] is True and report["n_workers_remote"] == 2
    assert (out / "coordinator.log").exists()


def test_coordinator_cli_refuses_unimplemented_policy(tmp_path: Path, tmp_state_dir: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mdc.coordinator",
            "--headless",
            "--output-dir",
            str(tmp_path),
            "--set",
            f"security.state_dir={tmp_state_dir}",
            "--set",
            "network.tcp_port=0",
            "--set",
            "workers.local_worker=false",
            "--set",
            "network.beacon_modes=[]",
            "--set",
            "scheduler.policy=greedy",  # every real policy exists since Phase 9
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=child_env(),
        cwd=tmp_path,
        check=False,
    )
    assert proc.returncode == 2
    assert "tesar" in proc.stderr and "round_robin" in proc.stderr


def test_coordinator_cli_insecure_combination(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "mdc.coordinator", "--set", "security.tls=false"],
        capture_output=True,
        text=True,
        timeout=60,
        env=child_env(),
        cwd=tmp_path,
        check=False,
    )
    assert proc.returncode == 2 and "--insecure-dev" in proc.stderr


async def test_e2e_insecure_dev_mode_on_loopback(tmp_path: Path, tmp_state_dir: Path) -> None:
    """``security.tls: false`` works only with --insecure-dev on loopback, and is logged."""
    cfg = load_config(
        overrides=[
            f"security.state_dir={tmp_state_dir}",
            "security.tls=false",
            "network.coordinator_bind=127.0.0.1",
            "network.tcp_port=0",
            "network.beacon_modes=[]",
            "scheduler.policy=local",
            *JOB,
        ],
        insecure_dev=True,
    )
    async with Coordinator(
        cfg, insecure_dev=True, output_dir=tmp_path / "out", cache_dir=tmp_path / "cache"
    ) as coord:
        await coord.wait_for_workers(0, 60)
        assert coord.local_active()
        report = await coord.run_job(timeout_s=120)
    assert report["output_correct"] is True
    log = read_security_log(tmp_state_dir / "logs" / "security.jsonl")
    assert any(r["type"] == "insecure_mode_enabled" for r in log)
