"""Phone deployment (spec Phase 12, G.3 row 12): worker zip, diagnostics, calibration."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from mdc import calibrate
from mdc import cli as mdc_cli
from mdc.coordinator.diag import port_free
from tests.helpers.procs import REPO, child_env

sys.path.insert(0, str(REPO / "scripts"))
import package_worker

pytestmark = pytest.mark.security


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    out = tmp_path_factory.mktemp("dist")
    return package_worker.build(out)


def test_zip_contains_only_the_worker(built: tuple[Path, str]) -> None:
    path, _ = built
    names = set(zipfile.ZipFile(path).namelist())
    for required in (
        "mdc/__init__.py",
        "mdc/worker/__main__.py",
        "mdc/worker/worker.py",
        "mdc/common/protocol.py",
        "mdc/tasks/image_pipeline.py",
        "requirements-worker.txt",
        "scripts/termux_setup.sh",
        "WORKER_README.txt",
    ):
        assert required in names, required
    assert not [n for n in names if n.startswith(("mdc/coordinator", "mdc/dashboard", "tests"))]
    assert not [n for n in names if n.endswith(".pyc") or "__pycache__" in n]
    assert not [n for n in names if n.endswith((".join", ".pem", ".key")) or "admin_token" in n]


def test_zip_has_no_secrets_and_a_matching_hash(built: tuple[Path, str]) -> None:
    path, digest = built
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            text = zf.read(name).decode("utf-8", errors="replace")
            for pattern in package_worker.SECRET_PATTERNS:
                assert not pattern.search(text), name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    line = (path.parent / f"{path.name}.sha256").read_text()
    assert line == f"{digest}  {path.name}\n"  # the format sha256sum -c expects


def test_zip_is_reproducible(built: tuple[Path, str], tmp_path: Path) -> None:
    _, digest = built
    assert package_worker.build(tmp_path)[1] == digest


def test_secret_scan_refuses_to_package(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    for rel in package_worker.EXTRA_FILES:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("# ok\n")
    for package in package_worker.PACKAGES:
        (root / package).mkdir(parents=True, exist_ok=True)
        (root / package / "__init__.py").write_text("")
    (root / "mdc/worker/leak.py").write_text("CODE = 'MDC1." + "A" * 40 + "'\n")
    with pytest.raises(ValueError, match="refusing to package secrets"):
        package_worker.build(tmp_path / "out", root)
    (root / "mdc/worker/leak.py").write_text(
        "-----BEGIN EC " + "PRIVATE KEY-----\n"
    )  # split: repo scan
    assert package_worker.find_secrets(root, [Path("mdc/worker/leak.py")])


def test_unpacked_zip_runs_without_coordinator_code(
    built: tuple[Path, str], tmp_path: Path
) -> None:
    path, _ = built
    target = tmp_path / "phone"
    zipfile.ZipFile(path).extractall(target)
    probe = (
        "import sys, mdc, mdc.worker.__main__, mdc.worker.worker\n"
        "print(mdc.__file__)\n"
        "bad = [m for m in sys.modules if m.startswith(('mdc.coordinator', 'mdc.dashboard',"
        " 'yaml', 'aiohttp', 'PIL', 'cryptography'))]\n"
        "print(bad)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=target,
        env=child_env(PYTHONPATH=""),
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    ).stdout.splitlines()
    assert out[0].startswith(str(target)) and out[1] == "[]"
    diag = subprocess.run(
        [sys.executable, "-m", "mdc.worker", "--diag", "--battery", "mock"],
        cwd=target,
        env=child_env(PYTHONPATH="", HOME=str(tmp_path)),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert diag.returncode == 0, diag.stderr
    assert "benchmark:" in diag.stdout


def test_coordinator_diag_prints_no_secrets(tmp_path: Path) -> None:
    state = tmp_path / "state"
    code = tmp_path / "w.join"
    assert (
        mdc_cli.main(
            ["enroll", "--name", "w1", "--out", str(code), "--set", f"security.state_dir={state}"]
        )
        == 0
    )
    join_text = code.read_text().strip()
    (state / "admin_token").write_text("dashboard-token-value\n")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mdc.coordinator",
            "--diag",
            "--set",
            f"security.state_dir={state}",
            "--set",
            "network.beacon_modes=[]",
            "--output-dir",
            str(tmp_path / "out"),
        ],
        env=child_env(HOME=str(tmp_path)),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout + proc.stderr
    assert "cert fp:" in out and "1 enrolled (1 active)" in out and "beacon:     disabled" in out
    assert join_text not in out and "MDC1." not in out and "dashboard-token-value" not in out


def test_port_check() -> None:
    import socket  # noqa: PLC0415

    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        assert port_free("127.0.0.1", port) == "in use"
    assert port_free("127.0.0.1", 0).startswith("free")


def write_report(path: Path, workers: dict[str, dict]) -> None:  # type: ignore[type-arg]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"makespan_s": 10.0, "per_worker": workers}))


def test_calibration_recovers_known_constants(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rng = np.random.default_rng(0)
    for i in range(6):
        compute, transfer = float(rng.uniform(5, 50)), float(rng.uniform(1, 20))
        energy = 2.5 * compute + 1.1 * transfer + 0.3 * 10.0  # idle 0.3 W over 10 s
        write_report(
            tmp_path / f"run{i}" / "report.json",
            {
                "phone-a": {
                    "device_class": "phone",
                    "compute_s": compute,
                    "transfer_s": transfer,
                    "energy_meas_j": energy,
                    "idle_baseline_w": 0.3,
                },
                "local": {"trusted": True, "energy_meas_j": 5.0, "compute_s": 1, "transfer_s": 0},
                "laptop": {"device_class": "laptop", "energy_meas_j": None},
            },
        )
    (tmp_path / "broken" / "report.json").parent.mkdir()
    (tmp_path / "broken" / "report.json").write_text("{not json")
    fits = calibrate.fit(calibrate.load_samples(tmp_path))
    assert len(fits) == 1 and fits[0].device_class == "phone" and fits[0].samples == 6
    assert fits[0].p_compute_w == pytest.approx(2.5, rel=1e-6)
    assert fits[0].p_tx_w == pytest.approx(1.1, rel=1e-6)
    assert mdc_cli.main(["calibrate-energy", "--runs", str(tmp_path)]) == 0
    assert "energy.p_compute_w_phone=2.500" in capsys.readouterr().out


def test_calibration_needs_enough_data(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert mdc_cli.main(["calibrate-energy", "--runs", str(tmp_path / "none")]) == 1
    write_report(
        tmp_path / "r" / "report.json",
        {"p": {"device_class": "phone", "compute_s": 1, "transfer_s": 1, "energy_meas_j": 3}},
    )
    assert mdc_cli.main(["calibrate-energy", "--runs", str(tmp_path)]) == 1
    assert "need at least 2" in capsys.readouterr().out


def test_nonnegative_fit() -> None:
    x = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    y = np.array([2.0, 4.0, -5.0])  # unconstrained would give a negative transfer power
    coef = calibrate.nnls2(x, y)
    assert (coef >= 0).all() and coef[0] == pytest.approx(2.0)
