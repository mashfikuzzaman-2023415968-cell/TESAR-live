"""Protecting worker devices and their owners (spec S.7, S.12 row 3, G.3 row 3)."""

from __future__ import annotations

import ast
import asyncio
import base64
import logging
import os
import stat
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mdc.common import protocol as P
from mdc.common.credentials import JoinCode, encode_join_code, read_join_file
from mdc.common.validation import TaskLimits
from mdc.coordinator.certs import CertInfo
from mdc.coordinator.enrollment import DeviceRegistry
from mdc.worker import __main__ as worker_cli
from mdc.worker.battery import MockBattery
from mdc.worker.discovery_listener import parse_beacon
from tests.helpers.procs import child_env
from tests.helpers.stub_coordinator import StubCoordinator
from tests.helpers.worker_harness import running_worker

pytestmark = pytest.mark.security
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
async def stub(tls_cert: CertInfo, registry: DeviceRegistry) -> AsyncIterator[StubCoordinator]:
    async with StubCoordinator(tls_cert, registry) as s:
        yield s


@pytest.fixture
def join_file(stub: StubCoordinator, enrolled: Callable[..., Path]) -> Path:
    return enrolled("phone-sec", port=stub.port)


# ------------------------------------------------------------ CLI gating


@pytest.mark.parametrize(
    ("argv", "fragment"),
    [
        (["--drop-prob", "0.5"], "requires --lab-mode"),
        (["--malicious-prob", "1"], "requires --lab-mode"),
        (["--crash-after-s", "3"], "requires --lab-mode"),
        (["--net-bw-kbps", "100"], "requires --lab-mode"),
        (["--allow-demo-control"], "requires --lab-mode"),
        (["--allow-root"], "requires --lab-mode"),
        (
            ["--local", "--coordinator", "127.0.0.1:1", "--lab-mode", "--slowdown", "2"],
            "never accepts faults",
        ),
        (["--local"], "requires --coordinator"),
        (["--join-code", "MDC1" + ".secret-on-the-command-line"], "process list"),
        (["--insecure-dev", "--coordinator", "192.168.0.5:50010"], "loopback"),
        (["--insecure-dev"], "requires --coordinator"),
        (["--slots", "9"], "1..8"),
        (["--coordinator", "nohost"], "HOST:PORT"),
        (["--coordinator", "h:0"], "1..65535"),
        (["--lab-mode", "--drop-prob", "1.5", "--join-file", "{join}"], "drop_prob"),
        # added after the Phase-3 audit (appended to keep earlier test ids stable)
        (["--local", "--coordinator", "192.168.0.10:50010"], "only connects to 127.0.0.1"),
        (["--name", "bad name; rm"], "--name must match"),
        (["--name", "x" * 33], "--name must match"),
        # Phase 10 (appended): colluding corruption is a fault like the others
        (["--collude-key", "gang"], "requires --lab-mode"),
    ],
)
def test_unsafe_cli_combinations_refused(
    argv: list[str],
    fragment: str,
    join_file: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MDC_JOIN_CODE", raising=False)
    argv = [a.replace("{join}", str(join_file)) for a in argv]
    assert worker_cli.main(argv) == worker_cli.EXIT_USAGE
    assert fragment in capsys.readouterr().err


def test_root_refused(
    monkeypatch: pytest.MonkeyPatch, join_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(os, "geteuid", lambda: 0, raising=False)
    assert worker_cli.main(["--join-file", str(join_file)]) == worker_cli.EXIT_USAGE
    assert "root" in capsys.readouterr().err
    args = worker_cli.build_parser().parse_args(["--lab-mode", "--allow-root"])
    worker_cli.check_safety(args)  # explicitly allowed for throw-away lab machines


def test_world_readable_join_file_refused(
    join_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    os.chmod(join_file, 0o644)
    assert worker_cli.main(["--join-file", str(join_file)]) == worker_cli.EXIT_USAGE
    assert "chmod 600" in capsys.readouterr().err


def test_missing_credentials_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("MDC_JOIN_CODE", raising=False)
    assert worker_cli.main([]) == worker_cli.EXIT_USAGE
    assert "no credentials" in capsys.readouterr().err


def test_join_code_from_stdin_and_env(join_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code = read_join_file(join_file)
    text = encode_join_code(code)
    monkeypatch.setattr(sys, "stdin", __import__("io").StringIO(text + "\n"))
    args = worker_cli.build_parser().parse_args(["--join-code", "-"])
    assert worker_cli.load_join(args) == code
    monkeypatch.setenv("MDC_JOIN_CODE", text)
    assert worker_cli.load_join(worker_cli.build_parser().parse_args([])) == code
    both = worker_cli.build_parser().parse_args(["--join-file", "x", "--join-code", "-"])
    with pytest.raises(worker_cli.UsageError):
        worker_cli.load_join(both)


# ------------------------------------------------------- network trust


async def test_pin_mismatch_sends_nothing(
    stub: StubCoordinator, join_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    good = read_join_file(join_file)
    wrong = JoinCode(
        good.host, good.port, good.worker_id, good.name, good.secret, "sha256:" + "00" * 32
    )
    async with running_worker(wrong, stub.port) as (_w, task, _r):
        for _ in range(100):
            if stub.connections:
                break
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.3)
        assert not task.done()  # keeps retrying with backoff
    assert stub.connections >= 1
    assert stub.first_messages == []  # not even a HELLO reached the impostor-checked peer
    assert "pin_mismatch" in caplog.text


async def test_unknown_task_kind_disconnects(
    stub: StubCoordinator, join_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    async with running_worker(read_join_file(join_file), stub.port):
        session = await stub.next_session()
        await session.send(
            P.make_message(
                P.TASK_ASSIGN, task_id="t", attempt_id="a", kind="shell", params={"cmd": "rm -rf /"}
            )
        )
        await asyncio.wait_for(session.closed.wait(), 5)
    assert "validation_failed" in caplog.text


@pytest.mark.parametrize(("lab_mode", "allow"), [(False, False), (True, False)])
async def test_control_ignored_without_permission(
    stub: StubCoordinator,
    join_file: Path,
    caplog: pytest.LogCaptureFixture,
    lab_mode: bool,
    allow: bool,
) -> None:
    caplog.set_level(logging.WARNING)
    async with running_worker(
        read_join_file(join_file), stub.port, lab_mode=lab_mode, allow_demo_control=allow
    ) as (worker, task, recorder):
        session = await stub.next_session()
        for action, value in [("crash", None), ("set_malicious", 1.0), ("leave", None)]:
            await session.send(P.make_message(P.CONTROL, action=action, value=value))
        await session.send(P.make_message(P.PING, seq=1, t_c=0.0))
        await session.recv(P.PONG)
        assert recorder.codes == []
        assert worker.faults.settings.malicious_prob == 0.0
        assert not task.done()
    assert caplog.text.count("control_ignored") == 3


async def test_battery_floor_protects_owner(stub: StubCoordinator, join_file: Path) -> None:
    low = MockBattery(start_pct=5.0, drain_pct_per_min=0.0)
    async with running_worker(
        read_join_file(join_file), stub.port, battery=low, min_battery_pct=15.0
    ):
        session = await stub.next_session()
        await session.send(
            P.make_message(
                P.TASK_ASSIGN,
                task_id="t",
                attempt_id="a",
                kind="synthetic",
                params={"seed": 1, "chain_len": 5},
            )
        )
        reply = await session.recv(P.TASK_RESULT)
        assert reply.header["error"] == "battery_low"


async def test_worker_limits_hold_against_coordinator(
    stub: StubCoordinator, join_file: Path
) -> None:
    async with running_worker(
        read_join_file(join_file), stub.port, limits=TaskLimits(blur_radius=1, blur_iterations=2)
    ):
        session = await stub.next_session()
        fields, payload = P.pack_array(np.zeros((30, 30), dtype=np.uint8))
        await session.send(
            P.make_message(
                P.TASK_ASSIGN,
                payload,
                task_id="t",
                attempt_id="a",
                kind="img_tile",
                params={"r": 1, "iters": 3},
                **fields,
            )
        )
        reply = await session.recv(P.TASK_RESULT)
        assert reply.header["status"] == "error"


def test_spoofed_and_oversize_beacons_ignored() -> None:
    fp = "sha256:" + "ab" * 32
    spoof = b'{"type":"MDC_BEACON","v":1,"host":"6.6.6.6","port":1,"fp":"sha256:' + b"cd" * 32
    assert parse_beacon(spoof + b'"}', fp) is None
    big = b'{"type":"MDC_BEACON","v":1,"host":"1.2.3.4","port":1,"fp":"' + fp.encode()
    assert parse_beacon(big + b'","x":"' + b"y" * 1100 + b'"}', fp) is None


# ---------------------------------------------------- code-level checks


def test_only_one_external_command_in_worker_code() -> None:
    """subprocess is used only by battery.py, only with argument lists and timeouts."""
    users = []
    for top in ("mdc/worker", "mdc/common", "mdc/tasks"):
        for path in (REPO / top).rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import | ast.ImportFrom):
                    names = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
                    if any(n.split(".")[0] == "subprocess" for n in names):
                        users.append(path.relative_to(REPO).as_posix())
    assert sorted(set(users)) == ["mdc/worker/battery.py"]
    source = (REPO / "mdc/worker/battery.py").read_text()
    assert "shell=" not in source and "timeout=TERMUX_TIMEOUT_S" in source


def test_worker_never_listens_in_code() -> None:
    for path in (REPO / "mdc/worker").rglob("*.py"):
        text = path.read_text()
        assert "start_server" not in text and ".listen(" not in text, path


# ------------------------------------------------- real worker process


def _listening_inodes() -> set[str]:
    inodes = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(table).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            cols = line.split()
            if len(cols) > 9 and cols[3] == "0A":  # 0A = LISTEN
                inodes.add(cols[9])
    return inodes


def _socket_inodes(pid: int) -> set[str]:
    inodes = set()
    for fd in Path(f"/proc/{pid}/fd").iterdir():
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if target.startswith("socket:["):
            inodes.add(target[8:-1])
    return inodes


@pytest.mark.integration
@pytest.mark.skipif(not Path("/proc/net/tcp").exists(), reason="needs Linux /proc")
@pytest.mark.timeout(60)
async def test_real_worker_has_no_listening_socket_and_writes_only_its_log(
    stub: StubCoordinator, join_file: Path, tmp_path: Path, worker_proc: Callable[..., Any]
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    worker = worker_proc(join_file, "--coordinator", f"127.0.0.1:{stub.port}", home=home)
    session = await stub.next_session(timeout=30)
    await session.send(
        P.make_message(
            P.TASK_ASSIGN,
            task_id="t",
            attempt_id="a",
            kind="synthetic",
            params={"seed": 1, "chain_len": 5},
        )
    )
    assert (await session.recv(P.TASK_RESULT)).header["status"] == "ok"
    worker_sockets = _socket_inodes(worker.pid)
    assert worker_sockets, "the worker should hold its TCP connection"
    assert not (worker_sockets & _listening_inodes()), "worker has a LISTENING socket"
    assert worker.stop() == 0, worker.log_text()
    created = {p.relative_to(home) for p in home.rglob("*")}
    log_dir = Path(".mdc-worker/logs")
    assert created == {Path(".mdc-worker"), log_dir, log_dir / "worker.log"}, created
    assert stat.S_IMODE((home / log_dir).stat().st_mode) == 0o700
    assert stat.S_IMODE((home / log_dir / "worker.log").stat().st_mode) == 0o600


@pytest.mark.timeout(60)
def test_diag_never_prints_secrets(join_file: Path, tmp_path: Path) -> None:
    code = read_join_file(join_file)
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "mdc.worker",
            "--diag",
            "--join-file",
            str(join_file),
            "--battery",
            "mock",
            "--diag-listen-s",
            "0.3",
            "--discovery-port",
            "0",
            "--coordinator",
            f"127.0.0.1:{code.port}",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        cwd=tmp_path,
        env=child_env(HOME=str(tmp_path)),
    )
    text = out.stdout + out.stderr
    assert out.returncode == 0, text
    assert "benchmark" in text and code.worker_id in text
    for secret_form in (
        code.secret.hex(),
        base64.urlsafe_b64encode(code.secret).decode().rstrip("="),
        base64.b64encode(code.secret).decode(),
        encode_join_code(code),
    ):
        assert secret_form not in text
    assert "MDC1." not in text


def test_lab_mode_banner(join_file: Path, tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mdc.worker",
            "--join-file",
            str(join_file),
            "--lab-mode",
            "--coordinator",
            "127.0.0.1:1",
            "--battery",
            "mock",
            "--crash-after-s",
            "0.5",
            "--log-dir",
            str(tmp_path / "logs"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env=child_env(),
    )
    assert "LAB MODE" in proc.stderr
    assert proc.returncode == 1  # the crash-after-s fault ended it


def test_sigterm_stops_worker_cleanly(join_file: Path, worker_proc: Callable[..., Any]) -> None:
    worker = worker_proc(join_file, "--coordinator", "127.0.0.1:1")  # keeps retrying
    log = Path(worker.log_path)
    deadline = time.monotonic() + 30
    while "starting" not in log.read_text(errors="replace"):  # ready for signals
        assert time.monotonic() < deadline, worker.log_text()
        time.sleep(0.05)
    assert worker.stop() == 0, worker.log_text()


def test_worker_proc_fixture_kills_process_group(
    join_file: Path, worker_proc: Callable[..., Any]
) -> None:
    worker = worker_proc(join_file, "--coordinator", "127.0.0.1:1")
    time.sleep(0.5)
    assert worker.proc.poll() is None
    worker.kill()
    assert worker.proc.poll() is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(worker.pid, 0)  # the whole group is gone


# ------------------------------------------------- in-process --diag paths


async def test_diag_verdicts(
    stub: StubCoordinator,
    join_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = read_join_file(join_file)
    parser = worker_cli.build_parser()
    base = ["--battery", "mock", "--diag-listen-s", "0.2", "--discovery-port", "0"]
    ok = parser.parse_args([*base, "--coordinator", f"127.0.0.1:{stub.port}"])
    assert await worker_cli.run_diag(ok, code) == 0
    out = capsys.readouterr().out
    assert "certificate pin matches" in out and code.worker_id in out
    wrong = JoinCode(
        code.host, code.port, code.worker_id, code.name, code.secret, "sha256:" + "00" * 32
    )
    assert await worker_cli.run_diag(ok, wrong) == 0
    assert "does NOT match" in capsys.readouterr().out
    unreachable = parser.parse_args([*base, "--coordinator", "127.0.0.1:1"])
    assert await worker_cli.run_diag(unreachable, code) == 0
    assert "failed" in capsys.readouterr().out
    assert await worker_cli.run_diag(parser.parse_args(base), None) == 0
    out = capsys.readouterr().out
    assert "no credentials" in out
    for text in (out,):
        assert code.secret.hex() not in text


def test_build_options_and_faults() -> None:
    parser = worker_cli.build_parser()
    args = parser.parse_args(
        [
            "--slots",
            "6",
            "--max-cpu-slots",
            "2",
            "--device-class",
            "phone",
            "--coordinator",
            "[::1]:50010",
            "--prefetch",
            "-3",
        ]
    )
    opts = worker_cli.build_options(args)
    assert (opts.slots, opts.device_class, opts.coordinator, opts.prefetch) == (
        2,
        "phone",
        ("::1", 50010),
        0,
    )
    faults, shaper = worker_cli.build_faults(
        parser.parse_args(["--lab-mode", "--slowdown", "3", "--net-delay-ms", "40", "--seed", "4"])
    )
    assert faults.settings.slowdown == 3.0 and shaper.delay_ms == 40
    with pytest.raises(worker_cli.UsageError):
        worker_cli.build_faults(parser.parse_args(["--lab-mode", "--net-bw-kbps", "0"]))
