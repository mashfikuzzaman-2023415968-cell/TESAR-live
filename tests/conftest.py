"""Shared pytest fixtures.

Provided so far:

* ``repo_root``      — absolute path of the repository.
* ``tmp_state_dir``  — a fresh private state directory (mode 0700).
* ``tls_cert``       — a fresh coordinator certificate inside ``tmp_state_dir``.
* ``registry``       — a device registry in ``tmp_state_dir`` with a security log.
* ``enrolled(name)`` — enroll a device; returns the path of its 0600 join file.
* ``worker_proc(join_file, *flags)`` — start ``python -m mdc.worker`` as a
  subprocess (own process group, output captured to a log file); every
  process is killed at teardown, even if the test failed.
* an autouse guard that fails any test during which asyncio logged
  "Task was destroyed but it is pending" (a leaked background task).

* ``coordinator(*overrides, local_worker=False)`` — an in-process coordinator
  on an ephemeral loopback port, sharing ``tmp_state_dir`` (so ``enrolled``
  devices and ``tls_cert`` are its own); always stopped at teardown.

* ``cluster(profile, port)`` — ``scripts/local_cluster.py`` for a cluster
  profile, in its own process group (killed at teardown with every worker).

Every fixture cleans up its processes even when the test fails.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import subprocess
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from mdc.common.config import load_config
from mdc.common.credentials import write_join_file
from mdc.common.statedir import ensure_private_dir
from mdc.coordinator.certs import CertInfo, generate_certificate
from mdc.coordinator.coordinator import Coordinator
from mdc.coordinator.enrollment import DeviceRegistry
from mdc.coordinator.security_log import SecurityLog
from tests.helpers.procs import child_env

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def repo_root() -> Path:
    """Absolute path of the repository root."""
    return REPO_ROOT


@pytest.fixture
def tmp_state_dir(tmp_path: Path) -> Path:
    """A fresh private state directory (0700), like ``~/.mdc`` but per test."""
    return ensure_private_dir(tmp_path / "state")


@pytest.fixture
def tls_cert(tmp_state_dir: Path) -> CertInfo:
    """A fresh self-signed coordinator certificate (key 0600) in ``tmp_state_dir``."""
    return generate_certificate(tmp_state_dir, days=30)


@pytest.fixture
def security_log(tmp_state_dir: Path) -> Iterator[SecurityLog]:
    """The security audit log of ``tmp_state_dir`` (closed after the test)."""
    sec_log = SecurityLog.for_state_dir(tmp_state_dir)
    yield sec_log
    sec_log.close()


@pytest.fixture
def registry(tmp_state_dir: Path, security_log: SecurityLog) -> DeviceRegistry:
    """An empty device registry in ``tmp_state_dir``."""
    return DeviceRegistry(tmp_state_dir, security_log)


@pytest.fixture
def enrolled(registry: DeviceRegistry, tls_cert: CertInfo, tmp_path: Path) -> Callable[..., Path]:
    """Factory: ``enrolled("phone-a", port=1234)`` -> path of a 0600 join file."""

    def make(name: str, host: str = "127.0.0.1", port: int = 50010) -> Path:
        code = registry.enroll(name, host, port, tls_cert.fingerprint)
        return write_join_file(tmp_path / "joins" / f"{name}.join", code)

    return make


@dataclass
class WorkerProcess:
    """A worker subprocess started by the ``worker_proc`` fixture."""

    proc: subprocess.Popen[bytes]
    log_path: Path

    @property
    def pid(self) -> int:
        """Process id."""
        return self.proc.pid

    def log_text(self) -> str:
        """Everything the worker printed so far."""
        return self.log_path.read_text(errors="replace") if self.log_path.exists() else ""

    def stop(self, timeout: float = 15.0) -> int:
        """SIGTERM (graceful), then SIGKILL of the whole group if it does not exit."""
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout)
            except subprocess.TimeoutExpired:
                self.kill()
        return self.proc.wait()

    def kill(self) -> None:
        """Kill the worker and anything it started."""
        if self.proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait(10)


@pytest.fixture
def worker_proc(tmp_path: Path) -> Iterator[Callable[..., WorkerProcess]]:
    """Factory: ``worker_proc(join_file, "--coordinator", "127.0.0.1:1234", ...)``.

    The worker runs with ``HOME`` set to a fresh directory and ``--battery mock``
    unless overridden; output goes to ``<tmp>/worker-N.log``.
    """
    started: list[WorkerProcess] = []

    def start(join_file: Path, *flags: str, home: Path | None = None) -> WorkerProcess:
        home_dir = home or (tmp_path / f"home-{len(started)}")
        home_dir.mkdir(parents=True, exist_ok=True)
        log_path = tmp_path / f"worker-{len(started)}.log"
        args = [sys.executable, "-m", "mdc.worker", "--join-file", str(join_file), *flags]
        if "--battery" not in flags:
            args += ["--battery", "mock"]
        with log_path.open("wb") as log_file:
            proc = subprocess.Popen(
                args,
                cwd=home_dir,
                env=child_env(HOME=str(home_dir)),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # own process group: killpg cleans up everything
            )
        handle = WorkerProcess(proc, log_path)
        started.append(handle)
        return handle

    try:
        yield start
    finally:
        for handle in started:
            handle.kill()


@pytest.fixture
async def coordinator(
    tmp_path: Path, tmp_state_dir: Path, tls_cert: CertInfo
) -> AsyncIterator[Callable[..., Awaitable[Coordinator]]]:
    """Factory: ``c = await coordinator("scheduler.policy=local", local_worker=True)``."""
    started: list[Coordinator] = []

    async def start(
        *overrides: str, local_worker: bool = False, revocation_check_s: float = 0.2
    ) -> Coordinator:
        cfg = load_config(
            overrides=[
                f"security.state_dir={tmp_state_dir}",
                "network.tcp_port=0",
                "network.coordinator_bind=127.0.0.1",
                "network.beacon_modes=[]",
                f"workers.local_worker={'true' if local_worker else 'false'}",
                "scheduler.policy=round_robin",
                *overrides,
            ]
        )
        instance = Coordinator(
            cfg,
            output_dir=tmp_path / f"out{len(started)}",
            cache_dir=tmp_path / "cache",
            revocation_check_s=revocation_check_s,
        )
        await instance.start()
        started.append(instance)
        return instance

    try:
        yield start
    finally:
        for instance in started:
            await instance.stop()


@pytest.fixture
def cluster(tmp_path: Path, tmp_state_dir: Path) -> Iterator[Callable[..., WorkerProcess]]:
    """Factory: ``cluster("configs/cluster_profiles/hetero5.yaml", port)``."""
    started: list[WorkerProcess] = []

    def start(profile: str | Path, port: int, seed: int = 1) -> WorkerProcess:
        log_path = tmp_path / f"cluster-{len(started)}.log"
        with log_path.open("wb") as log_file:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "local_cluster.py"),
                    str(REPO_ROOT / profile),
                    "--coordinator",
                    f"127.0.0.1:{port}",
                    "--state-dir",
                    str(tmp_state_dir),
                    "--runs-dir",
                    str(tmp_path / "runs"),
                    "--seed",
                    str(seed),
                ],
                cwd=tmp_path,
                env=child_env(),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # the launcher and all its workers in one group
            )
        handle = WorkerProcess(proc, log_path)
        started.append(handle)
        return handle

    try:
        yield start
    finally:
        for handle in started:
            handle.stop()
            handle.kill()


class _PendingTaskDetector(logging.Handler):
    """Collects asyncio's "Task was destroyed but it is pending" errors."""

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "Task was destroyed but it is pending" in message:
            self.messages.append(message)


_RESULTS_ENV = "MDC_GATE_RESULTS"
_gate_results: dict[str, dict[str, str]] = {}


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Record every test id and whether it was skipped (for the phase gate).

    The gate's skip/regression guard (spec G.2 step 15) needs the list of
    tests and skip reasons. Writing them as JSON avoids parsing XML.
    """
    if not os.environ.get(_RESULTS_ENV):
        return
    entry = _gate_results.setdefault(report.nodeid, {"outcome": "passed", "reason": ""})
    if report.skipped:
        entry["outcome"] = "skipped"
        longrepr = report.longrepr
        entry["reason"] = str(longrepr[2]) if isinstance(longrepr, tuple) else str(longrepr)
    elif report.failed:
        entry["outcome"] = "failed"


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Write the recorded results to ``$MDC_GATE_RESULTS`` (if set)."""
    target = os.environ.get(_RESULTS_ENV)
    if target:
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text(json.dumps(_gate_results, indent=1, sort_keys=True))


@pytest.fixture(autouse=True)
def _no_destroyed_pending_tasks() -> Iterator[None]:
    """Fail the test if a pending asyncio task was garbage-collected.

    That message means a background task was never awaited/cancelled — the
    kind of leak spec Section 12 forbids.
    """
    detector = _PendingTaskDetector()
    asyncio_log = logging.getLogger("asyncio")
    asyncio_log.addHandler(detector)
    try:
        yield
    finally:
        asyncio_log.removeHandler(detector)
    if detector.messages:
        pytest.fail("asyncio reported destroyed pending task(s): " + "; ".join(detector.messages))
