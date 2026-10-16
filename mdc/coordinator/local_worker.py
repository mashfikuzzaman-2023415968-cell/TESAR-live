"""The trusted local worker (spec Phase 4 item 3, S.4.7, S.8.4).

The laptop itself is also a worker: the ``local`` baseline, the verifier of
last resort (Phase 10) and the fallback for tasks that failed too often
(Phase 5). It is a **real subprocess** connected over loopback, so it uses
exactly the same code path as remote workers — no special-cased shortcuts
that could hide bugs.

Its credential is **ephemeral**: a fresh random secret for every coordinator
run, kept only in memory, never written to ``devices.json``. The join code
reaches the child through the ``MDC_JOIN_CODE`` environment variable — never
argv, which every user on the machine can read with ``ps``.

The server accepts this credential only from a loopback peer, and only a
connection using it may claim ``trusted_local`` (anything else is rejected
and security-logged as ``trusted_local_spoof``).

If the subprocess dies it is restarted (after 1 s, then with backoff), and a
``local_worker_restarted`` event is logged.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
from pathlib import Path

import mdc
from mdc.common.auth import make_secret
from mdc.common.credentials import JoinCode, encode_join_code
from mdc.common.events import EventLogger
from mdc.common.ids import new_id

log = logging.getLogger(__name__)

LOCAL_NAME = "local"
STOP_TIMEOUT_S = 5.0
MAX_RESTART_DELAY_S = 30.0


class LocalCredential:
    """In-memory credential of the local worker for this coordinator run."""

    def __init__(self, fp: str, port: int) -> None:
        self.worker_id = new_id("local")
        self.secret = make_secret()
        self.join = JoinCode(
            "127.0.0.1", port if port > 0 else 1, self.worker_id, LOCAL_NAME, self.secret, fp
        )

    def __repr__(self) -> str:
        return f"LocalCredential(worker_id={self.worker_id!r})"


class LocalWorker:
    """Spawns and supervises ``python -m mdc.worker --local``."""

    def __init__(
        self,
        credential: LocalCredential,
        port: int,
        slots: int,
        log_dir: Path,
        events: EventLogger,
        insecure_dev: bool = False,
    ) -> None:
        self.credential = credential
        self.port = port
        self.slots = slots
        self.log_dir = log_dir
        self.events = events
        self.insecure_dev = insecure_dev
        self.restarts = 0
        self._proc: asyncio.subprocess.Process | None = None
        self._task: asyncio.Task[None] | None = None
        self._stopping = False

    def _argv(self) -> list[str]:
        argv = [
            sys.executable,
            "-m",
            "mdc.worker",
            "--local",
            "--coordinator",
            f"127.0.0.1:{self.port}",
            "--slots",
            str(self.slots),
            # The local worker is the laptop the operator is already running the
            # coordinator on: no battery floor (spec Phase 8 exempts it from energy limits).
            "--min-battery",
            "0",
            "--log-dir",
            str(self.log_dir),
        ]
        if self.insecure_dev:
            argv.append("--insecure-dev")
        return argv

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["MDC_JOIN_CODE"] = encode_join_code(self.credential.join)
        # Make sure the child imports this very package, wherever it lives.
        package_root = str(Path(mdc.__file__).resolve().parent.parent)
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = package_root + (os.pathsep + existing if existing else "")
        return env

    async def _spawn(self) -> asyncio.subprocess.Process:
        # Argument list, no shell; output goes to the worker's own log file.
        return await asyncio.create_subprocess_exec(
            *self._argv(),
            env=self._env(),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

    async def _supervise(self) -> None:
        delay = 1.0
        while not self._stopping:
            self._proc = await self._spawn()
            log.info("local worker started (pid %d)", self._proc.pid)
            code = await self._proc.wait()
            if self._stopping:
                break
            self.restarts += 1
            log.warning("local worker exited with %s; restarting in %.0f s", code, delay)
            self.events.emit("local_worker_restarted", exit_code=code, restarts=self.restarts)
            await asyncio.sleep(delay)
            delay = min(delay * 2, MAX_RESTART_DELAY_S)

    @property
    def pid(self) -> int | None:
        """PID of the current subprocess."""
        return self._proc.pid if self._proc is not None else None

    def start(self) -> None:
        """Start the supervisor task."""
        if self._task is None:
            self._stopping = False
            self._task = asyncio.create_task(self._supervise())

    async def stop(self) -> None:
        """Terminate the subprocess (SIGTERM, then SIGKILL after 5 s)."""
        self._stopping = True
        proc = self._proc
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), STOP_TIMEOUT_S)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
