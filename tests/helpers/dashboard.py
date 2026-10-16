"""Test helpers for the dashboard: a fake backend and a logged-in test client."""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiohttp.test_utils import TestClient, TestServer

from mdc.common.config import Config, load_config
from mdc.common.events import EventLogger
from mdc.dashboard import auth
from mdc.dashboard.web import Dashboard, JobBusyError, JobRequest

TOKEN = secrets.token_urlsafe(24)  # fresh per test run, never a literal


@dataclass
class FakeBackend:
    """Records what the dashboard asked for."""

    images_dir: Path
    events: EventLogger = field(default_factory=EventLogger)
    busy: bool = False
    started: list[JobRequest] = field(default_factory=list)
    controls: list[tuple[str, str, float | None]] = field(default_factory=list)
    connected: set[str] = field(default_factory=lambda: {"w1"})
    output: Path | None = None
    last_report: dict[str, Any] | None = None
    fail_state: bool = False

    def state(self) -> dict[str, Any]:
        if self.fail_state:
            raise RuntimeError("secret internal detail /home/user/.mdc")
        return {
            "coordinator": {"address": "127.0.0.1:50010", "job_status": "idle"},
            "workers": [{"worker_id": "w1", "name": "phone-<b>", "state": "ACTIVE"}],
            "grid": None,
            "tiles": [],
            "events": self.events.recent(),
            "metrics": {},
            "result": None,
        }

    def state_update(self) -> dict[str, Any]:
        return {k: v for k, v in self.state().items() if k != "events"}

    def start_job(self, request: JobRequest) -> None:
        if self.busy:
            raise JobBusyError
        self.busy = True
        self.started.append(request)

    async def control(self, worker_id: str, action: str, value: float | None) -> bool:
        if worker_id not in self.connected:
            return False
        self.controls.append((worker_id, action, value))
        return True

    def output_png(self) -> Path | None:
        return self.output

    def report(self) -> dict[str, Any] | None:
        return self.last_report


def images_dir(tmp_path: Path) -> Path:
    d = tmp_path / "images"
    d.mkdir(parents=True)
    for name in ("a.png", "b.jpg"):
        (d / name).write_bytes(b"\x89PNG fake")
    (d / "notes.txt").write_text("not an image")
    return d


async def make_client(
    tmp_path: Path, *overrides: str, tls: bool = False, cfg: Config | None = None, **kw: Any
) -> tuple[TestClient, Dashboard, FakeBackend]:
    """A started test client (not logged in) for a Dashboard over a FakeBackend."""
    config = cfg or load_config(overrides=[f"security.state_dir={tmp_path / 'state'}", *overrides])
    backend = FakeBackend(images_dir(tmp_path))
    dash = Dashboard(config, backend, TOKEN, port=0, tls=tls, **kw)
    client = TestClient(TestServer(dash.build(), host="127.0.0.1"))
    await client.start_server()
    dash.set_port(client.port)
    return client, dash, backend


def origin(client: TestClient, scheme: str = "http") -> str:
    return f"{scheme}://127.0.0.1:{client.port}"


async def login(client: TestClient) -> str:
    """Log in with JSON; return the CSRF token."""
    r = await client.post("/login", json={"token": TOKEN}, headers={"Origin": origin(client)})
    assert r.status == 200, await r.text()
    state = await client.get("/api/state")
    assert state.status == 200
    return str((await state.json())["csrf"])


def headers(client: TestClient, csrf: str) -> dict[str, str]:
    return {"Origin": origin(client), auth.CSRF_HEADER: csrf}
