"""Dashboard API behaviour (spec Phase 11 tests): state, images, job start/409, report,
websocket, and one run against a real coordinator."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import aiohttp
import pytest

from mdc.coordinator.coordinator import Coordinator
from mdc.coordinator.job import TaskNode, TaskStatus
from mdc.dashboard import auth
from mdc.dashboard.backend import CoordinatorBackend, tile_status
from mdc.dashboard.web import Dashboard, parse_control, parse_job_request
from tests.helpers.dashboard import TOKEN, headers, login, make_client, origin
from tests.helpers.procs import REPO

JOB = {
    "policy": "tesar",
    "alpha": 0.5,
    "image": "a.png",
    "blur_iterations": 4,
    "verification": True,
    "adaptive": True,
}


async def test_state_structure_and_images(tmp_path: Path) -> None:
    client, _dash, _backend = await make_client(tmp_path)
    try:
        csrf = await login(client)
        state = await (await client.get("/api/state")).json()
        assert {"coordinator", "workers", "grid", "tiles", "events", "metrics"} <= set(state)
        assert state["csrf"] == csrf and state["demo_controls"] is False
        images = await (await client.get("/api/images")).json()
        assert images == {"images": ["a.png", "b.jpg"]}  # notes.txt is not an image
        health = await client.get("/healthz")
        assert await health.json() == {"ok": True}
    finally:
        await client.close()


async def test_job_start_then_409(tmp_path: Path) -> None:
    client, _dash, backend = await make_client(tmp_path)
    try:
        csrf = await login(client)
        r = await client.post("/api/job/start", json=JOB, headers=headers(client, csrf))
        assert r.status == 202
        assert backend.started[0].image.name == "a.png" and backend.started[0].alpha == 0.5
        again = await client.post("/api/job/start", json=JOB, headers=headers(client, csrf))
        assert again.status == 409 and await again.json() == {"error": "job_running"}
    finally:
        await client.close()


@pytest.mark.parametrize(
    "change",
    [
        {"policy": "greedy"},
        {"alpha": 1.5},
        {"alpha": True},
        {"blur_iterations": 0},
        {"blur_iterations": 65},
        {"verification": "yes"},
        {"extra": 1},
    ],
)
async def test_job_start_rejects_bad_bodies(tmp_path: Path, change: dict) -> None:  # type: ignore[type-arg]
    client, _dash, backend = await make_client(tmp_path)
    try:
        csrf = await login(client)
        r = await client.post(
            "/api/job/start", json={**JOB, **change}, headers=headers(client, csrf)
        )
        assert r.status == 400 and await r.json() == {"error": "bad_request"}
        assert backend.started == []
    finally:
        await client.close()


async def test_report_and_output(tmp_path: Path) -> None:
    client, _dash, backend = await make_client(tmp_path)
    try:
        await login(client)
        assert (await client.get("/api/report")).status == 404
        assert (await client.get("/api/output.png")).status == 404
        backend.last_report = {"output_correct": True}
        backend.output = tmp_path / "out.png"
        backend.output.write_bytes(b"\x89PNG data")
        assert await (await client.get("/api/report")).json() == {"output_correct": True}
        png = await client.get("/api/output.png")
        assert png.status == 200 and await png.read() == b"\x89PNG data"
    finally:
        await client.close()


async def test_websocket_sends_snapshot_events_and_updates(tmp_path: Path) -> None:
    client, dash, backend = await make_client(tmp_path)
    try:
        await login(client)
        ws = await client.ws_connect("/ws", headers={"Origin": origin(client)})
        first = json.loads((await ws.receive(timeout=5)).data)
        assert first["kind"] == "state_full" and "workers" in first["data"]
        backend.events.emit("worker_quarantined", worker_id="w1")
        kinds = set()
        for _ in range(4):
            kinds.add(json.loads((await ws.receive(timeout=5)).data)["kind"])
        assert {"event", "state"} <= kinds
        await ws.close()
        await dash.stop()
    finally:
        await client.close()


async def test_logout_ends_the_session(tmp_path: Path) -> None:
    client, _dash, _backend = await make_client(tmp_path)
    try:
        csrf = await login(client)
        r = await client.post("/logout", json={}, headers=headers(client, csrf))
        assert r.status == 200
        assert (await client.get("/api/state")).status == 401
    finally:
        await client.close()


def test_parsers_directly(tmp_path: Path) -> None:
    images = tmp_path / "img"
    images.mkdir()
    (images / "x.png").write_bytes(b"x")
    job = parse_job_request({"policy": "local", "image": "x.png"}, images, 64)
    assert (job.alpha, job.blur_iterations, job.verification) == (0.5, 8, True)
    with pytest.raises(ValueError):
        parse_job_request([], images, 64)
    assert parse_control({"action": "crash"}) == ("crash", None)
    assert parse_control({"action": "set_slowdown", "value": 3}) == ("set_slowdown", 3.0)
    for bad in (
        {"action": "crash", "value": 1},
        {"action": "set_malicious", "value": 2},
        {"action": "set_slowdown", "value": 0.5},
        {"action": "set_drop", "value": True},
        {"action": "explode"},
        {"action": "crash", "x": 1},
        "crash",
    ):
        with pytest.raises(ValueError):
            parse_control(bad)


def test_tile_status_mapping() -> None:
    task = TaskNode("t", 1, "img_tile", [])
    for status, expected in [
        (TaskStatus.PENDING, "pending"),
        (TaskStatus.READY, "pending"),
        (TaskStatus.RUNNING, "running"),
        (TaskStatus.VERIFYING, "verifying"),
        (TaskStatus.DONE, "done"),
        (TaskStatus.INVALIDATED, "invalidated"),
    ]:
        task.status = status
        assert tile_status(task) == expected
    task.status, task.verified = TaskStatus.DONE, True
    assert tile_status(task) == "verified"


@pytest.mark.integration
async def test_dashboard_runs_a_job_on_a_real_coordinator(coordinator) -> None:  # type: ignore[no-untyped-def]
    coord: Coordinator = await coordinator(
        "scheduler.policy=local",
        "dashboard.port=0",
        "security.lab_mode=true",
        "dashboard.demo_controls=true",
        local_worker=True,
    )
    for _ in range(100):
        if coord.local_active():
            break
        await asyncio.sleep(0.1)
    backend = CoordinatorBackend(coord, images_dir=REPO / "data" / "images")
    dash = Dashboard(coord.cfg, backend, TOKEN)
    port = await dash.start()
    base = f"http://127.0.0.1:{port}"
    jar = aiohttp.CookieJar(unsafe=True)
    try:
        async with aiohttp.ClientSession(base_url=base, cookie_jar=jar) as http:
            r = await http.post("/login", json={"token": TOKEN}, headers={"Origin": base})
            assert r.status == 200
            state = await (await http.get("/api/state")).json()
            assert state["coordinator"]["job_status"] == "idle" and state["demo_controls"]
            assert any(w["trusted"] for w in state["workers"])
            hdrs = {"Origin": base, auth.CSRF_HEADER: state["csrf"]}
            body = {**JOB, "policy": "local", "image": "test_512.png", "blur_iterations": 2}
            assert (await http.post("/api/job/start", json=body, headers=hdrs)).status == 202
            busy = await http.post("/api/job/start", json=body, headers=hdrs)
            assert busy.status == 409  # the first job is still running
            for _ in range(300):
                state = await (await http.get("/api/state")).json()
                if state["result"] is not None:
                    break
                await asyncio.sleep(0.1)
            assert state["result"]["output_correct"] is True
            assert state["grid"]["cols"] > 0 and state["tiles"]
            assert all(t[2] in {"done", "verified"} for t in state["tiles"])
            report = await (await http.get("/api/report")).json()
            assert report["policy"] == "local" and report["output_correct"] is True
            png = await http.get("/api/output.png")
            assert png.status == 200 and (await png.read()).startswith(b"\x89PNG")
            ghost = await http.post(
                "/api/workers/nobody/control", json={"action": "crash"}, headers=hdrs
            )
            assert ghost.status == 404
            local = next(w["worker_id"] for w in state["workers"] if w["trusted"])
            slow = await http.post(
                f"/api/workers/{local}/control",
                json={"action": "set_slowdown", "value": 2},
                headers=hdrs,
            )
            assert slow.status == 200  # sent (the local worker ignores CONTROL by design)
            assert "demo_control" in {e["type"] for e in coord.events.recent()}
            update = backend.state_update()
            assert set(update) == {"coordinator", "workers", "grid", "tiles", "metrics", "result"}
    finally:
        await backend.stop()
        await dash.stop()
