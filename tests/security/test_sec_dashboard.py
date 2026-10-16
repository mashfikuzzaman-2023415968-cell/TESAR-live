"""Dashboard security (spec S.9, G.3 row 11)."""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path

import pytest

from mdc.common.config import load_config
from mdc.common.statedir import InsecurePermissionsError
from mdc.coordinator import __main__ as coordinator_cli
from mdc.dashboard import auth
from mdc.dashboard.headers import CSP
from mdc.dashboard.web import STATIC_DIR, list_images, require_safe_binding, resolve_image
from tests.helpers.dashboard import TOKEN, headers, login, make_client, origin

pytestmark = pytest.mark.security

JOB = {"policy": "tesar", "image": "a.png"}


async def test_everything_needs_a_session(tmp_path: Path) -> None:
    client, _d, _b = await make_client(tmp_path)
    try:
        for path in ("/api/state", "/api/images", "/api/report", "/api/output.png"):
            r = await client.get(path)
            assert r.status == 401 and await r.json() == {"error": "unauthorized"}, path
        r = await client.post("/api/job/start", json=JOB, headers={"Origin": origin(client)})
        assert r.status == 401
        page = await client.get("/", allow_redirects=False)
        assert page.status == 303 and page.headers["Location"] == "/login"
        ws = await client.get("/ws", headers={"Origin": origin(client), "Upgrade": "websocket"})
        assert ws.status == 401  # the websocket too
        assert (await client.get("/healthz")).status == 200  # public
        assert (await client.get("/static/style.css")).status == 200  # public
    finally:
        await client.close()


async def test_login_wrong_token_origin_and_rate_limit(tmp_path: Path) -> None:
    client, _d, _b = await make_client(tmp_path, "dashboard.login_rate_limit_per_min=3")
    try:
        good = {"Origin": origin(client)}
        assert (await client.post("/login", json={"token": TOKEN})).status == 403  # no Origin
        evil = {"Origin": "http://evil.example"}
        assert (await client.post("/login", json={"token": TOKEN}, headers=evil)).status == 403
        for _ in range(3):
            r = await client.post("/login", json={"token": "wrong"}, headers=good)
            assert r.status == 401
        r = await client.post("/login", json={"token": TOKEN}, headers=good)
        assert r.status == 429  # even the right token, once the budget is spent
        assert (await client.get("/api/state")).status == 401
    finally:
        await client.close()


async def test_form_login_sets_a_strict_cookie(tmp_path: Path) -> None:
    client, _d, _b = await make_client(tmp_path)
    try:
        r = await client.post(
            "/login",
            data={"token": TOKEN},
            headers={"Origin": origin(client)},
            allow_redirects=False,
        )
        assert r.status == 303 and r.headers["Location"] == "/"
        cookie = r.headers["Set-Cookie"]
        assert "HttpOnly" in cookie and "SameSite=Strict" in cookie and "Secure" not in cookie
        bad = await client.post(
            "/login",
            data={"token": "nope"},
            headers={"Origin": origin(client)},
            allow_redirects=False,
        )
        assert bad.status == 303 and bad.headers["Location"] == "/login?failed=1"
        page = await client.get("/login?failed=1")
        assert "Wrong token" in await page.text()
    finally:
        await client.close()


async def test_secure_cookie_on_https(tmp_path: Path) -> None:
    client, _d, _b = await make_client(tmp_path, tls=True)
    try:
        r = await client.post(
            "/login", json={"token": TOKEN}, headers={"Origin": origin(client, "https")}
        )
        assert r.status == 200 and "Secure" in r.headers["Set-Cookie"]
    finally:
        await client.close()


async def test_csrf_and_origin_are_required(tmp_path: Path) -> None:
    client, _d, backend = await make_client(tmp_path)
    try:
        csrf = await login(client)
        cases = [
            {"Origin": origin(client)},  # CSRF header missing
            {"Origin": origin(client), auth.CSRF_HEADER: "wrong"},
            {auth.CSRF_HEADER: csrf},  # Origin missing
            {"Origin": "http://evil.example", auth.CSRF_HEADER: csrf},
            {"Origin": origin(client, "https"), auth.CSRF_HEADER: csrf},  # wrong scheme
        ]
        for hdrs in cases:
            r = await client.post("/api/job/start", json=JOB, headers=hdrs)
            assert r.status == 403 and await r.json() == {"error": "csrf"}, hdrs
        assert backend.started == []
        ok = await client.post("/api/job/start", json=JOB, headers=headers(client, csrf))
        assert ok.status == 202
    finally:
        await client.close()


async def test_bad_host_is_rejected(tmp_path: Path) -> None:
    client, _d, _b = await make_client(tmp_path)
    try:
        for host in (f"evil.example:{client.port}", "127.0.0.1:1", "attacker"):
            r = await client.get("/healthz", headers={"Host": host})
            assert r.status == 400 and await r.json() == {"error": "bad_host"}
        ok = await client.get("/healthz", headers={"Host": f"localhost:{client.port}"})
        assert ok.status == 200
    finally:
        await client.close()


async def test_security_headers_and_csp(tmp_path: Path) -> None:
    client, _d, _b = await make_client(tmp_path)
    try:
        await login(client)
        for path in ("/", "/api/state", "/login", "/static/app.js"):
            r = await client.get(path)
            assert r.status == 200, path
            h = r.headers
            assert h["Content-Security-Policy"] == CSP
            assert "unsafe-inline" not in CSP and "unsafe-eval" not in CSP
            assert h["X-Content-Type-Options"] == "nosniff"
            assert h["X-Frame-Options"] == "DENY"
            assert h["Referrer-Policy"] == "same-origin"  # see DECISIONS.md (Origin: null)
            assert h["Cross-Origin-Opener-Policy"] == "same-origin"
        api = await client.get("/api/state")
        assert api.headers["Cache-Control"] == "no-store"
        missing = await client.get("/nope")
        assert missing.status == 404 and "Content-Security-Policy" in missing.headers
    finally:
        await client.close()


async def test_path_traversal_and_symlinks_rejected(tmp_path: Path) -> None:
    client, _d, backend = await make_client(tmp_path)
    secret = tmp_path / "secret.png"
    secret.write_bytes(b"secret")
    (backend.images_dir / "link.png").symlink_to(secret)
    try:
        csrf = await login(client)
        for name in ("../secret.png", "/etc/passwd", "link.png", "notes.txt", "a.png/..", 5):
            r = await client.post(
                "/api/job/start", json={**JOB, "image": name}, headers=headers(client, csrf)
            )
            assert r.status == 400, name
        assert "link.png" not in list_images(backend.images_dir)
        with pytest.raises(ValueError):
            resolve_image(backend.images_dir, "../secret.png")
        assert list_images(tmp_path / "missing") == []
    finally:
        await client.close()


async def test_demo_controls_need_lab_mode_and_the_switch(tmp_path: Path) -> None:
    client, _d, backend = await make_client(tmp_path)
    try:
        csrf = await login(client)
        r = await client.post(
            "/api/workers/w1/control", json={"action": "crash"}, headers=headers(client, csrf)
        )
        assert r.status == 403 and backend.controls == []
    finally:
        await client.close()
    client, _d, backend = await make_client(
        tmp_path / "lab", "security.lab_mode=true", "dashboard.demo_controls=true"
    )
    try:
        csrf = await login(client)
        hdr = headers(client, csrf)
        r = await client.post("/api/workers/w1/control", json={"action": "crash"}, headers=hdr)
        assert r.status == 200 and backend.controls == [("w1", "crash", None)]
        r = await client.post(
            "/api/workers/w1/control", json={"action": "format_disk"}, headers=hdr
        )
        assert r.status == 400
        r = await client.post("/api/workers/ghost/control", json={"action": "leave"}, headers=hdr)
        assert r.status == 404
    finally:
        await client.close()


def test_non_loopback_dashboard_requires_https(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = load_config(overrides=["dashboard.host=192.168.0.10"])
    assert require_safe_binding(cfg) is True  # served over HTTPS with the coordinator cert
    cfg.security.tls = False
    with pytest.raises(ValueError, match="HTTPS"):
        require_safe_binding(cfg)
    assert require_safe_binding(load_config()) is False  # loopback: plain HTTP is fine
    code = coordinator_cli.main(
        [
            "--insecure-dev",
            "--set",
            "security.tls=false",
            "--set",
            "network.coordinator_bind=127.0.0.1",
            "--set",
            "dashboard.host=192.168.0.10",
            "--set",
            f"security.state_dir={tmp_path / 'state'}",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert code == 2 and "HTTPS" in capsys.readouterr().err


async def test_errors_never_leak_details(tmp_path: Path) -> None:
    client, _d, backend = await make_client(tmp_path)
    try:
        csrf = await login(client)
        backend.fail_state = True
        r = await client.get("/api/state")
        body = await r.text()
        assert r.status == 500 and body == '{"error": "internal_error"}'
        assert "secret internal detail" not in body and "Traceback" not in body
        hdr = headers(client, csrf)
        bad_json = await client.post(
            "/api/job/start", data="{not json", headers={**hdr, "Content-Type": "application/json"}
        )
        assert bad_json.status == 400 and await bad_json.json() == {"error": "bad_request"}
        form = await client.post("/api/job/start", data="a=b", headers=hdr)
        assert form.status == 400  # JSON only
        huge = await client.post("/api/job/start", json={"x": "y" * 100_000}, headers=hdr)
        assert huge.status == 413 and await huge.json() == {"error": "too_large"}
    finally:
        await client.close()


async def test_websocket_rejects_foreign_origin(tmp_path: Path) -> None:
    client, _d, _b = await make_client(tmp_path)
    try:
        await login(client)
        r = await client.get(
            "/ws", headers={"Origin": "http://evil.example", "Upgrade": "websocket"}
        )
        assert r.status == 403
    finally:
        await client.close()


def test_frontend_has_no_dangerous_sinks_or_inline_code() -> None:
    js = (STATIC_DIR / "app.js").read_text()
    for sink in (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "new Function",
        "http://",
        "https://",
    ):
        assert sink not in js, sink
    for page in STATIC_DIR.glob("*.html"):
        html = page.read_text()
        assert not re.search(r"<[^>]*\son[a-z]+\s*=", html, re.IGNORECASE), page
        for match in re.finditer(r"<script\b[^>]*>(.*?)</script>", html, re.DOTALL):
            assert match.group(1).strip() == "", page  # only <script src=...>
        assert "http://" not in html and "https://" not in html


def test_admin_token_file_is_private(tmp_path: Path) -> None:
    token = auth.load_or_create_admin_token(tmp_path)
    path = tmp_path / auth.ADMIN_FILE
    assert len(token) >= 40 and auth.load_or_create_admin_token(tmp_path) == token
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    os.chmod(path, 0o644)
    with pytest.raises(InsecurePermissionsError):
        auth.load_or_create_admin_token(tmp_path)


def test_auth_primitives() -> None:
    assert auth.token_matches("abc", "abc")
    assert not auth.token_matches("abd", "abc")
    assert not auth.token_matches("", "abc")
    assert not auth.token_matches("é", "abc")
    assert not auth.token_matches("a" * 300, "a" * 300)
    assert auth.is_loopback_host("localhost") and auth.is_loopback_host("127.0.0.2")
    assert auth.is_loopback_host("::1") and not auth.is_loopback_host("192.168.0.10")
    assert not auth.is_loopback_host("example.com")
    assert "dash.lan" in auth.allowed_hosts("dash.lan", 443)
    assert "[::1]:8080" in auth.allowed_hosts("::1", 8080)
    assert not auth.host_ok(None, frozenset({"x"}))
    assert not auth.origin_ok(None, "http", frozenset({"x"}))


def test_sessions_expire_are_bounded_and_log_out() -> None:
    now = [0.0]
    store = auth.SessionStore(10.0, clock=lambda: now[0])
    first = store.create()
    assert store.get(first.session_id) == first
    assert store.get(None) is None and store.get("x" * 300) is None and store.get("nope") is None
    assert auth.csrf_matches(first.csrf, first)
    assert not auth.csrf_matches(None, first) and not auth.csrf_matches("x", first)
    now[0] = 10.0
    assert store.get(first.session_id) is None  # expired and dropped
    sessions = [store.create() for _ in range(auth.MAX_SESSIONS + 5)]
    assert len(store) == auth.MAX_SESSIONS and store.get(sessions[0].session_id) is None
    store.delete(sessions[-1].session_id)
    store.delete(None)
    assert store.get(sessions[-1].session_id) is None


def test_login_limiter_is_per_client_and_bounded() -> None:
    now = [0.0]
    limiter = auth.LoginLimiter(2, clock=lambda: now[0])
    assert limiter.allow("a") and limiter.allow("a") and not limiter.allow("a")
    assert limiter.allow("b")
    now[0] = 61.0
    assert limiter.allow("a")  # the window slid
    for i in range(auth.MAX_LIMITER_KEYS + 10):
        limiter.allow(f"ip{i}")
    assert len(limiter._counters) == auth.MAX_LIMITER_KEYS


async def test_null_origin_is_still_refused(tmp_path: Path) -> None:
    """Browsers send `Origin: null` from sandboxed or no-referrer contexts: never accept it."""
    client, _d, _b = await make_client(tmp_path)
    try:
        r = await client.post("/login", json={"token": TOKEN}, headers={"Origin": "null"})
        assert r.status == 403
        csrf = await login(client)
        bad = {"Origin": "null", auth.CSRF_HEADER: csrf}
        assert (await client.post("/api/job/start", json=JOB, headers=bad)).status == 403
    finally:
        await client.close()
