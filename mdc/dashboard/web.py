"""Live dashboard: aiohttp REST API, websocket and static files (spec Phase 11, S.9).

Routes (all but ``/login``, ``/static/*`` and ``/healthz`` need a session)::

    GET  /                       index.html (redirects to /login without a session)
    GET  /login, POST /login     admin-token login (form or JSON), rate-limited
    POST /logout
    GET  /healthz                {"ok": true}
    GET  /api/state              full snapshot + the session's CSRF token
    GET  /api/images             selectable images (basenames from data/images)
    POST /api/job/start          start a job (409 while one runs)
    POST /api/workers/{id}/control   demo controls (lab mode + demo_controls only)
    GET  /api/output.png, /api/report
    GET  /ws                     snapshot, then every event and a state update every 0.5 s

The app talks to the coordinator only through a :class:`Backend`, so the
security layer is tested without starting a coordinator.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import ssl
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from aiohttp import WSMsgType, web

from mdc.common.config import POLICIES, Config
from mdc.common.events import EventLogger, to_jsonable
from mdc.common.validation import CONTROL_ACTIONS, MAX_SLOWDOWN
from mdc.dashboard import auth
from mdc.dashboard.headers import Handler, error, errors_and_headers

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
MAX_BODY_BYTES = 16 * 1024
STATE_PUSH_S = 0.5
WS_QUEUE_MAX = 1000
PUBLIC_PATHS = frozenset({"/login", "/healthz"})
SESSION_KEY = web.RequestKey("session", auth.Session)  # the authenticated Session

LOGIN_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TESAR-Live login</title><link rel="stylesheet" href="/static/style.css"></head>
<body class="login"><main class="login-box"><h1>TESAR-Live</h1>
<p>Paste the admin token from <code>state_dir/admin_token</code>.</p>
<form method="post" action="/login">
<input type="password" name="token" autocomplete="current-password" required autofocus>
<button type="submit">Log in</button></form>{message}</main></body></html>
"""


class JobBusyError(RuntimeError):
    """A job is already running."""


@dataclass(frozen=True)
class JobRequest:
    """Validated body of ``POST /api/job/start``."""

    policy: str
    alpha: float
    image: Path
    blur_iterations: int
    verification: bool
    adaptive: bool


class Backend(Protocol):
    """What the dashboard needs from the coordinator."""

    events: EventLogger
    images_dir: Path

    def state(self) -> dict[str, Any]: ...

    def state_update(self) -> dict[str, Any]: ...

    def start_job(self, request: JobRequest) -> None: ...

    async def control(self, worker_id: str, action: str, value: float | None) -> bool: ...

    def output_png(self) -> Path | None: ...

    def report(self) -> dict[str, Any] | None: ...


# ------------------------------------------------------------------ helpers


def list_images(images_dir: Path) -> list[str]:
    """Basenames of the selectable images (regular files with an image suffix)."""
    if not images_dir.is_dir():
        return []
    return sorted(
        p.name
        for p in images_dir.iterdir()
        if p.is_file() and not p.is_symlink() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def resolve_image(images_dir: Path, name: object) -> Path:
    """The image file for a basename chosen in the browser (S.9 item 7).

    Raises ``ValueError`` unless ``name`` is one of :func:`list_images` and its
    resolved path lies directly in the resolved images directory.
    """
    if not isinstance(name, str) or name not in list_images(images_dir):
        raise ValueError("unknown image")
    base = images_dir.resolve()
    path = (base / name).resolve()
    if path.parent != base or path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError("unknown image")
    return path


def parse_job_request(body: object, images_dir: Path, max_iterations: int) -> JobRequest:
    """Validate a job-start body (raises ``ValueError``)."""
    if not isinstance(body, dict):
        raise ValueError("body must be an object")
    allowed = {"policy", "alpha", "image", "blur_iterations", "verification", "adaptive"}
    if set(body) - allowed:
        raise ValueError("unknown field")
    policy = body.get("policy")
    if policy not in POLICIES:
        raise ValueError("policy")
    alpha = body.get("alpha", 0.5)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 <= alpha <= 1:
        raise ValueError("alpha")
    iterations = body.get("blur_iterations", 8)
    if type(iterations) is not int or not 1 <= iterations <= max_iterations:
        raise ValueError("blur_iterations")
    flags = [body.get("verification", True), body.get("adaptive", True)]
    if any(type(flag) is not bool for flag in flags):
        raise ValueError("flags")
    return JobRequest(
        str(policy),
        float(alpha),
        resolve_image(images_dir, body.get("image")),
        iterations,
        bool(flags[0]),
        bool(flags[1]),
    )


def parse_control(body: object) -> tuple[str, float | None]:
    """Validate a control body (raises ``ValueError``)."""
    if not isinstance(body, dict) or set(body) - {"action", "value"}:
        raise ValueError("body")
    action = body.get("action")
    if action not in CONTROL_ACTIONS:
        raise ValueError("action")
    value = body.get("value")
    if action in {"crash", "leave"}:
        if value is not None:
            raise ValueError("value")
        return str(action), None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("value")
    high = MAX_SLOWDOWN if action == "set_slowdown" else 1.0
    low = 1.0 if action == "set_slowdown" else 0.0
    if not low <= value <= high:
        raise ValueError("value")
    return str(action), float(value)


async def read_json(request: web.Request) -> object:
    """A JSON body (``application/json`` only, size-limited by the app)."""
    if request.content_type != "application/json":
        raise web.HTTPBadRequest
    try:
        return json.loads(await request.text())
    except (ValueError, UnicodeDecodeError):
        raise web.HTTPBadRequest from None


# ---------------------------------------------------------------------- app


class Dashboard:
    """Builds and runs the dashboard application."""

    def __init__(
        self,
        config: Config,
        backend: Backend,
        token: str,
        port: int | None = None,
        tls: bool = False,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.cfg = config
        self.backend = backend
        self.token = token
        self.tls = tls
        self.scheme = "https" if tls else "http"
        self.port = config.dashboard.port if port is None else port
        kwargs = {} if clock is None else {"clock": clock}
        self.sessions = auth.SessionStore(config.dashboard.session_ttl_s, **kwargs)
        self.limiter = auth.LoginLimiter(config.dashboard.login_rate_limit_per_min, **kwargs)
        self.hosts = auth.allowed_hosts(config.dashboard.host, self.port)
        self.runner: web.AppRunner | None = None
        self._sockets: set[web.WebSocketResponse] = set()

    @property
    def controls_enabled(self) -> bool:
        """Demo controls need both switches (spec S.9 / Phase 11)."""
        return self.cfg.dashboard.demo_controls and self.cfg.security.lab_mode

    def set_port(self, port: int) -> None:
        """Use the actually bound port for Host/Origin checks (port 0 in tests)."""
        self.port = port
        self.hosts = auth.allowed_hosts(self.cfg.dashboard.host, port)

    # ------------------------------------------------------------ middleware

    @web.middleware
    async def guard(self, request: web.Request, handler: Handler) -> web.StreamResponse:
        """Host check, then session, then CSRF + Origin for state changes."""
        if not auth.host_ok(request.headers.get("Host"), self.hosts):
            return error(400, "bad_host")
        if request.path in PUBLIC_PATHS or request.path.startswith("/static/"):
            return await handler(request)
        session = self.sessions.get(request.cookies.get(auth.SESSION_COOKIE))
        if session is None:
            if request.path == "/" and request.method == "GET":
                return redirect("/login")
            return error(401)
        if request.method not in {"GET", "HEAD"} and not (
            auth.origin_ok(request.headers.get("Origin"), self.scheme, self.hosts)
            and auth.csrf_matches(request.headers.get(auth.CSRF_HEADER), session)
        ):
            return error(403, "csrf")
        request[SESSION_KEY] = session
        return await handler(request)

    def build(self) -> web.Application:
        """The aiohttp application."""
        app = web.Application(
            middlewares=[errors_and_headers, self.guard], client_max_size=MAX_BODY_BYTES
        )
        app.router.add_get("/", self.index)
        app.router.add_get("/login", self.login_page)
        app.router.add_post("/login", self.login)
        app.router.add_post("/logout", self.logout)
        app.router.add_get("/healthz", self.healthz)
        app.router.add_get("/api/state", self.api_state)
        app.router.add_get("/api/images", self.api_images)
        app.router.add_post("/api/job/start", self.api_job_start)
        app.router.add_post("/api/workers/{worker_id}/control", self.api_control)
        app.router.add_get("/api/output.png", self.api_output)
        app.router.add_get("/api/report", self.api_report)
        app.router.add_get("/ws", self.websocket)
        app.router.add_static("/static/", STATIC_DIR, show_index=False, follow_symlinks=False)
        return app

    # ---------------------------------------------------------------- pages

    async def index(self, request: web.Request) -> web.StreamResponse:
        return web.FileResponse(STATIC_DIR / "index.html")

    async def healthz(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def login_page(self, request: web.Request) -> web.Response:
        failed = request.query.get("failed") == "1"
        message = '<p class="error">Wrong token or too many attempts.</p>' if failed else ""
        return web.Response(text=LOGIN_PAGE.format(message=message), content_type="text/html")

    async def login(self, request: web.Request) -> web.StreamResponse:
        """Check Origin and rate limit, then the token; set the session cookie."""
        form = request.content_type != "application/json"
        if not auth.origin_ok(request.headers.get("Origin"), self.scheme, self.hosts):
            return error(403, "origin")
        if not self.limiter.allow(request.remote or "?"):
            return self._login_failed(form, 429)
        if form:
            token = (await request.post()).get("token")
        else:
            body = await read_json(request)
            token = body.get("token") if isinstance(body, dict) else None
        if not isinstance(token, str) or not auth.token_matches(token, self.token):
            return self._login_failed(form, 401)
        session = self.sessions.create()
        response = redirect("/") if form else web.json_response({"ok": True})
        response.set_cookie(
            auth.SESSION_COOKIE,
            session.session_id,
            httponly=True,
            samesite="Strict",
            secure=self.tls,
            max_age=int(self.cfg.dashboard.session_ttl_s),
            path="/",
        )
        return response

    @staticmethod
    def _login_failed(form: bool, status: int) -> web.Response:
        return redirect("/login?failed=1") if form else error(status)

    async def logout(self, request: web.Request) -> web.Response:
        session = request[SESSION_KEY]
        self.sessions.delete(session.session_id)
        response = web.json_response({"ok": True})
        response.del_cookie(auth.SESSION_COOKIE, path="/")
        return response

    # ------------------------------------------------------------------ API

    async def api_state(self, request: web.Request) -> web.Response:
        state = self.backend.state()
        session = request[SESSION_KEY]
        state["csrf"] = session.csrf
        state["demo_controls"] = self.controls_enabled
        return web.json_response(state, dumps=_dumps)

    async def api_images(self, request: web.Request) -> web.Response:
        return web.json_response({"images": list_images(self.backend.images_dir)})

    async def api_job_start(self, request: web.Request) -> web.Response:
        body = await read_json(request)
        limit = self.cfg.security.max_task_params.blur_iterations
        try:
            job = parse_job_request(body, self.backend.images_dir, limit)
        except ValueError:
            return error(400)
        try:
            self.backend.start_job(job)
        except JobBusyError:
            return error(409, "job_running")
        return web.json_response({"ok": True}, status=202)

    async def api_control(self, request: web.Request) -> web.Response:
        if not self.controls_enabled:
            return error(403, "controls_disabled")
        try:
            action, value = parse_control(await read_json(request))
        except ValueError:
            return error(400)
        sent = await self.backend.control(request.match_info["worker_id"], action, value)
        return web.json_response({"ok": True}) if sent else error(404)

    async def api_output(self, request: web.Request) -> web.StreamResponse:
        path = self.backend.output_png()
        return web.FileResponse(path) if path is not None else error(404)

    async def api_report(self, request: web.Request) -> web.Response:
        report = self.backend.report()
        return web.json_response(report, dumps=_dumps) if report is not None else error(404)

    # ------------------------------------------------------------ websocket

    async def websocket(self, request: web.Request) -> web.StreamResponse:
        """Push events and state; the Origin must be ours (cross-site websocket hijack)."""
        if not auth.origin_ok(request.headers.get("Origin"), self.scheme, self.hosts):
            return error(403, "origin")
        ws = web.WebSocketResponse(heartbeat=20.0, max_msg_size=4096)
        await ws.prepare(request)
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(WS_QUEUE_MAX)

        def on_event(event: dict[str, Any]) -> None:
            with contextlib.suppress(asyncio.QueueFull):  # a slow browser loses events
                queue.put_nowait(event)

        self.backend.events.subscribe(on_event)
        self._sockets.add(ws)
        pusher = asyncio.create_task(self._push(ws, queue))
        try:
            await ws.send_str(_dumps({"kind": "state_full", "data": self.backend.state()}))
            async for msg in ws:  # the browser sends nothing we act on
                if msg.type in {WSMsgType.ERROR, WSMsgType.CLOSE}:
                    break
        finally:
            pusher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pusher
            self.backend.events.unsubscribe(on_event)
            self._sockets.discard(ws)
        return ws

    async def _push(self, ws: web.WebSocketResponse, queue: asyncio.Queue[dict[str, Any]]) -> None:
        while not ws.closed:
            await asyncio.sleep(STATE_PUSH_S)
            while not queue.empty():
                await ws.send_str(_dumps({"kind": "event", "data": queue.get_nowait()}))
            await ws.send_str(_dumps({"kind": "state", "data": self.backend.state_update()}))

    # -------------------------------------------------------------- serving

    async def start(self, ssl_context: ssl.SSLContext | None = None) -> int:
        """Serve on ``dashboard.host:port``; returns the bound port."""
        self.runner = web.AppRunner(self.build(), access_log=None, handle_signals=False)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.cfg.dashboard.host, self.port, ssl_context=ssl_context)
        await site.start()
        server = site._server  # aiohttp exposes no public accessor for the bound port
        sockets = getattr(server, "sockets", None) or []
        if sockets:
            self.set_port(int(sockets[0].getsockname()[1]))
        return self.port

    async def stop(self) -> None:
        """Close websockets and stop serving."""
        for ws in list(self._sockets):
            await ws.close()
        if self.runner is not None:
            await self.runner.cleanup()
            self.runner = None


def _dumps(value: Any) -> str:
    return json.dumps(to_jsonable(value), separators=(",", ":"))


def redirect(location: str) -> web.Response:
    """A 303 See Other (returned, not raised, so cookies set on it are kept)."""
    return web.Response(status=303, headers={"Location": location})


def require_safe_binding(config: Config) -> bool:
    """True if the dashboard must use HTTPS (non-loopback host); raises if it cannot."""
    if auth.is_loopback_host(config.dashboard.host):
        return False
    if not config.security.tls:
        raise ValueError(
            f"dashboard.host {config.dashboard.host!r} is not loopback: the dashboard must "
            "then be served over HTTPS, which needs security.tls: true"
        )
    return True
