"""Security headers and generic errors for every dashboard response (spec S.9 items 5, 9).

The Content-Security-Policy allows nothing but the dashboard's own files: no
inline script or style, no external resources, no framing. API responses are
never cached. Errors never carry stack traces: a handler that raises gets
``{"error": "internal_error"}`` and the exception goes to the coordinator log.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from aiohttp import web

log = logging.getLogger(__name__)

Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]

CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)
SECURITY_HEADERS = {
    "Content-Security-Policy": CSP,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    # "same-origin", not "no-referrer": under no-referrer browsers send `Origin: null` on the
    # dashboard's own POSTs (login form, fetch), which the CSRF Origin check must reject.
    # same-origin still sends no referrer to any other site (DECISIONS.md, 2026-10-16).
    "Referrer-Policy": "same-origin",
    "Cross-Origin-Opener-Policy": "same-origin",
}
FIRST_ERROR_STATUS = 400
ERROR_NAMES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "too_large",
    429: "too_many_requests",
}


def error(status: int, name: str | None = None) -> web.Response:
    """A generic JSON error body (never details of the failure)."""
    return web.json_response({"error": name or ERROR_NAMES.get(status, "error")}, status=status)


def apply_headers(request: web.Request, response: web.StreamResponse) -> None:
    """Add the security headers; no caching for API and websocket responses."""
    for key, value in SECURITY_HEADERS.items():
        response.headers.setdefault(key, value)
    if request.path.startswith("/api/") or request.path in {"/login", "/logout", "/healthz"}:
        response.headers["Cache-Control"] = "no-store"


@web.middleware
async def errors_and_headers(request: web.Request, handler: Handler) -> web.StreamResponse:
    """Turn every exception into a generic JSON error and add security headers."""
    try:
        response = await handler(request)
    except web.HTTPException as exc:
        if exc.status < FIRST_ERROR_STATUS:  # redirects keep their Location header
            apply_headers(request, exc)
            raise
        response = error(exc.status)
    except Exception:
        log.exception("dashboard handler failed: %s %s", request.method, request.path)
        response = error(500, "internal_error")
    if not response.prepared:
        apply_headers(request, response)
    return response
