"""TLS contexts and certificate pinning (spec S.3.3, S.3.4).

Trust model: the coordinator uses a **self-signed** certificate (there is no
certificate authority on a lab LAN). Each worker's join code contains the
SHA-256 fingerprint of that exact certificate. After the TLS handshake the
worker hashes the certificate the server actually presented and compares it
with the pinned fingerprint; on mismatch it closes the connection before
sending a single application byte.

Why disabling CA verification is safe *here*: normal TLS asks "was this
certificate signed by any of ~150 trusted CAs for this host name?". Pinning
asks the much stricter question "is this exactly the one certificate I was
given out of band?". Any other certificate — even one validly signed by a
real CA — is rejected. So ``CERT_NONE`` plus an exact pin check is stronger
than CA validation, as long as the pin check is never skipped. This module is
the only place allowed to turn CA verification off (enforced by
``scripts/check_forbidden.py``), and it never returns a connection without
first checking the pin.

Standard library only: this module runs on phones.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import re
import ssl
from pathlib import Path

FINGERPRINT_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class PinMismatchError(ssl.SSLError):
    """The server's certificate is not the pinned one (possible MITM)."""


def fingerprint_der(der: bytes) -> str:
    """``sha256:<hex>`` fingerprint of a DER-encoded certificate."""
    return "sha256:" + hashlib.sha256(der).hexdigest()


def fingerprint_pem(pem_text: str) -> str:
    """Fingerprint of a PEM certificate (stdlib conversion PEM -> DER)."""
    return fingerprint_der(ssl.PEM_cert_to_DER_cert(pem_text))


def fingerprint_file(cert_path: str | Path) -> str:
    """Fingerprint of a PEM certificate file."""
    return fingerprint_pem(Path(cert_path).read_text(encoding="ascii"))


def is_valid_fingerprint(value: object) -> bool:
    """True for strings of the form ``sha256:`` + 64 lower-case hex characters."""
    return isinstance(value, str) and FINGERPRINT_RE.fullmatch(value) is not None


def fingerprints_equal(a: str, b: str) -> bool:
    """Constant-time equality of two fingerprints."""
    return hmac.compare_digest(a.encode("ascii"), b.encode("ascii"))


def _harden(ctx: ssl.SSLContext) -> ssl.SSLContext:
    # TLS 1.0/1.1 are deprecated and weak; TLS compression enables CRIME-style
    # attacks. Cipher suites are left at the library's secure defaults.
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.options |= ssl.OP_NO_COMPRESSION
    return ctx


def server_context(cert_path: str | Path, key_path: str | Path) -> ssl.SSLContext:
    """TLS server context for the coordinator."""
    ctx = _harden(ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER))
    ctx.load_cert_chain(str(cert_path), str(key_path))
    return ctx


def client_context() -> ssl.SSLContext:
    """TLS client context for workers: no CA check — trust comes from pinning.

    Never use this context without calling :func:`check_pin` after the
    handshake; :func:`open_pinned_connection` does both.
    """
    ctx = _harden(ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT))
    ctx.check_hostname = False  # forbidden-ok: pinning replaces hostname checks
    ctx.verify_mode = ssl.CERT_NONE  # forbidden-ok: exact-certificate pinning instead of CAs
    return ctx


def peer_fingerprint(ssl_object: ssl.SSLObject | ssl.SSLSocket | None) -> str:
    """Fingerprint of the certificate the peer presented in this TLS session."""
    if ssl_object is None:
        raise PinMismatchError("connection is not using TLS")
    der = ssl_object.getpeercert(binary_form=True)
    if not der:
        raise PinMismatchError("server presented no certificate")
    return fingerprint_der(der)


def check_pin(ssl_object: ssl.SSLObject | ssl.SSLSocket | None, expected_fp: str) -> str:
    """Raise :class:`PinMismatchError` unless the peer certificate matches ``expected_fp``.

    Returns the (verified) fingerprint.
    """
    if not is_valid_fingerprint(expected_fp):
        raise ValueError("pinned fingerprint must look like sha256:<64 hex>")
    actual = peer_fingerprint(ssl_object)
    if not fingerprints_equal(actual, expected_fp):
        raise PinMismatchError("server certificate does not match the pinned fingerprint")
    return actual


async def open_pinned_connection(
    host: str,
    port: int,
    expected_fp: str,
    timeout: float,
    ssl_context: ssl.SSLContext | None = None,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Connect with TLS and verify the certificate pin before returning.

    The whole TCP connect + TLS handshake must finish within ``timeout``
    seconds. On a pin mismatch the connection is closed immediately — no
    application data has been sent — and :class:`PinMismatchError` is raised.
    """
    if not is_valid_fingerprint(expected_fp):
        raise ValueError("pinned fingerprint must look like sha256:<64 hex>")
    ctx = ssl_context if ssl_context is not None else client_context()
    reader, writer = await asyncio.wait_for(
        # server_hostname="" disables SNI/hostname use: identity comes from the pin.
        asyncio.open_connection(host, port, ssl=ctx, server_hostname=""),
        timeout=timeout,
    )
    try:
        check_pin(writer.get_extra_info("ssl_object"), expected_fp)
    except BaseException:
        writer.close()
        with contextlib.suppress(OSError, ssl.SSLError, asyncio.TimeoutError):
            await asyncio.wait_for(writer.wait_closed(), timeout=timeout)
        raise
    return reader, writer
