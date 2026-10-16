"""TLS and certificate pinning (spec S.3.2-S.3.4), over real loopback sockets."""

from __future__ import annotations

import asyncio
import contextlib
import ssl
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from mdc.common import protocol as P
from mdc.common import tls
from mdc.coordinator.certs import CertInfo, generate_certificate

pytestmark = [pytest.mark.security, pytest.mark.integration]

OTHER_FP = "sha256:" + "00" * 32


@dataclass
class TLSServer:
    """A TLS echo-ish server that records what the handler saw."""

    port: int
    handled: int = 0
    received: list[bytes] = field(default_factory=list)


@contextlib.asynccontextmanager
async def tls_server(cert: CertInfo, reply: bool = True) -> AsyncIterator[TLSServer]:
    state = TLSServer(port=0)

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        state.handled += 1
        try:
            data = await asyncio.wait_for(reader.read(65536), 5)
            state.received.append(data)
            if reply and data:
                msg = P.decode_frame(data)
                await P.write_frame(
                    writer, P.make_message("BYE", reason=msg.header["reason"]), asyncio.Lock()
                )
        except (OSError, ssl.SSLError, asyncio.TimeoutError, P.WireError):
            pass
        finally:
            writer.close()
            with contextlib.suppress(OSError, ssl.SSLError):
                await writer.wait_closed()

    ctx = tls.server_context(cert.cert_path, cert.key_path)
    server = await asyncio.start_server(handle, "127.0.0.1", 0, ssl=ctx)
    state.port = server.sockets[0].getsockname()[1]
    try:
        yield state
    finally:
        server.close()
        await server.wait_closed()


async def test_pinned_connection_succeeds_and_carries_frames(tls_cert: CertInfo) -> None:
    async with tls_server(tls_cert) as srv:
        reader, writer = await tls.open_pinned_connection(
            "127.0.0.1", srv.port, tls_cert.fingerprint, timeout=5
        )
        try:
            ssl_obj = writer.get_extra_info("ssl_object")
            assert ssl_obj.version() in {"TLSv1.2", "TLSv1.3"}
            assert tls.peer_fingerprint(ssl_obj) == tls_cert.fingerprint
            await P.write_frame(writer, P.make_message("BYE", reason="hello"), asyncio.Lock())
            reply = await asyncio.wait_for(P.read_frame(reader), 5)
            assert reply.header["reason"] == "hello"
        finally:
            writer.close()
            await writer.wait_closed()


async def test_wrong_pin_aborts_before_application_data(tls_cert: CertInfo) -> None:
    async with tls_server(tls_cert) as srv:
        with pytest.raises(tls.PinMismatchError):
            await tls.open_pinned_connection("127.0.0.1", srv.port, OTHER_FP, timeout=5)
        for _ in range(100):  # let the server handler observe the close
            if srv.received:
                break
            await asyncio.sleep(0.02)
    assert srv.handled == 1
    assert srv.received in ([b""], [])  # EOF, not a single application byte


async def test_different_certificate_same_name_is_rejected(
    tls_cert: CertInfo, tmp_path: Path
) -> None:
    # An attacker with *another* self-signed cert (same CN) is still rejected.
    impostor = generate_certificate(tmp_path / "attacker", days=30)
    assert impostor.fingerprint != tls_cert.fingerprint
    async with tls_server(impostor) as srv:
        with pytest.raises(tls.PinMismatchError):
            await tls.open_pinned_connection("127.0.0.1", srv.port, tls_cert.fingerprint, 5)


async def test_plaintext_client_fails(tls_cert: CertInfo) -> None:
    async with tls_server(tls_cert) as srv:
        reader, writer = await asyncio.open_connection("127.0.0.1", srv.port)
        writer.write(P.encode_frame(P.make_message("BYE", reason="plain")))
        await writer.drain()
        try:
            data = await asyncio.wait_for(reader.read(1024), 5)
        except (ConnectionResetError, OSError):
            data = b""
        writer.close()
        with contextlib.suppress(OSError):
            await writer.wait_closed()
    # The server answered with (at most) a TLS alert and the handler never ran.
    assert srv.handled == 0
    assert b"MDC1" not in data


# The test deliberately selects deprecated TLS versions to prove they are refused.
@pytest.mark.filterwarnings("ignore:ssl.TLSVersion:DeprecationWarning")
@pytest.mark.parametrize(
    "configure",
    [
        lambda c: setattr(c, "maximum_version", ssl.TLSVersion.TLSv1_1),
        lambda c: setattr(c, "maximum_version", ssl.TLSVersion.TLSv1),
    ],
)
async def test_tls_below_1_2_refused(
    tls_cert: CertInfo, configure: Callable[[ssl.SSLContext], None]
) -> None:
    async with tls_server(tls_cert) as srv:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.minimum_version = ssl.TLSVersion.MINIMUM_SUPPORTED
        with contextlib.suppress(ValueError, ssl.SSLError):
            configure(ctx)
        with pytest.raises((ssl.SSLError, ConnectionError, OSError)):
            await tls.open_pinned_connection(
                "127.0.0.1", srv.port, tls_cert.fingerprint, 5, ssl_context=ctx
            )
    assert srv.handled == 0


async def test_handshake_timeout() -> None:
    # A server that accepts TCP but never speaks TLS (slowloris-like).
    handler_done = asyncio.Event()

    async def silent(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await reader.read()  # never answer; return once the client gives up
        finally:
            # Close explicitly: on Python 3.10 Server.wait_closed() does not wait
            # for handlers, so a cancelled handler would leak this socket.
            writer.close()
            handler_done.set()

    server = await asyncio.start_server(silent, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        with pytest.raises(asyncio.TimeoutError):
            await tls.open_pinned_connection("127.0.0.1", port, OTHER_FP, timeout=0.3)
        await asyncio.wait_for(handler_done.wait(), 5)
    finally:
        server.close()
        await server.wait_closed()


async def test_invalid_pin_format_refused_before_connecting() -> None:
    with pytest.raises(ValueError):
        await tls.open_pinned_connection("127.0.0.1", 9, "md5:abc", timeout=1)


def test_contexts_are_hardened(tls_cert: CertInfo) -> None:
    server = tls.server_context(tls_cert.cert_path, tls_cert.key_path)
    client = tls.client_context()
    for ctx in (server, client):
        assert ctx.minimum_version >= ssl.TLSVersion.TLSv1_2
        assert ctx.options & ssl.OP_NO_COMPRESSION
    # Trust comes from the pin, so CA checks are off — only in this module.
    assert client.verify_mode == ssl.CERT_NONE
    assert client.check_hostname is False


def test_check_pin_requires_tls_and_valid_pin() -> None:
    with pytest.raises(tls.PinMismatchError, match="not using TLS"):
        tls.check_pin(None, OTHER_FP)
    with pytest.raises(ValueError):
        tls.check_pin(None, "sha256:XYZ")

    class NoCert:
        def getpeercert(self, binary_form: bool = False) -> None:
            return None

    with pytest.raises(tls.PinMismatchError, match="no certificate"):
        tls.check_pin(NoCert(), OTHER_FP)  # type: ignore[arg-type]


def test_fingerprint_helpers(tls_cert: CertInfo) -> None:
    pem = tls_cert.cert_path.read_text()
    der = ssl.PEM_cert_to_DER_cert(pem)
    assert tls.fingerprint_pem(pem) == tls.fingerprint_der(der) == tls_cert.fingerprint
    assert tls.fingerprint_file(tls_cert.cert_path) == tls_cert.fingerprint
    assert tls.is_valid_fingerprint(tls_cert.fingerprint)
    for bad in ["", "sha256:", "sha256:" + "A" * 64, "sha1:" + "a" * 64, None, 5]:
        assert not tls.is_valid_fingerprint(bad)
    assert tls.fingerprints_equal(OTHER_FP, OTHER_FP)
    assert not tls.fingerprints_equal(OTHER_FP, tls_cert.fingerprint)
