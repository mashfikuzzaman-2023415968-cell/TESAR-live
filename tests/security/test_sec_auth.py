"""Mutual authentication (spec 6.4, S.3.5, S.4): replay, reflection, channel
binding, unknown/revoked devices, rogue coordinators — over real TLS."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from pathlib import Path

import numpy as np
import pytest

from mdc.common import auth as A
from mdc.common import protocol as P
from mdc.common import tls
from mdc.common.credentials import JoinCode, read_join_file
from mdc.coordinator.certs import CertInfo
from mdc.coordinator.coordinator import Coordinator
from mdc.coordinator.enrollment import DeviceRegistry
from mdc.coordinator.job_builder import build_image_job_from_array
from mdc.coordinator.security_log import read_security_log
from mdc.coordinator.server import is_loopback, trusted_local_decision
from mdc.tasks.image_pipeline import reference
from tests.helpers.handshake import coordinator_side, hello_for, worker_side
from tests.helpers.raw_worker import connect_raw

pytestmark = [pytest.mark.security, pytest.mark.integration]


async def run_handshake(
    cert: CertInfo,
    registry: DeviceRegistry,
    code: JoinCode,
    proof_secret_override: bytes | None = None,
) -> tuple[str, str]:
    """Coordinator + worker over TLS loopback; return (coordinator result, worker result)."""
    results: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            results.set_result(
                await coordinator_side(
                    reader, writer, registry, cert.fingerprint, proof_secret_override
                )
            )
        finally:
            writer.close()
            with contextlib.suppress(OSError):
                await writer.wait_closed()

    server = await asyncio.start_server(
        handle, "127.0.0.1", 0, ssl=tls.server_context(cert.cert_path, cert.key_path)
    )
    port = server.sockets[0].getsockname()[1]
    try:
        reader, writer = await tls.open_pinned_connection("127.0.0.1", port, code.fp, 5)
        try:
            worker_result = await asyncio.wait_for(worker_side(reader, writer, code), 5)
        finally:
            writer.close()
            with contextlib.suppress(OSError):
                await writer.wait_closed()
        coord_result = await asyncio.wait_for(results, 5)
    finally:
        server.close()
        await server.wait_closed()
    return coord_result, worker_result


async def test_enrolled_device_authenticates_mutually(
    tls_cert: CertInfo, registry: DeviceRegistry, enrolled: Callable[..., Path]
) -> None:
    code = read_join_file(enrolled("phone-a"))
    assert await run_handshake(tls_cert, registry, code) == ("welcome", "ok")


async def test_unknown_device_rejected(tls_cert: CertInfo, registry: DeviceRegistry) -> None:
    code = JoinCode("127.0.0.1", 1, "w_unknown", "ghost", A.make_secret(), tls_cert.fingerprint)
    assert await run_handshake(tls_cert, registry, code) == ("auth", "rejected:auth")


async def test_revoked_device_rejected(
    tls_cert: CertInfo, registry: DeviceRegistry, enrolled: Callable[..., Path]
) -> None:
    code = read_join_file(enrolled("phone-b"))
    registry.revoke("phone-b")
    assert await run_handshake(tls_cert, registry, code) == ("revoked", "rejected:revoked")


async def test_wrong_secret_rejected(
    tls_cert: CertInfo, registry: DeviceRegistry, enrolled: Callable[..., Path]
) -> None:
    code = read_join_file(enrolled("phone-c"))
    forged = JoinCode(code.host, code.port, code.worker_id, code.name, A.make_secret(), code.fp)
    assert await run_handshake(tls_cert, registry, forged) == ("auth", "rejected:auth")


async def test_rotated_secret_old_code_rejected(
    tls_cert: CertInfo, registry: DeviceRegistry, enrolled: Callable[..., Path]
) -> None:
    old = read_join_file(enrolled("phone-d"))
    new = registry.rotate_secret("phone-d", old.host, old.port, tls_cert.fingerprint)
    assert await run_handshake(tls_cert, registry, old) == ("auth", "rejected:auth")
    assert await run_handshake(tls_cert, registry, new) == ("welcome", "ok")


async def test_rogue_coordinator_without_secret_fails_proof(
    tls_cert: CertInfo, registry: DeviceRegistry, enrolled: Callable[..., Path]
) -> None:
    # Even with the right certificate (e.g. a stolen key), a coordinator that
    # does not know the device secret cannot produce a valid WELCOME proof.
    code = read_join_file(enrolled("phone-e"))
    result = await run_handshake(tls_cert, registry, code, proof_secret_override=A.make_secret())
    assert result == ("welcome", "proof_failed")


def test_replayed_hello_rejected() -> None:
    secret = A.make_secret()
    code = JoinCode("h", 1, "w_1", "n", secret, "sha256:" + "a" * 64)
    nonce_c1, nonce_c2, nonce_w = A.make_nonce(), A.make_nonce(), A.make_nonce()
    recorded = hello_for(code, nonce_c1, nonce_w).header["auth"]
    # The attacker replays the recorded HELLO on a new connection (new nonce_c).
    assert not A.verify_worker_auth(secret, nonce_c2, nonce_w, "w_1", code.fp, recorded)
    assert A.verify_worker_auth(secret, nonce_c1, nonce_w, "w_1", code.fp, recorded)


def test_reflected_proof_rejected() -> None:
    secret = A.make_secret()
    nonce_c, nonce_w = A.make_nonce(), A.make_nonce()
    fp = "sha256:" + "b" * 64
    # An attacker posing as a worker obtains a coordinator proof and sends it back as HELLO auth,
    # even choosing nonces so that the field order lines up.
    proof = A.coordinator_proof(secret, nonce_w, nonce_c, "w_1", fp)
    assert not A.verify_worker_auth(secret, nonce_c, nonce_w, "w_1", fp, proof)


def test_wrong_fingerprint_rejected() -> None:
    # Channel binding: a MITM terminating TLS with its own certificate makes the
    # worker MAC over the MITM's fingerprint, which the real coordinator rejects.
    secret = A.make_secret()
    nonce_c, nonce_w = A.make_nonce(), A.make_nonce()
    mitm_fp, real_fp = "sha256:" + "c" * 64, "sha256:" + "d" * 64
    auth = A.worker_auth(secret, nonce_c, nonce_w, "w_1", mitm_fp)
    assert not A.verify_worker_auth(secret, nonce_c, nonce_w, "w_1", real_fp, auth)


async def test_non_hello_first_message_is_protocol_reject(
    tls_cert: CertInfo, registry: DeviceRegistry
) -> None:
    results: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            results.set_result(
                await coordinator_side(reader, writer, registry, tls_cert.fingerprint)
            )
        finally:
            writer.close()

    server = await asyncio.start_server(
        handle, "127.0.0.1", 0, ssl=tls.server_context(tls_cert.cert_path, tls_cert.key_path)
    )
    port = server.sockets[0].getsockname()[1]
    reader, writer = await tls.open_pinned_connection("127.0.0.1", port, tls_cert.fingerprint, 5)
    await P.read_frame(reader)  # CHALLENGE
    await P.write_frame(writer, P.make_message(P.PING, seq=1, t_c=0.0), asyncio.Lock())
    reply = await asyncio.wait_for(P.read_frame(reader), 5)
    writer.close()
    with contextlib.suppress(OSError):
        await writer.wait_closed()
    server.close()
    await server.wait_closed()
    assert reply.type == P.REJECT
    assert reply.header["reason"] == "protocol"
    assert await results == "protocol"


# ------------------------------------------------ live coordinator (Phase 4)

StartCoordinator = Callable[..., Awaitable[Coordinator]]


def _sec(coord: Coordinator, event_type: str) -> list[dict]:
    path = Path(coord.cfg.security.state_dir) / "logs" / "security.jsonl"
    return [r for r in read_security_log(path) if r["type"] == event_type]


async def _wait(cond: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not cond():
        assert asyncio.get_running_loop().time() < deadline, "condition not reached"
        await asyncio.sleep(0.05)


async def test_result_from_wrong_worker_rejected(
    coordinator: StartCoordinator, enrolled: Callable[..., Path]
) -> None:
    coord = await coordinator()
    owner = await connect_raw(read_join_file(enrolled("owner")), coord.port)
    image = np.random.default_rng(1).integers(0, 256, size=(40, 40), dtype=np.uint8)
    job = build_image_job_from_array(image, 40, 1, 1, 64)  # a single tile
    job_task = asyncio.create_task(coord.run_job(job, timeout_s=20))
    assign = await owner.recv(P.TASK_ASSIGN)
    thief_code = read_join_file(enrolled("thief"))
    thief = await connect_raw(thief_code, coord.port)
    fields, payload = P.pack_array(np.zeros((40, 40), dtype=np.uint8))
    await thief.send(
        P.make_message(
            P.TASK_RESULT,
            payload,
            task_id=assign.header["task_id"],
            attempt_id=assign.header["attempt_id"],
            status="ok",
            error=None,
            compute_ms=1.0,
            queue_ms=0.0,
            t_recv_w=0.0,
            t_send_w=0.0,
            **fields,
        )
    )
    await _wait(lambda: bool(_sec(coord, "result_from_wrong_worker")))
    assert coord.registry.get(thief_code.worker_id).failed == 1
    assert job.tasks[assign.header["task_id"]].result is None  # nothing was accepted
    # The rightful owner's result is still accepted afterwards.
    fields, payload = P.pack_array(reference(image, 1, 1))
    await owner.send(
        P.make_message(
            P.TASK_RESULT,
            payload,
            task_id=assign.header["task_id"],
            attempt_id=assign.header["attempt_id"],
            status="ok",
            error=None,
            compute_ms=1.0,
            queue_ms=0.0,
            t_recv_w=0.0,
            t_send_w=0.0,
            **fields,
        )
    )
    await _wait(lambda: job.tasks[assign.header["task_id"]].result is not None)
    job_task.cancel()
    with contextlib.suppress(BaseException):
        await job_task
    await owner.close()
    await thief.close()


async def test_enrolled_device_cannot_claim_trusted_local(
    coordinator: StartCoordinator, enrolled: Callable[..., Path]
) -> None:
    coord = await coordinator()
    raw = await connect_raw(read_join_file(enrolled("pretender")), coord.port, trusted_local=True)
    assert raw.reply.type == P.REJECT and raw.reply.header["reason"] == "auth"
    spoofs = _sec(coord, "trusted_local_spoof")
    assert spoofs and "enrolled credential" in spoofs[-1]["detail"]
    await raw.close()


async def test_local_credential_must_claim_trusted(coordinator: StartCoordinator) -> None:
    coord = await coordinator(local_worker=True)
    assert coord.local_credential is not None
    raw = await connect_raw(coord.local_credential.join, coord.port, trusted_local=False)
    assert raw.reply.type == P.REJECT
    assert _sec(coord, "trusted_local_spoof")
    await raw.close()


@pytest.mark.parametrize(
    ("uses_local", "claims", "ip", "verdict"),
    [
        (True, True, "127.0.0.1", None),
        (True, True, "::1", None),
        (True, True, "::ffff:127.0.0.1", None),
        (True, True, "192.168.1.5", "non-loopback"),
        (False, True, "127.0.0.1", "enrolled credential"),
        (True, False, "127.0.0.1", "without trusted_local"),
        (False, False, "192.168.1.5", None),
    ],
)
def test_trusted_local_rules(uses_local: bool, claims: bool, ip: str, verdict: str | None) -> None:
    result = trusted_local_decision(uses_local, claims, ip)
    assert result is None if verdict is None else verdict in (result or "")


def test_is_loopback() -> None:
    assert is_loopback("127.0.0.5") and is_loopback("::1")
    assert not is_loopback("10.0.0.1") and not is_loopback("garbage")


async def test_live_revocation_closes_session(
    coordinator: StartCoordinator, enrolled: Callable[..., Path], tmp_state_dir: Path
) -> None:
    coord = await coordinator(revocation_check_s=0.1)
    raw = await connect_raw(read_join_file(enrolled("lost-phone")), coord.port)
    assert raw.welcomed
    DeviceRegistry(tmp_state_dir).revoke("lost-phone")  # e.g. `mdc.cli revoke` in a terminal
    assert await raw.closed(5)
    closed = _sec(coord, "device_revoked_session_closed")
    assert closed and closed[-1]["status"] == "revoked"
    again = await connect_raw(read_join_file(enrolled("lost-phone-2")), coord.port)
    assert again.welcomed  # others unaffected
    await again.close()


async def test_secret_rotation_closes_old_session(
    coordinator: StartCoordinator,
    enrolled: Callable[..., Path],
    tmp_state_dir: Path,
    tls_cert: CertInfo,
) -> None:
    coord = await coordinator(revocation_check_s=0.1)
    old = read_join_file(enrolled("rekeyed"))
    raw = await connect_raw(old, coord.port)
    new = DeviceRegistry(tmp_state_dir).rotate_secret(
        "rekeyed", old.host, old.port, tls_cert.fingerprint
    )
    assert await raw.closed(5)
    fresh = await connect_raw(new, coord.port)
    assert fresh.welcomed
    await fresh.close()


async def test_live_coordinator_rejects_unknown_and_revoked(
    coordinator: StartCoordinator,
    enrolled: Callable[..., Path],
    tmp_state_dir: Path,
) -> None:
    coord = await coordinator()
    code = read_join_file(enrolled("gone"))
    DeviceRegistry(tmp_state_dir).revoke("gone")
    raw = await connect_raw(code, coord.port)
    assert raw.reply.header["reason"] == "revoked"
    await raw.close()
    ghost = JoinCode(code.host, code.port, "w_ghost", "ghost", A.make_secret(), code.fp)
    raw = await connect_raw(ghost, coord.port)
    assert raw.reply.header["reason"] == "auth"
    await raw.close()
    reasons = [r["reason"] for r in _sec(coord, "auth_failed")]
    assert reasons == ["revoked", "auth"]


async def test_device_enrolled_after_start_is_accepted_at_once(
    coordinator: StartCoordinator, enrolled: Callable[..., Path]
) -> None:
    coord = await coordinator(revocation_check_s=60)  # no periodic reload to help
    raw = await connect_raw(read_join_file(enrolled("newcomer")), coord.port)
    assert raw.welcomed
    await raw.close()
