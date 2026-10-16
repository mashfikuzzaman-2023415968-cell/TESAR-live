"""Reference handshake (spec 6.4) used by the Phase-1 security tests.

The production versions live in ``mdc/coordinator/server.py`` (Phase 4) and
``mdc/worker/worker.py`` (Phase 3); this small one proves that the building
blocks (protocol + validation + auth + tls + enrollment) compose correctly.
"""

from __future__ import annotations

import asyncio

from mdc.common import auth as A
from mdc.common import protocol as P
from mdc.common.credentials import JoinCode
from mdc.common.validation import DEFAULT_TASK_LIMITS, validate_message
from mdc.coordinator.enrollment import DeviceRegistry, LookupStatus


async def coordinator_side(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    registry: DeviceRegistry,
    fp: str,
    proof_secret_override: bytes | None = None,
) -> str:
    """Run the coordinator half; return "welcome" or the REJECT reason."""
    lock = asyncio.Lock()
    nonce_c = A.make_nonce()
    await P.write_frame(
        writer, P.make_message(P.CHALLENGE, nonce=nonce_c.hex(), server_version="0.1.0"), lock
    )
    msg = await P.read_frame(reader)
    if msg.type != P.HELLO:
        await P.write_frame(writer, P.make_message(P.REJECT, reason="protocol"), lock)
        return "protocol"
    try:
        validate_message(msg)
    except P.ValidationError:
        await P.write_frame(writer, P.make_message(P.REJECT, reason="protocol"), lock)
        return "protocol"
    h = msg.header
    status, secret = registry.lookup(h["worker_id"])
    if status is LookupStatus.REVOKED:
        await P.write_frame(writer, P.make_message(P.REJECT, reason="revoked"), lock)
        return "revoked"
    nonce_w = A.nonce_from_hex(h["nonce_w"])
    if secret is None or not A.verify_worker_auth(
        secret, nonce_c, nonce_w, h["worker_id"], fp, h["auth"]
    ):
        await P.write_frame(writer, P.make_message(P.REJECT, reason="auth"), lock)
        return "auth"
    proof = A.coordinator_proof(
        proof_secret_override or secret, nonce_c, nonce_w, h["worker_id"], fp
    )
    await P.write_frame(
        writer,
        P.make_message(
            P.WELCOME,
            session_id="s_test",
            ping_interval_s=1.0,
            accepted_slots=1,
            proof=proof,
            limits=DEFAULT_TASK_LIMITS.to_dict(),
        ),
        lock,
    )
    return "welcome"


def hello_for(code: JoinCode, nonce_c: bytes, nonce_w: bytes, fp: str | None = None) -> P.Message:
    """A HELLO for ``code`` answering ``nonce_c``."""
    pinned = fp or code.fp
    return P.make_message(
        P.HELLO,
        worker_id=code.worker_id,
        name=code.name,
        platform="linux",
        device_class="laptop",
        cpu_count=4,
        slots=1,
        bench_ms_per_unit=10.0,
        battery={
            "pct": None,
            "charging": None,
            "current_ma": None,
            "voltage_v": None,
            "source": "mock",
            "ts_w": None,
        },
        version="0.1.0",
        trusted_local=False,
        nonce_w=nonce_w.hex(),
        auth=A.worker_auth(code.secret, nonce_c, nonce_w, code.worker_id, pinned),
    )


async def worker_side(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, code: JoinCode
) -> str:
    """Run the worker half; return "ok", "proof_failed" or "rejected:<reason>"."""
    lock = asyncio.Lock()
    challenge = validate_message(await P.read_frame(reader))
    nonce_c = A.nonce_from_hex(challenge.header["nonce"])
    nonce_w = A.make_nonce()
    await P.write_frame(writer, hello_for(code, nonce_c, nonce_w), lock)
    reply = validate_message(await P.read_frame(reader))
    if reply.type == P.REJECT:
        return f"rejected:{reply.header['reason']}"
    if not A.verify_coordinator_proof(
        code.secret, nonce_c, nonce_w, code.worker_id, code.fp, reply.header["proof"]
    ):
        return "proof_failed"
    return "ok"
