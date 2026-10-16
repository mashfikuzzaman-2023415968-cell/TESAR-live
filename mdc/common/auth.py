"""Mutual HMAC-SHA256 challenge-response authentication (spec 6.4, S.3.5).

Both sides share a 32-byte **device secret** issued at enrollment. Inside the
TLS connection:

1. coordinator -> worker: ``CHALLENGE{nonce_c}``
2. worker -> coordinator: ``HELLO{nonce_w, auth}`` with
   ``auth  = HMAC(secret, "MDC-W" | nonce_c | nonce_w | worker_id | fingerprint)``
3. coordinator -> worker: ``WELCOME{proof}`` with
   ``proof = HMAC(secret, "MDC-C" | nonce_w | nonce_c | worker_id | fingerprint)``

Why each ingredient is there:

* **Fresh nonces from both sides** (``secrets.token_bytes``): every connection
  gets a new ``nonce_c``, so a recorded HELLO is useless later (no replay);
  the worker's ``nonce_w`` makes the coordinator's proof fresh too.
* **Domain-separation labels** ``MDC-W`` / ``MDC-C`` (``MDC-BEACON`` reserved):
  a value computed in one role can never be accepted in the other role, so an
  attacker cannot "reflect" a coordinator proof back as a worker auth.
* **Certificate fingerprint (channel binding)**: the MAC is tied to the exact
  TLS certificate the worker pinned. Even a man-in-the-middle with a
  different certificate cannot reuse a MAC across the two TLS sessions.
* **Unambiguous encoding**: every field is written as ``uint32_be(len) || bytes``.
  Plain concatenation would let ``("ab", "c")`` and ``("a", "bc")`` produce the
  same message.
* **Constant-time comparison** (``hmac.compare_digest``): an ``==`` comparison
  can stop at the first different byte, leaking through timing how many
  leading bytes of a guess were right.

No other cryptography is invented here: HMAC-SHA256 is a standard MAC.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import struct

NONCE_BYTES = 16
SECRET_BYTES = 32
MAC_HEX_LEN = 64

LABEL_WORKER = b"MDC-W"
LABEL_COORDINATOR = b"MDC-C"
LABEL_BEACON = b"MDC-BEACON"  # reserved for signed beacons; unused for now

# bytes.fromhex() tolerates whitespace, so check the exact format first.
_NONCE_HEX_RE = re.compile(r"^[0-9a-f]{32}$")


def make_nonce() -> bytes:
    """16 fresh random bytes from the OS CSPRNG."""
    return secrets.token_bytes(NONCE_BYTES)


def make_secret() -> bytes:
    """A new 32-byte device secret."""
    return secrets.token_bytes(SECRET_BYTES)


def length_prefixed(*fields: bytes) -> bytes:
    """Encode fields as ``uint32_be(len) || bytes`` each (unambiguous concatenation)."""
    return b"".join(struct.pack("!I", len(f)) + f for f in fields)


def _check_inputs(secret: bytes, nonce_c: bytes, nonce_w: bytes) -> None:
    if not isinstance(secret, bytes) or len(secret) != SECRET_BYTES:
        raise ValueError(f"device secret must be {SECRET_BYTES} bytes")
    for name, nonce in (("nonce_c", nonce_c), ("nonce_w", nonce_w)):
        if not isinstance(nonce, bytes) or len(nonce) != NONCE_BYTES:
            raise ValueError(f"{name} must be {NONCE_BYTES} bytes")


def _mac(secret: bytes, message: bytes) -> str:
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


def worker_auth(secret: bytes, nonce_c: bytes, nonce_w: bytes, worker_id: str, fp: str) -> str:
    """The worker's HELLO ``auth`` value (hex)."""
    _check_inputs(secret, nonce_c, nonce_w)
    message = length_prefixed(
        LABEL_WORKER, nonce_c, nonce_w, worker_id.encode("utf-8"), fp.encode("utf-8")
    )
    return _mac(secret, message)


def coordinator_proof(
    secret: bytes, nonce_c: bytes, nonce_w: bytes, worker_id: str, fp: str
) -> str:
    """The coordinator's WELCOME ``proof`` value (hex). Note the swapped nonce order."""
    _check_inputs(secret, nonce_c, nonce_w)
    message = length_prefixed(
        LABEL_COORDINATOR, nonce_w, nonce_c, worker_id.encode("utf-8"), fp.encode("utf-8")
    )
    return _mac(secret, message)


def verify(expected_hex: str, received: object) -> bool:
    """Constant-time comparison of an expected MAC with an untrusted value."""
    if not isinstance(received, str) or len(received) != MAC_HEX_LEN:
        return False
    try:
        received_bytes = received.encode("ascii")
    except UnicodeEncodeError:
        return False
    return hmac.compare_digest(expected_hex.encode("ascii"), received_bytes)


def verify_worker_auth(
    secret: bytes, nonce_c: bytes, nonce_w: bytes, worker_id: str, fp: str, auth: object
) -> bool:
    """Coordinator side: does the HELLO ``auth`` prove knowledge of ``secret``?"""
    return verify(worker_auth(secret, nonce_c, nonce_w, worker_id, fp), auth)


def verify_coordinator_proof(
    secret: bytes, nonce_c: bytes, nonce_w: bytes, worker_id: str, fp: str, proof: object
) -> bool:
    """Worker side: does the WELCOME ``proof`` show the coordinator knows ``secret``?"""
    return verify(coordinator_proof(secret, nonce_c, nonce_w, worker_id, fp), proof)


def nonce_from_hex(value: object) -> bytes:
    """Decode a 16-byte nonce sent as hex; raise ``ValueError`` if malformed."""
    if not isinstance(value, str) or not _NONCE_HEX_RE.fullmatch(value):
        raise ValueError("nonce must be 32 lower-case hex characters")
    return bytes.fromhex(value)
