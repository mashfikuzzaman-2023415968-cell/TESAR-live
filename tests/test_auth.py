"""Tests for mdc.common.auth: HMAC construction, verification, encoding."""

from __future__ import annotations

import hashlib
import hmac
import struct

import pytest

from mdc.common import auth as A

SECRET = bytes(range(32))
NC = b"\x01" * 16
NW = b"\x02" * 16
WID = "w_0123456789abcdef"
FP = "sha256:" + "ab" * 32


def test_worker_auth_matches_spec_construction() -> None:
    def lp(*fields: bytes) -> bytes:
        return b"".join(struct.pack("!I", len(f)) + f for f in fields)

    expected = hmac.new(
        SECRET, lp(b"MDC-W", NC, NW, WID.encode(), FP.encode()), hashlib.sha256
    ).hexdigest()
    assert A.worker_auth(SECRET, NC, NW, WID, FP) == expected
    proof = hmac.new(
        SECRET, lp(b"MDC-C", NW, NC, WID.encode(), FP.encode()), hashlib.sha256
    ).hexdigest()
    assert A.coordinator_proof(SECRET, NC, NW, WID, FP) == proof


def test_correct_secret_passes() -> None:
    auth = A.worker_auth(SECRET, NC, NW, WID, FP)
    assert A.verify_worker_auth(SECRET, NC, NW, WID, FP, auth)
    proof = A.coordinator_proof(SECRET, NC, NW, WID, FP)
    assert A.verify_coordinator_proof(SECRET, NC, NW, WID, FP, proof)


def test_wrong_secret_fails() -> None:
    auth = A.worker_auth(SECRET, NC, NW, WID, FP)
    assert not A.verify_worker_auth(bytes(32), NC, NW, WID, FP, auth)


def test_wrong_worker_id_fails() -> None:
    auth = A.worker_auth(SECRET, NC, NW, WID, FP)
    assert not A.verify_worker_auth(SECRET, NC, NW, "w_other", FP, auth)


def test_swapped_nonces_fail() -> None:
    auth = A.worker_auth(SECRET, NC, NW, WID, FP)
    assert not A.verify_worker_auth(SECRET, NW, NC, WID, FP, auth)


def test_different_fingerprint_fails() -> None:
    auth = A.worker_auth(SECRET, NC, NW, WID, FP)
    assert not A.verify_worker_auth(SECRET, NC, NW, WID, "sha256:" + "cd" * 32, auth)


def test_domain_separation_proof_is_not_a_worker_auth() -> None:
    # Even with identical nonces on both sides, the labels differ.
    proof = A.coordinator_proof(SECRET, NC, NC, WID, FP)
    assert not A.verify_worker_auth(SECRET, NC, NC, WID, FP, proof)
    auth = A.worker_auth(SECRET, NC, NC, WID, FP)
    assert not A.verify_coordinator_proof(SECRET, NC, NC, WID, FP, auth)


def test_length_prefix_makes_encoding_unambiguous() -> None:
    assert A.length_prefixed(b"ab", b"c") != A.length_prefixed(b"a", b"bc")
    assert A.length_prefixed(b"", b"x") == b"\x00\x00\x00\x00\x00\x00\x00\x01x"
    a = A.worker_auth(SECRET, NC, NW, "w_ab", "sha256:" + "0" * 64)
    b = A.worker_auth(SECRET, NC, NW, "w_a", "bsha256:" + "0" * 64)
    assert a != b


@pytest.mark.parametrize(
    "received",
    [None, 123, b"00" * 32, "", "0" * 63, "0" * 65, "é" * 64, "Z" * 64],
)
def test_verify_rejects_malformed_values(received: object) -> None:
    assert not A.verify(A.worker_auth(SECRET, NC, NW, WID, FP), received)


def test_verify_is_case_sensitive_exact() -> None:
    auth = A.worker_auth(SECRET, NC, NW, WID, FP)
    assert not A.verify(auth, auth.upper())


@pytest.mark.parametrize(
    ("secret", "nc", "nw"),
    [
        (b"short", NC, NW),
        (SECRET + b"x", NC, NW),
        ("text" * 8, NC, NW),
        (SECRET, b"\x01" * 15, NW),
        (SECRET, NC, b"\x02" * 17),
        (SECRET, "nonce", NW),
    ],
)
def test_bad_inputs_raise(secret: object, nc: object, nw: object) -> None:
    with pytest.raises(ValueError):
        A.worker_auth(secret, nc, nw, WID, FP)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        A.coordinator_proof(secret, nc, nw, WID, FP)  # type: ignore[arg-type]


def test_nonces_and_secrets_are_fresh() -> None:
    nonces = {A.make_nonce() for _ in range(1000)}
    assert len(nonces) == 1000
    assert all(len(n) == 16 for n in nonces)
    assert len(A.make_secret()) == 32
    assert A.make_secret() != A.make_secret()


def test_nonce_from_hex() -> None:
    assert A.nonce_from_hex("01" * 16) == b"\x01" * 16


@pytest.mark.parametrize(
    "value", [None, "01" * 15, "01" * 17, "zz" * 16, "AB" * 16, "01 " * 10 + "01", 5]
)
def test_nonce_from_hex_rejects(value: object) -> None:
    with pytest.raises(ValueError):
        A.nonce_from_hex(value)
