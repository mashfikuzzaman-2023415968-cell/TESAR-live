"""Tests for the SHA-256 hash-chain task and XOR aggregation (spec 8.5)."""

from __future__ import annotations

import hashlib
import time

import pytest

from mdc.common.validation import TaskLimits, ValidationError
from mdc.tasks.errors import TaskTimeLimitError
from mdc.tasks.synthetic import hash_chain, xor_digests


def by_definition(seed: int, chain_len: int) -> str:
    h = hashlib.sha256(seed.to_bytes(8, "big")).digest()
    for _ in range(chain_len):
        h = hashlib.sha256(h).digest()
    return h.hex()


def test_known_digest_for_tiny_chain() -> None:
    # sha256(sha256(8 zero bytes)), computed independently with a hashlib one-liner.
    assert hash_chain(0, 1) == ("7ef0ca626bbb058dd443bb78e33b888bdec8295c96e51f5545f96370870c10b9")


@pytest.mark.parametrize(("seed", "chain_len"), [(0, 1), (1, 1), (42, 10), (2**63 - 1, 5)])
def test_matches_definition(seed: int, chain_len: int) -> None:
    assert hash_chain(seed, chain_len) == by_definition(seed, chain_len)


def test_deterministic_and_seed_sensitive() -> None:
    assert hash_chain(7, 100) == hash_chain(7, 100)
    assert hash_chain(7, 100) != hash_chain(8, 100)
    assert hash_chain(7, 100) != hash_chain(7, 101)


@pytest.mark.parametrize(
    ("seed", "chain_len"),
    [(-1, 1), (2**63, 1), (True, 1), (1.0, 1), (1, 0), (1, 5_000_001), (1, "10")],
)
def test_rejects_bad_params(seed: object, chain_len: object) -> None:
    with pytest.raises(ValidationError):
        hash_chain(seed, chain_len)  # type: ignore[arg-type]


def test_custom_chain_limit() -> None:
    with pytest.raises(ValidationError):
        hash_chain(1, 11, TaskLimits(chain_len=10))
    hash_chain(1, 10, TaskLimits(chain_len=10))


def test_deadline() -> None:
    with pytest.raises(TaskTimeLimitError):
        hash_chain(1, 50_000, deadline=time.monotonic() - 1)


def test_xor_digests() -> None:
    a, b = hash_chain(1, 3), hash_chain(2, 3)
    assert xor_digests([a, a]) == "0" * 64
    assert xor_digests([a]) == a
    assert xor_digests([a, b]) == xor_digests([b, a])  # order independent
    assert xor_digests([]) == "0" * 64
    with pytest.raises(ValidationError):
        xor_digests(["abcd"])
