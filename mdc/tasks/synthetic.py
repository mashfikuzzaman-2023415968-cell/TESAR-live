"""Synthetic CPU task: a SHA-256 hash chain (spec 8.5).

For controlled experiments without images::

    h = sha256(seed as 8 big-endian bytes)
    repeat chain_len times: h = sha256(h)
    result = {"digest": h.hex()}

Hash chains are inherently sequential (each step needs the previous one),
deterministic, use no memory and cost exactly ``chain_len`` hashes — an ideal
knob for CPU load. ``work_units = chain_len / 1000``.

A synthetic job's final aggregation XORs all digests together
(:func:`xor_digests`).
"""

from __future__ import annotations

import hashlib

from mdc.common.validation import DEFAULT_TASK_LIMITS, TaskLimits, ValidationError
from mdc.tasks.errors import check_deadline

MAX_SEED = 2**63 - 1
DEADLINE_CHECK_EVERY = 10_000  # hashes between deadline checks (~10 ms on a phone)
DIGEST_BYTES = 32


def hash_chain(
    seed: int,
    chain_len: int,
    limits: TaskLimits = DEFAULT_TASK_LIMITS,
    deadline: float | None = None,
) -> str:
    """Return the hex digest at the end of the chain."""
    if type(seed) is not int or not 0 <= seed <= MAX_SEED:
        raise ValidationError(f"seed must be an int in 0..{MAX_SEED}")
    if type(chain_len) is not int or not 1 <= chain_len <= limits.chain_len:
        raise ValidationError(f"chain_len must be an int in 1..{limits.chain_len}")
    digest = hashlib.sha256(seed.to_bytes(8, "big")).digest()
    for i in range(chain_len):
        if i % DEADLINE_CHECK_EVERY == 0:
            check_deadline(deadline)
        digest = hashlib.sha256(digest).digest()
    return digest.hex()


def xor_digests(digests: list[str]) -> str:
    """XOR of hex digests (order-independent aggregation of a synthetic job)."""
    acc = bytearray(DIGEST_BYTES)
    for hex_digest in digests:
        raw = bytes.fromhex(hex_digest)
        if len(raw) != DIGEST_BYTES:
            raise ValidationError("digest must be 32 bytes")
        for i, byte in enumerate(raw):
            acc[i] ^= byte
    return acc.hex()
