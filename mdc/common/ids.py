"""Identifier generation.

Every id that appears on the wire (session ids, attempt ids, job ids) is built
from ``secrets.token_hex`` — never the ``random`` module — because predictable
ids would let an attacker guess or forge them (e.g. an attempt id to inject a
fake result).

The ids always match the protocol's id regex ``^[A-Za-z0-9_-]{1,64}$``.

Note: a worker's persistent identity is the ``worker_id`` issued at enrollment
and stored in its join code (Phase 1, spec S.4) — never a self-chosen id.
"""

from __future__ import annotations

import re
import secrets

ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_PREFIX_RE = re.compile(r"^[A-Za-z0-9-]{1,24}$")
MIN_ID_BYTES = 4
MAX_ID_BYTES = 16


def new_id(prefix: str, nbytes: int = 8) -> str:
    """Return ``"<prefix>_<random hex>"``, e.g. ``"att_3f9a0c1d2e4b5a69"``.

    Args:
        prefix: short label saying what kind of id this is (letters, digits, ``-``).
        nbytes: random bytes (hex length is twice this). 8 bytes = 64 bits,
            plenty to make collisions within a run practically impossible.
    """
    if not _PREFIX_RE.match(prefix):
        raise ValueError(f"invalid id prefix {prefix!r}")
    if not (MIN_ID_BYTES <= nbytes <= MAX_ID_BYTES):
        raise ValueError("nbytes must be in 4..16")
    return f"{prefix}_{secrets.token_hex(nbytes)}"


def is_valid_id(value: object) -> bool:
    """True if ``value`` is a string matching the protocol id regex."""
    return isinstance(value, str) and ID_RE.fullmatch(value) is not None
