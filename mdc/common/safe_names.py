"""Validation of names/ids and sanitising of peer-supplied strings (spec S.4.6, S.10.2).

* Device names (``phone-a``) are chosen by the operator at enrollment and must
  match ``^[A-Za-z0-9_.-]{1,32}$`` — no spaces, no control characters, no
  HTML, nothing that could confuse a log line, a file name or the dashboard.
* Protocol ids (task, attempt, session, worker ids) match ``^[A-Za-z0-9_-]{1,64}$``.
* Any other string that came from the network is passed through
  :func:`sanitize_for_log` before it is logged: control and format characters
  are removed (no fake log lines via ``\\n``, no terminal escape sequences, no
  bidi-override tricks) and the length is capped.
"""

from __future__ import annotations

import re
import unicodedata

from mdc.common.ids import ID_RE

NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,32}$")
MAX_LOG_STRING = 256


def is_valid_name(value: object) -> bool:
    """True if ``value`` is a valid device name."""
    return (
        isinstance(value, str)
        and NAME_RE.fullmatch(value) is not None
        and value
        not in {
            ".",
            "..",
        }
    )


def validate_name(value: object) -> str:
    """Return ``value`` if it is a valid device name, else raise ``ValueError``."""
    if not is_valid_name(value):
        raise ValueError("device name must match ^[A-Za-z0-9_.-]{1,32}$ (and not be . or ..)")
    return str(value)


def is_valid_id(value: object) -> bool:
    """True if ``value`` matches the protocol id regex."""
    return isinstance(value, str) and ID_RE.fullmatch(value) is not None


def validate_id(value: object, what: str = "id") -> str:
    """Return ``value`` if it is a valid protocol id, else raise ``ValueError``."""
    if not is_valid_id(value):
        raise ValueError(f"{what} must match ^[A-Za-z0-9_-]{{1,64}}$")
    return str(value)


def sanitize_for_log(value: object, max_len: int = MAX_LOG_STRING) -> str:
    """Make an untrusted value safe to put in a log line.

    Removes Unicode control (Cc) and format (Cf) characters — this covers
    newlines, ANSI escapes and right-to-left overrides — and truncates.
    """
    text = value if isinstance(value, str) else repr(value)
    cleaned = "".join(ch for ch in text if unicodedata.category(ch) not in {"Cc", "Cf"})
    if len(cleaned) > max_len:
        return cleaned[:max_len] + "...[truncated]"
    return cleaned
