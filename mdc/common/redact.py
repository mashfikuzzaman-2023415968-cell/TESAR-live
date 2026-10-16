"""Secret redaction for logs, config dumps, reports and error messages (spec S.10.1).

Two independent nets, so a secret has to slip through both to leak:

1. **By key name** — in dicts, the value of any key that names a secret is
   replaced by ``"***"``. A key is secret-like if one of its components
   (split on ``_ - .``) is ``secret``, ``token``, ``password``, ``passphrase``
   or ``proof``; or if the key is exactly ``auth``; or if its *last* component
   is ``key`` (``private_key``, ``api_key``). Matching whole components rather
   than substrings keeps useful non-secret settings readable: ``auth_fail_limit``
   and ``key_path`` are not redacted (see DECISIONS.md).
2. **By value** — in any string, join codes (``MDC1.`` + base64url) and PEM
   private-key blocks are replaced, wherever they appear.

:class:`RedactingFilter` applies net 2 to a log record (message, arguments
and formatted exception text). ``logging_setup`` installs it on every handler
it creates **and** installs :func:`install_record_redaction`, which redacts
each record when it is created — so handlers added by anyone else (libraries,
test frameworks) also only ever see redacted records.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

MASK = "***"
_ALWAYS_SECRET_PARTS = frozenset({"secret", "token", "password", "passphrase", "proof"})
_EXACT_SECRET_KEYS = frozenset({"auth"})
_SPLIT_RE = re.compile(r"[_.\-]+")
# Built by concatenation so that this file does not itself look like a join code.
_JOIN_CODE_RE = re.compile("MDC1" + r"\.[A-Za-z0-9_=-]+")
_PEM_KEY_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(-----END [A-Z ]*PRIVATE KEY-----|$)",
    re.DOTALL,
)
_MAX_DEPTH = 32


def is_secret_key(key: object) -> bool:
    """True if a dict key names a secret value."""
    if not isinstance(key, str):
        return False
    lowered = key.lower()
    parts = [p for p in _SPLIT_RE.split(lowered) if p]
    if lowered in _EXACT_SECRET_KEYS:
        return True
    if any(p in _ALWAYS_SECRET_PARTS for p in parts):
        return True
    return bool(parts) and parts[-1] == "key"


def redact_text(text: str) -> str:
    """Mask join codes and PEM private keys inside free text."""
    text = _JOIN_CODE_RE.sub("MDC1.<redacted>", text)
    return _PEM_KEY_RE.sub("<redacted private key>", text)


def redact(value: Any, _depth: int = 0) -> Any:
    """Return a redacted deep copy of ``value`` (dicts, lists, tuples, strings)."""
    if _depth > _MAX_DEPTH:
        return MASK
    if isinstance(value, dict):
        return {
            k: (MASK if is_secret_key(k) and v is not None else redact(v, _depth + 1))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        items = [redact(v, _depth + 1) for v in value]
        return items if isinstance(value, list) else tuple(items)
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, (bytes, bytearray)):
        # Raw bytes in a structure to be dumped are never shown (they may be a secret).
        return MASK
    return value


def redact_record(record: logging.LogRecord) -> logging.LogRecord:
    """Redact a log record in place: merged message, exception and stack text."""
    try:
        message = record.getMessage()
    except Exception:  # a broken format string must not kill logging
        message = f"{record.msg!r} {record.args!r}"
    record.msg = redact_text(message)
    record.args = None
    if record.exc_info and not record.exc_text:
        record.exc_text = logging.Formatter().formatException(record.exc_info)
    if record.exc_text:
        record.exc_text = redact_text(record.exc_text)
    if record.stack_info:
        record.stack_info = redact_text(record.stack_info)
    return record


class RedactingFilter(logging.Filter):
    """Logging filter that redacts the final message and exception text."""

    def filter(self, record: logging.LogRecord) -> bool:
        redact_record(record)
        return True


_original_factory: Callable[..., logging.LogRecord] | None = None


def install_record_redaction() -> None:
    """Redact every log record at creation time (idempotent).

    Wraps the global ``LogRecordFactory``. ``logger.exception()`` passes
    ``exc_info`` to the factory, so exception text is redacted here too.
    """
    global _original_factory  # noqa: PLW0603 - one process-wide hook, by design
    if _original_factory is not None:
        return
    original = logging.getLogRecordFactory()

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        return redact_record(original(*args, **kwargs))

    _original_factory = original
    logging.setLogRecordFactory(factory)


def uninstall_record_redaction() -> None:
    """Restore the original ``LogRecordFactory`` (tests, teardown)."""
    global _original_factory  # noqa: PLW0603
    if _original_factory is not None:
        logging.setLogRecordFactory(_original_factory)
        _original_factory = None
