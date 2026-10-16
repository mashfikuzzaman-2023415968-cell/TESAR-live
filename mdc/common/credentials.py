"""Join codes: how a worker device receives its credentials (spec S.4.3).

A join code is ``MDC1.`` + base64url(JSON) where the JSON is::

    {"v": 1, "host": "192.168.0.10", "port": 50010, "worker_id": "w_...",
     "name": "phone-a", "secret": "<base64url of 32 bytes>", "fp": "sha256:<hex>"}

It carries everything a worker needs: where the coordinator is, who the
worker is, its device secret, and the coordinator certificate fingerprint to
pin. It is therefore a **secret** and is handled accordingly:

* decoding is strict: at most 1 KB, exact field set, every field validated;
* join files are written with mode 0600 and a join file readable by other
  users is refused (with instructions to fix it);
* the secret is never shown in ``repr()`` or logs (see ``redact.py``);
* it is never passed as a command-line argument (visible in ``ps``) — only as
  a file path or the ``MDC_JOIN_CODE`` environment variable.

Standard library only: this module runs on phones.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from mdc.common.auth import SECRET_BYTES
from mdc.common.protocol import ProtocolError, strict_json_loads
from mdc.common.safe_names import is_valid_id, is_valid_name
from mdc.common.statedir import read_private_file, write_private_file
from mdc.common.tls import is_valid_fingerprint

JOIN_PREFIX = "MDC1."
JOIN_VERSION = 1
MAX_JOIN_CODE_BYTES = 1024
ENV_VAR = "MDC_JOIN_CODE"
MAX_PORT = 65535
_FIELDS = frozenset({"v", "host", "port", "worker_id", "name", "secret", "fp"})
# Host names, IPv4 and IPv6 literals; deliberately conservative.
_HOST_RE = re.compile(r"^[A-Za-z0-9.:-]{1,253}$")
_B64URL_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class CredentialError(ValueError):
    """A join code or join file is malformed, tampered with, or unsafe."""


@dataclass(frozen=True)
class JoinCode:
    """Decoded join code. ``secret`` is excluded from ``repr`` on purpose."""

    host: str
    port: int
    worker_id: str
    name: str
    secret: bytes = field(repr=False)
    fp: str


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(text: str) -> bytes:
    if not _B64URL_RE.fullmatch(text):
        raise CredentialError("join code contains characters outside base64url")
    try:
        return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except (binascii.Error, ValueError):
        raise CredentialError("join code is not valid base64url") from None


def validate_join_fields(
    host: object, port: object, worker_id: object, name: object, secret: object, fp: object
) -> None:
    """Raise :class:`CredentialError` if any join-code field is invalid."""
    if not isinstance(host, str) or not _HOST_RE.fullmatch(host):
        raise CredentialError("join code: invalid host")
    if type(port) is not int or not 1 <= port <= MAX_PORT:
        raise CredentialError("join code: port must be in 1..65535")
    if not is_valid_id(worker_id):
        raise CredentialError("join code: invalid worker_id")
    if not is_valid_name(name):
        raise CredentialError("join code: invalid name")
    if not isinstance(secret, bytes) or len(secret) != SECRET_BYTES:
        raise CredentialError(f"join code: secret must be {SECRET_BYTES} bytes")
    if not is_valid_fingerprint(fp):
        raise CredentialError("join code: invalid certificate fingerprint")


def encode_join_code(code: JoinCode) -> str:
    """Serialize a :class:`JoinCode` to its ``MDC1.`` text form."""
    validate_join_fields(code.host, code.port, code.worker_id, code.name, code.secret, code.fp)
    body = json.dumps(
        {
            "v": JOIN_VERSION,
            "host": code.host,
            "port": code.port,
            "worker_id": code.worker_id,
            "name": code.name,
            "secret": _b64url_encode(code.secret),
            "fp": code.fp,
        },
        separators=(",", ":"),
    )
    # Every field is length-bounded (host <= 253, name <= 32, id <= 64), so the
    # result is always well under MAX_JOIN_CODE_BYTES (about 710 bytes at most).
    return JOIN_PREFIX + _b64url_encode(body.encode("utf-8"))


def decode_join_code(text: object) -> JoinCode:
    """Parse and strictly validate an untrusted join code."""
    if not isinstance(text, str):
        raise CredentialError("join code must be text")
    stripped = text.strip()
    if len(stripped.encode("utf-8")) > MAX_JOIN_CODE_BYTES:
        raise CredentialError("join code longer than 1 KB")
    if not stripped.startswith(JOIN_PREFIX):
        raise CredentialError("join code must start with 'MDC1.'")
    raw = _b64url_decode(stripped[len(JOIN_PREFIX) :])
    try:
        data = strict_json_loads(raw)
    except ProtocolError:
        raise CredentialError("join code payload is not valid JSON") from None
    if not isinstance(data, dict):
        raise CredentialError("join code payload must be a JSON object")
    if set(data) != _FIELDS:
        raise CredentialError("join code has missing or unexpected fields")
    if type(data["v"]) is not int or data["v"] != JOIN_VERSION:
        raise CredentialError("unsupported join code version")
    secret_text = data["secret"]
    if not isinstance(secret_text, str):
        raise CredentialError("join code: secret must be base64url text")
    secret = _b64url_decode(secret_text)
    validate_join_fields(
        data["host"], data["port"], data["worker_id"], data["name"], secret, data["fp"]
    )
    return JoinCode(data["host"], data["port"], data["worker_id"], data["name"], secret, data["fp"])


def write_join_file(path: str | Path, code: JoinCode) -> Path:
    """Write a join file atomically with mode 0600."""
    return write_private_file(path, encode_join_code(code) + "\n")


def read_join_file(path: str | Path) -> JoinCode:
    """Read a join file, refusing it if other users can read it (POSIX)."""
    try:
        raw = read_private_file(path)
    except FileNotFoundError:
        raise CredentialError(f"join file {path} not found") from None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise CredentialError("join file is not UTF-8 text") from None
    return decode_join_code(text)


def join_code_from_env(var: str = ENV_VAR) -> JoinCode | None:
    """Decode the join code from an environment variable, or ``None`` if unset."""
    value = os.environ.get(var)
    if value is None or not value.strip():
        return None
    return decode_join_code(value)
