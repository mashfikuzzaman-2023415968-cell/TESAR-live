"""Wire protocol: framing, message types and array packing (spec Section 6).

Every message on a coordinator<->worker TCP connection is one **frame**::

    +--------+------------+-------------+---------------------+-------------------+
    | magic  | header_len | payload_len | header (UTF-8 JSON) | payload (bytes)   |
    | "MDC1" | uint32 BE  | uint32 BE   | header_len bytes    | payload_len bytes |
    +--------+------------+-------------+---------------------+-------------------+

Why this design?

* TCP is a byte *stream*: it has no message boundaries. A length prefix tells
  the reader exactly how many bytes belong to the current message, so two
  messages can never be mixed up or split wrongly.
* The JSON header is human-readable (easy to debug with logs), while bulk data
  (image tiles) travels as raw bytes in the payload without base64 overhead.
* The fixed 12-byte prefix lets the reader check the magic and both lengths
  **before** reading (and allocating memory for) the body. A peer that claims
  a 4 GB header is rejected after 12 bytes, never after 4 GB.

Errors (all subclasses of :class:`WireError`):

* :class:`ProtocolError`   — malformed frame or header; the connection is closed.
* :class:`ValidationError` — a well-formed frame whose content breaks a schema
  or a limit (a kind of ProtocolError).
* :class:`IntegrityError`  — the payload's SHA-256 does not match the header:
  data was corrupted in transit. This is *transport integrity*, deliberately
  separate from *computational correctness* (Phase 10).
* :class:`ConnectionClosed` — the peer closed the connection cleanly between frames.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import re
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

MAGIC = b"MDC1"
PROTOCOL_VERSION = 1
PREFIX = struct.Struct("!4sII")  # magic, header_len, payload_len (network byte order)
PREFIX_LEN = PREFIX.size  # 12 bytes

DEFAULT_MAX_HEADER_BYTES = 16384
DEFAULT_MAX_PAYLOAD_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_RAW_ARRAY_BYTES = 64 * 1024 * 1024

MAX_JSON_DEPTH = 8  # spec S.5.2
MAX_INT_DIGITS = 20  # enough for any 64-bit value; blocks huge-int parsing cost
MAX_ARRAY_NDIM = 3  # spec S.5.4
MAX_ARRAY_DIM = 16384
SHA256_HEX_LEN = 64

# Message types (spec 6.2). Direction: C = coordinator, W = worker.
CHALLENGE = "CHALLENGE"  # C->W
HELLO = "HELLO"  # W->C
WELCOME = "WELCOME"  # C->W
REJECT = "REJECT"  # C->W
PING = "PING"  # C->W
PONG = "PONG"  # W->C
PROBE = "PROBE"  # C->W
PROBE_ACK = "PROBE_ACK"  # W->C
TASK_ASSIGN = "TASK_ASSIGN"  # C->W
TASK_RESULT = "TASK_RESULT"  # W->C
TASK_CANCEL = "TASK_CANCEL"  # C->W
CONTROL = "CONTROL"  # C->W
BYE = "BYE"  # either direction

MESSAGE_TYPES = frozenset(
    {
        CHALLENGE,
        HELLO,
        WELCOME,
        REJECT,
        PING,
        PONG,
        PROBE,
        PROBE_ACK,
        TASK_ASSIGN,
        TASK_RESULT,
        TASK_CANCEL,
        CONTROL,
        BYE,
    }
)
# Syntax of a type name. A syntactically valid but *unknown* type is accepted
# by the framing layer (forward compatibility); validation decides what to do.
_TYPE_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,31}$")

COMPRESSION_NONE = "none"
COMPRESSION_ZLIB = "zlib"
COMPRESSIONS = frozenset({COMPRESSION_NONE, COMPRESSION_ZLIB})

# Array dtypes allowed on the wire. Multi-byte types are always sent
# little-endian, so x86 laptops and ARM phones agree byte for byte.
ARRAY_DTYPES: dict[str, np.dtype[Any]] = {
    "uint8": np.dtype("<u1"),
    "int32": np.dtype("<i4"),
    "int64": np.dtype("<i8"),
}
ARRAY_FIELDS = ("shape", "dtype", "comp", "raw_len", "sha256")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class WireError(Exception):
    """Base class of every error caused by data received from the network."""


class ProtocolError(WireError):
    """Malformed frame or header. The connection must be closed."""


class ValidationError(ProtocolError):
    """Well-formed message whose content violates a schema or limit."""


class IntegrityError(WireError):
    """Payload hash mismatch: corrupted in transit (not a dishonest worker)."""


class ConnectionClosed(WireError):
    """The peer closed the connection at a frame boundary."""


# --------------------------------------------------------------------------
# Data types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameLimits:
    """Size limits applied before any allocation (spec S.5.1, S.5.4)."""

    max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES
    max_raw_array_bytes: int = DEFAULT_MAX_RAW_ARRAY_BYTES


DEFAULT_LIMITS = FrameLimits()


@dataclass
class Message:
    """One protocol message.

    ``header`` always contains ``"type"`` and ``"v"`` once encoded or decoded;
    ``type`` is duplicated as an attribute for convenient dispatching.
    """

    type: str
    header: dict[str, Any] = field(default_factory=dict)
    payload: bytes = b""


def make_message(msg_type: str, payload: bytes = b"", **fields: Any) -> Message:
    """Build a message with a header containing ``type``, ``v`` and ``fields``."""
    header: dict[str, Any] = {"type": msg_type, "v": PROTOCOL_VERSION}
    header.update(fields)
    return Message(msg_type, header, payload)


# --------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------


def sha256_hex(data: bytes) -> str:
    """Lower-case hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------
# Header (JSON) encoding/decoding
# --------------------------------------------------------------------------


def _reject_constant(name: str) -> Any:
    """``json.loads`` hook for the non-standard tokens NaN/Infinity/-Infinity."""
    raise ProtocolError(f"non-finite JSON constant {name!r} not allowed")


def _parse_float(text: str) -> float:
    """``json.loads`` hook: reject numbers like ``1e999`` that overflow to inf."""
    value = float(text)
    if not math.isfinite(value):
        raise ProtocolError("non-finite JSON number not allowed")
    return value


def _parse_int(text: str) -> int:
    """``json.loads`` hook: refuse absurdly long integers before converting."""
    if len(text.lstrip("-")) > MAX_INT_DIGITS:
        raise ProtocolError("JSON integer too long")
    return int(text)


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """``json.loads`` hook: duplicate keys are ambiguous (which one wins?), so reject."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def json_depth(value: Any) -> int:
    """Nesting depth of a JSON value (scalars = 0, ``{}``/``[]`` = 1).

    Iterative, so a hostile deeply nested value cannot cause a RecursionError here.
    """
    depth = 0
    stack: list[tuple[Any, int]] = [(value, 0)]
    while stack:
        node, level = stack.pop()
        if isinstance(node, dict):
            depth = max(depth, level + 1)
            stack.extend((child, level + 1) for child in node.values())
        elif isinstance(node, list):
            depth = max(depth, level + 1)
            stack.extend((child, level + 1) for child in node)
    return depth


def strict_json_loads(data: bytes | str) -> Any:
    """Parse untrusted JSON strictly.

    Rejects: invalid UTF-8, NaN/Infinity, numbers that overflow to infinity,
    integers longer than 20 digits, duplicate keys and nesting deeper than 8.
    Raises only :class:`ProtocolError`.
    """
    try:
        text = data.decode("utf-8") if isinstance(data, bytes) else data
        value = json.loads(
            text,
            parse_constant=_reject_constant,
            parse_float=_parse_float,
            parse_int=_parse_int,
            object_pairs_hook=_no_duplicate_keys,
        )
    except ProtocolError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        # JSONDecodeError is a ValueError. RecursionError: the C parser's own
        # guard against extremely deep nesting.
        raise ProtocolError(f"invalid JSON: {type(exc).__name__}") from None
    if json_depth(value) > MAX_JSON_DEPTH:
        raise ProtocolError(f"JSON nesting deeper than {MAX_JSON_DEPTH}")
    return value


def encode_header(header: dict[str, Any]) -> bytes:
    """Compact, ASCII-only JSON for a header; NaN/Infinity are refused."""
    try:
        return json.dumps(header, separators=(",", ":"), allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"header not JSON-serializable: {exc}") from None


def decode_header(data: bytes) -> dict[str, Any]:
    """Parse and check a frame header: an object with a valid ``type`` and ``v == 1``."""
    header = strict_json_loads(data)
    if not isinstance(header, dict):
        raise ProtocolError("header must be a JSON object")
    msg_type = header.get("type")
    if not isinstance(msg_type, str) or not _TYPE_RE.fullmatch(msg_type):
        raise ProtocolError("header 'type' missing or malformed")
    version = header.get("v")
    # ``True == 1`` in Python, so check the exact type.
    if type(version) is not int or version != PROTOCOL_VERSION:
        raise ProtocolError("unsupported or missing protocol version 'v'")
    return header


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------


def _check_prefix(prefix: bytes, limits: FrameLimits) -> tuple[int, int]:
    """Validate the 12-byte prefix; return (header_len, payload_len)."""
    magic, header_len, payload_len = PREFIX.unpack(prefix)
    if magic != MAGIC:
        raise ProtocolError("bad magic (not an MDC1 frame)")
    if header_len == 0 or header_len > limits.max_header_bytes:
        raise ProtocolError(f"header length {header_len} outside 1..{limits.max_header_bytes}")
    if payload_len > limits.max_payload_bytes:
        raise ProtocolError(f"payload length {payload_len} > {limits.max_payload_bytes}")
    return header_len, payload_len


def encode_frame(msg: Message, limits: FrameLimits = DEFAULT_LIMITS) -> bytes:
    """Serialize ``msg`` into one frame.

    The header's ``type`` and ``v`` are always set from ``msg.type`` and the
    protocol version. Oversized headers/payloads are refused on the sending
    side too, so we never emit a frame the peer would reject.
    """
    if not _TYPE_RE.fullmatch(msg.type):
        raise ProtocolError(f"invalid message type {msg.type!r}")
    header = dict(msg.header)
    header["type"] = msg.type
    header["v"] = PROTOCOL_VERSION
    header_bytes = encode_header(header)
    if len(header_bytes) > limits.max_header_bytes:
        raise ProtocolError(f"header too large ({len(header_bytes)} bytes)")
    if len(msg.payload) > limits.max_payload_bytes:
        raise ProtocolError(f"payload too large ({len(msg.payload)} bytes)")
    return PREFIX.pack(MAGIC, len(header_bytes), len(msg.payload)) + header_bytes + msg.payload


def decode_frame(data: bytes, limits: FrameLimits = DEFAULT_LIMITS) -> Message:
    """Parse exactly one complete frame from ``data`` (used by tests and fuzzing)."""
    if len(data) < PREFIX_LEN:
        raise ProtocolError("frame shorter than its 12-byte prefix")
    header_len, payload_len = _check_prefix(data[:PREFIX_LEN], limits)
    if len(data) != PREFIX_LEN + header_len + payload_len:
        raise ProtocolError("frame length does not match its prefix")
    header = decode_header(data[PREFIX_LEN : PREFIX_LEN + header_len])
    payload = bytes(data[PREFIX_LEN + header_len :])
    return Message(header["type"], header, payload)


async def read_frame(reader: asyncio.StreamReader, limits: FrameLimits = DEFAULT_LIMITS) -> Message:
    """Read one frame from a stream.

    ``readexactly`` waits until the requested number of bytes has arrived, so
    frames split across many TCP segments (or delivered byte by byte) are
    reassembled correctly.

    Raises:
        ConnectionClosed: EOF exactly at a frame boundary (normal close).
        ProtocolError: bad prefix/header, or EOF in the middle of a frame.
    """
    try:
        prefix = await reader.readexactly(PREFIX_LEN)
    except asyncio.IncompleteReadError as exc:
        if not exc.partial:
            raise ConnectionClosed("peer closed the connection") from None
        raise ProtocolError("connection closed inside a frame prefix") from None
    header_len, payload_len = _check_prefix(prefix, limits)
    try:
        header_bytes = await reader.readexactly(header_len)
        header = decode_header(header_bytes)
        payload = await reader.readexactly(payload_len) if payload_len else b""
    except asyncio.IncompleteReadError:
        raise ProtocolError("connection closed inside a frame") from None
    return Message(header["type"], header, payload)


async def write_frame(
    writer: asyncio.StreamWriter,
    msg: Message,
    lock: asyncio.Lock,
    limits: FrameLimits = DEFAULT_LIMITS,
) -> int:
    """Encode and send ``msg``; return the number of bytes written.

    The per-connection ``lock`` serializes writers: without it, two coroutines
    (e.g. the ping loop and the dispatcher) could interleave their bytes and
    corrupt both frames. ``drain()`` applies back-pressure when the socket's
    send buffer is full.
    """
    frame = encode_frame(msg, limits)
    async with lock:
        writer.write(frame)
        await writer.drain()
    return len(frame)


# --------------------------------------------------------------------------
# Arrays
# --------------------------------------------------------------------------


def pack_array(
    array: np.ndarray[Any, Any], compression: str = COMPRESSION_ZLIB, level: int = 1
) -> tuple[dict[str, Any], bytes]:
    """Serialize a numpy array for a TASK_ASSIGN / TASK_RESULT payload.

    Returns ``(header_fields, payload)``. ``sha256`` is computed over the
    **uncompressed** little-endian bytes, so the receiver can check exactly
    the data it will compute on.
    """
    if compression not in COMPRESSIONS:
        raise ValueError(f"unknown compression {compression!r}")
    dtype_name = array.dtype.name
    if dtype_name not in ARRAY_DTYPES:
        raise ValueError(f"dtype {dtype_name!r} not allowed on the wire")
    if not 1 <= array.ndim <= MAX_ARRAY_NDIM or any(
        not 1 <= d <= MAX_ARRAY_DIM for d in array.shape
    ):
        raise ValueError(f"array shape {array.shape} not allowed on the wire")
    wire = np.ascontiguousarray(array, dtype=ARRAY_DTYPES[dtype_name])
    raw = wire.tobytes()
    payload = zlib.compress(raw, level) if compression == COMPRESSION_ZLIB else raw
    fields: dict[str, Any] = {
        "shape": [int(d) for d in array.shape],
        "dtype": dtype_name,
        "comp": compression,
        "raw_len": len(raw),
        "sha256": sha256_hex(raw),
    }
    return fields, payload


def validate_array_fields(
    fields: dict[str, Any], max_raw_bytes: int = DEFAULT_MAX_RAW_ARRAY_BYTES
) -> tuple[tuple[int, ...], np.dtype[Any], str, int, str]:
    """Check array header fields (spec S.5.4) *before* touching the payload.

    Returns ``(shape, wire_dtype, comp, raw_len, sha256)``.
    """
    shape = fields.get("shape")
    if not isinstance(shape, list) or not 1 <= len(shape) <= MAX_ARRAY_NDIM:
        raise ValidationError(f"array 'shape' must be a list of 1..{MAX_ARRAY_NDIM} ints")
    for dim in shape:
        if type(dim) is not int or not 1 <= dim <= MAX_ARRAY_DIM:
            raise ValidationError(f"array dimension must be an int in 1..{MAX_ARRAY_DIM}")
    dtype_name = fields.get("dtype")
    if not isinstance(dtype_name, str) or dtype_name not in ARRAY_DTYPES:
        raise ValidationError(f"array 'dtype' must be one of {sorted(ARRAY_DTYPES)}")
    comp = fields.get("comp")
    if not isinstance(comp, str) or comp not in COMPRESSIONS:
        raise ValidationError(f"array 'comp' must be one of {sorted(COMPRESSIONS)}")
    raw_len = fields.get("raw_len")
    wire_dtype = ARRAY_DTYPES[dtype_name]
    expected = math.prod(shape) * wire_dtype.itemsize
    if type(raw_len) is not int or raw_len != expected:
        raise ValidationError("array 'raw_len' must equal prod(shape) * itemsize")
    if raw_len > max_raw_bytes:
        raise ValidationError(f"array of {raw_len} bytes exceeds limit {max_raw_bytes}")
    digest = fields.get("sha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise ValidationError("array 'sha256' must be 64 lower-case hex characters")
    return tuple(shape), wire_dtype, comp, raw_len, digest


def safe_decompress(payload: bytes, raw_len: int) -> bytes:
    """Decompress zlib data that must expand to exactly ``raw_len`` bytes.

    Decompression-bomb guard (spec S.5.4): ``decompress(data, raw_len + 1)``
    never produces more than ``raw_len + 1`` bytes, however small and
    malicious the input. If more output is pending (``unconsumed_tail``), or
    the stream ends early, has trailing garbage or is corrupt, it is rejected.
    """
    decompressor = zlib.decompressobj()
    try:
        out = decompressor.decompress(payload, raw_len + 1)
    except zlib.error:
        raise ValidationError("corrupt zlib payload") from None
    if len(out) != raw_len or decompressor.unconsumed_tail:
        raise ValidationError("decompressed size does not match 'raw_len'")
    if not decompressor.eof or decompressor.unused_data:
        raise ValidationError("zlib stream incomplete or followed by extra data")
    return out


def unpack_array(
    fields: dict[str, Any],
    payload: bytes,
    max_raw_bytes: int = DEFAULT_MAX_RAW_ARRAY_BYTES,
) -> np.ndarray[Any, Any]:
    """Inverse of :func:`pack_array`, for untrusted input.

    Order of checks: header fields -> size -> (bounded) decompression ->
    SHA-256 integrity. The returned array is read-only (it views the received
    bytes); call ``.copy()`` if you need to modify it.

    Raises:
        ValidationError: bad fields, size mismatch, decompression bomb/corruption.
        IntegrityError: payload does not hash to the advertised ``sha256``.
    """
    shape, wire_dtype, comp, raw_len, digest = validate_array_fields(fields, max_raw_bytes)
    if comp == COMPRESSION_ZLIB:
        raw = safe_decompress(payload, raw_len)
    else:
        if len(payload) != raw_len:
            raise ValidationError("uncompressed payload length does not match 'raw_len'")
        raw = bytes(payload)
    # Transport integrity: constant-time comparison is not strictly needed for
    # a public hash, but using it everywhere keeps the rule simple.
    if not hmac.compare_digest(sha256_hex(raw).encode("ascii"), digest.encode("ascii")):
        raise IntegrityError("payload SHA-256 mismatch (corrupted in transit)")
    array = np.frombuffer(raw, dtype=wire_dtype).reshape(shape)
    return array.astype(wire_dtype.newbyteorder("="), copy=False)
