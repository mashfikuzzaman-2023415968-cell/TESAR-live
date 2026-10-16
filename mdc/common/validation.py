"""Strict validation of every message received from the network (spec S.5).

Every frame from the network is untrusted, **on both sides**: a worker may be
compromised, and a phone must not trust the coordinator blindly either. The
framing layer (``protocol.py``) already guarantees a well-formed JSON object
with a valid ``type`` and ``v``; this module checks the *content*:

* one small, explicit validator per message type (required fields, types,
  ranges, regexes, allowed values) — readable code instead of a schema
  library;
* payload rules per type (most control messages must have none);
* task parameters against hard limits (``max_task_params``), enforced by the
  worker with its *own* copy of the limits, never trusting the coordinator's;
* per-kind result checks (histogram sums, digest format, output shape);
* clamping of self-reported values that only steer scheduling estimates.

Unknown fields are ignored (forward compatibility) but logged at DEBUG.
Unknown message *types* raise :class:`UnknownMessageType` so the caller can
count them (more than 10 after authentication -> disconnect).

Every failure raises :class:`ValidationError` (never ``assert``, which
``python -O`` removes).
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mdc.common import protocol as P
from mdc.common.auth import MAC_HEX_LEN
from mdc.common.protocol import ValidationError, validate_array_fields
from mdc.common.safe_names import NAME_RE
from mdc.common.tls import FINGERPRINT_RE

__all__ = [
    "BATTERY_SOURCES",
    "DEFAULT_TASK_LIMITS",
    "TASK_KINDS",
    "TaskLimits",
    "UnknownMessageType",
    "ValidationError",
    "clamp",
    "clamp_battery",
    "clamp_hello",
    "validate_json_result",
    "validate_message",
    "validate_result_shape",
    "validate_task_input",
    "validate_task_params",
]

log = logging.getLogger(__name__)

ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
HEX32_RE = re.compile(r"^[0-9a-f]{32}$")
HEX64_RE = re.compile(r"^[0-9a-f]{%d}$" % MAC_HEX_LEN)  # noqa: UP031 - regex braces
VERSION_RE = re.compile(r"^[0-9A-Za-z.+-]{1,32}$")
MAX_STRING = 256
MAX_SEQ = 2**53  # largest integer a float (and JSON in other languages) represents exactly
MAX_TIME = 1e12  # monotonic seconds; generous but finite
MAX_MS = 1e9
MAX_COUNT = 10**6
MAX_SELF_REPORTED_SLOTS = 64
MAX_CPU_COUNT = 1024
MAX_PING_INTERVAL_S = 60.0
MAX_VOLTAGE = 100.0
MAX_CURRENT_MA = 1e6
MAX_SLOWDOWN = 100.0
PERCENT = 100.0
HIST_BINS = 256

PLATFORMS = frozenset({"android", "linux", "windows", "darwin"})
DEVICE_CLASSES = frozenset({"phone", "laptop"})
BATTERY_SOURCES = frozenset({"termux", "sysfs", "mock"})
REJECT_REASONS = frozenset({"auth", "revoked", "busy", "protocol", "version"})
RESULT_STATUSES = frozenset({"ok", "error"})
CONTROL_ACTIONS = frozenset({"crash", "leave", "set_malicious", "set_slowdown", "set_drop"})

# The only computations a worker will ever run (spec S.7.1). Phase 2 binds each
# name to its kernel in ``mdc.tasks.base.TASK_KINDS``.
TASK_KINDS = frozenset({"img_tile", "row_stats", "synthetic"})
# Payload expectations per kind: does TASK_ASSIGN carry an input array?
KINDS_WITH_INPUT = frozenset({"img_tile", "row_stats"})

# Clamping ranges for self-reported values (spec S.6.8).
SLOTS_CLAMP = (1, 8)
BENCH_CLAMP = (0.01, 1e6)


# --------------------------------------------------------------------------
# Small typed field checkers
# --------------------------------------------------------------------------


class UnknownMessageType(ValidationError):
    """A syntactically valid message type this version does not know."""


def _fail(field_name: str, why: str) -> ValidationError:
    return ValidationError(f"field {field_name!r}: {why}")


def req_str(
    h: dict[str, Any],
    key: str,
    *,
    max_len: int = MAX_STRING,
    pattern: re.Pattern[str] | None = None,
    choices: frozenset[str] | None = None,
) -> str:
    """Required string field, optionally constrained by regex or allowed set."""
    value = h.get(key)
    if not isinstance(value, str):
        raise _fail(key, "must be a string")
    if len(value) > max_len:
        raise _fail(key, f"longer than {max_len} characters")
    if pattern is not None and not pattern.fullmatch(value):
        raise _fail(key, "has an invalid format")
    if choices is not None and value not in choices:
        raise _fail(key, f"must be one of {sorted(choices)}")
    return value


def opt_str(h: dict[str, Any], key: str, *, max_len: int = MAX_STRING) -> str | None:
    """Optional (missing or null) string field."""
    if h.get(key) is None:
        return None
    return req_str(h, key, max_len=max_len)


def req_int(h: dict[str, Any], key: str, lo: int, hi: int) -> int:
    """Required integer field in ``[lo, hi]`` (``bool`` is not an integer here)."""
    value = h.get(key)
    if type(value) is not int:
        raise _fail(key, "must be an integer")
    if not lo <= value <= hi:
        raise _fail(key, f"must be in [{lo}, {hi}]")
    return value


def req_float(h: dict[str, Any], key: str, lo: float, hi: float) -> float:
    """Required finite number in ``[lo, hi]`` (ints accepted, bools rejected)."""
    value = h.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(key, "must be a number")
    as_float = float(value)
    if not math.isfinite(as_float) or not lo <= as_float <= hi:
        raise _fail(key, f"must be a finite number in [{lo}, {hi}]")
    return as_float


def opt_float(h: dict[str, Any], key: str, lo: float, hi: float) -> float | None:
    """Optional (missing or null) number field."""
    if h.get(key) is None:
        return None
    return req_float(h, key, lo, hi)


def req_bool(h: dict[str, Any], key: str) -> bool:
    """Required boolean field."""
    value = h.get(key)
    if not isinstance(value, bool):
        raise _fail(key, "must be a boolean")
    return value


def opt_bool(h: dict[str, Any], key: str) -> bool | None:
    """Optional (missing or null) boolean field."""
    if h.get(key) is None:
        return None
    return req_bool(h, key)


def req_dict(h: dict[str, Any], key: str) -> dict[str, Any]:
    """Required JSON object field."""
    value = h.get(key)
    if not isinstance(value, dict):
        raise _fail(key, "must be an object")
    return value


# --------------------------------------------------------------------------
# Composite objects
# --------------------------------------------------------------------------


def validate_battery(value: object) -> dict[str, Any]:
    """Battery object (spec 6.3). Every field may be null; types are strict."""
    if not isinstance(value, dict):
        raise _fail("battery", "must be an object")
    opt_float(value, "pct", -1e6, 1e6)  # range clamped later, see clamp_battery
    opt_bool(value, "charging")
    opt_float(value, "current_ma", -MAX_CURRENT_MA, MAX_CURRENT_MA)
    opt_float(value, "voltage_v", 0.0, MAX_VOLTAGE)
    opt_float(value, "ts_w", -MAX_TIME, MAX_TIME)
    source = value.get("source")
    if source is not None and (not isinstance(source, str) or source not in BATTERY_SOURCES):
        raise _fail("battery.source", f"must be one of {sorted(BATTERY_SOURCES)}")
    return value


@dataclass(frozen=True)
class TaskLimits:
    """Hard limits on task parameters (spec ``security.max_task_params``)."""

    blur_radius: int = 5
    blur_iterations: int = 64
    chain_len: int = 5_000_000
    max_tile_pixels: int = 4_194_304

    def to_dict(self) -> dict[str, int]:
        """As sent in WELCOME ``limits``."""
        return {
            "blur_radius": self.blur_radius,
            "blur_iterations": self.blur_iterations,
            "chain_len": self.chain_len,
            "max_tile_pixels": self.max_tile_pixels,
        }


DEFAULT_TASK_LIMITS = TaskLimits()


def validate_limits(value: object) -> TaskLimits:
    """WELCOME ``limits`` object: four positive integers."""
    if not isinstance(value, dict):
        raise _fail("limits", "must be an object")
    return TaskLimits(
        blur_radius=req_int(value, "blur_radius", 1, 1000),
        blur_iterations=req_int(value, "blur_iterations", 1, 100_000),
        chain_len=req_int(value, "chain_len", 1, 10**9),
        max_tile_pixels=req_int(value, "max_tile_pixels", 1, P.MAX_ARRAY_DIM**2),
    )


def validate_task_params(
    kind: str, params: object, limits: TaskLimits = DEFAULT_TASK_LIMITS
) -> dict[str, Any]:
    """Check a task's parameters against its kind and the hard limits (spec S.5.5)."""
    if kind not in TASK_KINDS:
        raise _fail("kind", f"unknown task kind; allowed: {sorted(TASK_KINDS)}")
    if not isinstance(params, dict):
        raise _fail("params", "must be an object")
    expected: dict[str, set[str]] = {
        "img_tile": {"r", "iters"},
        "row_stats": {"edge_threshold"},
        "synthetic": {"seed", "chain_len"},
    }
    if set(params) != expected[kind]:
        raise _fail("params", f"{kind} needs exactly {sorted(expected[kind])}")
    if kind == "img_tile":
        req_int(params, "r", 1, limits.blur_radius)
        req_int(params, "iters", 1, limits.blur_iterations)
    elif kind == "row_stats":
        req_int(params, "edge_threshold", 0, 255)
    else:
        req_int(params, "seed", 0, 2**63 - 1)
        req_int(params, "chain_len", 1, limits.chain_len)
    return params


def img_tile_halo(params: dict[str, Any]) -> int:
    """Total shrink per side of the blur+Sobel pipeline: ``r * iters + 1`` (spec 8.1)."""
    return int(params["r"]) * int(params["iters"]) + 1


def validate_task_input(
    kind: str,
    params: dict[str, Any],
    shape: tuple[int, ...] | None,
    dtype: str | None,
    limits: TaskLimits = DEFAULT_TASK_LIMITS,
) -> None:
    """Check that the input array (or its absence) fits the task kind."""
    if kind not in KINDS_WITH_INPUT:
        if shape is not None:
            raise _fail("payload", f"{kind} takes no input array")
        return
    if shape is None or dtype is None:
        raise _fail("payload", f"{kind} requires an input array")
    if dtype != "uint8" or len(shape) != 2:  # noqa: PLR2004 - 2-D grayscale image
        raise _fail("shape", f"{kind} input must be a 2-D uint8 array")
    if shape[0] * shape[1] > limits.max_tile_pixels:
        raise _fail("shape", f"input larger than max_tile_pixels={limits.max_tile_pixels}")
    if kind == "img_tile":
        halo = img_tile_halo(params)
        if shape[0] <= 2 * halo or shape[1] <= 2 * halo:
            raise _fail("shape", f"img_tile input must be larger than 2*halo={2 * halo}")


def validate_json_result(kind: str, result: object) -> dict[str, Any]:
    """Check a ``json_result`` against its kind (spec S.5.6)."""
    if not isinstance(result, dict):
        raise _fail("json_result", "must be an object")
    if kind == "row_stats":
        if set(result) != {"hist", "edge_pixels", "pixels"}:
            raise _fail("json_result", "row_stats needs exactly hist, edge_pixels, pixels")
        pixels = req_int(result, "pixels", 0, P.MAX_ARRAY_DIM**2)
        req_int(result, "edge_pixels", 0, pixels)
        hist = result["hist"]
        if not isinstance(hist, list) or len(hist) != HIST_BINS:
            raise _fail("hist", f"must be a list of exactly {HIST_BINS} integers")
        for count in hist:
            if type(count) is not int or count < 0:
                raise _fail("hist", "entries must be non-negative integers")
        if sum(hist) != pixels:
            raise _fail("hist", "entries must sum to 'pixels'")
        return result
    if kind == "synthetic":
        if set(result) != {"digest"}:
            raise _fail("json_result", "synthetic needs exactly 'digest'")
        req_str(result, "digest", pattern=HEX64_RE)
        return result
    raise _fail("json_result", f"kind {kind!r} does not produce a json_result")


def validate_result_shape(expected: tuple[int, ...], actual: tuple[int, ...]) -> None:
    """The output array must have exactly the expected chunk output shape."""
    if tuple(expected) != tuple(actual):
        raise _fail("shape", f"expected output shape {tuple(expected)}, got {tuple(actual)}")


# --------------------------------------------------------------------------
# Per-message validators
# --------------------------------------------------------------------------


def _v_challenge(h: dict[str, Any]) -> None:
    req_str(h, "nonce", pattern=HEX32_RE)
    req_str(h, "server_version", pattern=VERSION_RE)


def _v_hello(h: dict[str, Any]) -> None:
    req_str(h, "worker_id", pattern=ID_RE)
    req_str(h, "name", pattern=NAME_RE)
    req_str(h, "platform", choices=PLATFORMS)
    req_str(h, "device_class", choices=DEVICE_CLASSES)
    req_int(h, "cpu_count", 1, MAX_CPU_COUNT)
    req_int(h, "slots", 1, MAX_SELF_REPORTED_SLOTS)
    req_float(h, "bench_ms_per_unit", 0.0, 1e12)
    validate_battery(h.get("battery"))
    req_str(h, "version", pattern=VERSION_RE)
    req_bool(h, "trusted_local")
    req_str(h, "nonce_w", pattern=HEX32_RE)
    req_str(h, "auth", pattern=HEX64_RE)


def _v_welcome(h: dict[str, Any]) -> None:
    req_str(h, "session_id", pattern=ID_RE)
    req_float(h, "ping_interval_s", 0.01, MAX_PING_INTERVAL_S)
    req_int(h, "accepted_slots", SLOTS_CLAMP[0], SLOTS_CLAMP[1])
    req_str(h, "proof", pattern=HEX64_RE)
    validate_limits(h.get("limits"))


def _v_reject(h: dict[str, Any]) -> None:
    req_str(h, "reason", choices=REJECT_REASONS)


def _v_ping(h: dict[str, Any]) -> None:
    req_int(h, "seq", 0, MAX_SEQ)
    req_float(h, "t_c", -MAX_TIME, MAX_TIME)


def _v_pong(h: dict[str, Any]) -> None:
    _v_ping(h)
    req_int(h, "active_tasks", 0, MAX_COUNT)
    req_int(h, "queued_tasks", 0, MAX_COUNT)
    validate_battery(h.get("battery"))


def _v_probe(h: dict[str, Any]) -> None:
    req_str(h, "probe_id", pattern=ID_RE)
    req_float(h, "t_c", -MAX_TIME, MAX_TIME)


def _v_probe_ack(h: dict[str, Any]) -> None:
    _v_probe(h)
    req_int(h, "received_bytes", 0, 2**32)


def _v_task_assign(h: dict[str, Any]) -> None:
    req_str(h, "task_id", pattern=ID_RE)
    req_str(h, "attempt_id", pattern=ID_RE)
    kind = req_str(h, "kind", choices=TASK_KINDS)
    req_dict(h, "params")
    if kind in KINDS_WITH_INPUT:
        validate_array_fields(h)


def _v_task_result(h: dict[str, Any]) -> None:
    req_str(h, "task_id", pattern=ID_RE)
    req_str(h, "attempt_id", pattern=ID_RE)
    status = req_str(h, "status", choices=RESULT_STATUSES)
    opt_str(h, "error")
    req_float(h, "compute_ms", 0.0, MAX_MS)
    req_float(h, "queue_ms", 0.0, MAX_MS)
    req_float(h, "t_recv_w", -MAX_TIME, MAX_TIME)
    req_float(h, "t_send_w", -MAX_TIME, MAX_TIME)
    has_array = "shape" in h
    has_json = h.get("json_result") is not None
    if has_array and has_json:
        raise _fail("json_result", "a result carries an array or a json_result, not both")
    if has_array:
        validate_array_fields(h)
    if has_json:
        req_dict(h, "json_result")
    if status == "error" and (has_array or has_json):
        raise _fail("status", "an error result carries no output")
    if status == "ok" and not (has_array or has_json):
        raise _fail("status", "an ok result must carry an array or a json_result")


def _v_task_cancel(h: dict[str, Any]) -> None:
    req_str(h, "attempt_id", pattern=ID_RE)


def _v_control(h: dict[str, Any]) -> None:
    action = req_str(h, "action", choices=CONTROL_ACTIONS)
    if action in {"set_malicious", "set_drop"}:
        req_float(h, "value", 0.0, 1.0)
    elif action == "set_slowdown":
        req_float(h, "value", 1.0, MAX_SLOWDOWN)
    elif h.get("value") is not None:
        raise _fail("value", f"{action} takes no value")


def _v_bye(h: dict[str, Any]) -> None:
    req_str(h, "reason")


# Message type -> (validator, payload rule). Payload rules:
#   "none"  - payload must be empty
#   "any"   - any payload (PROBE: random bytes; size already capped by framing)
#   "array" - payload present iff the header has array fields
VALIDATORS: dict[str, tuple[Callable[[dict[str, Any]], None], str]] = {
    P.CHALLENGE: (_v_challenge, "none"),
    P.HELLO: (_v_hello, "none"),
    P.WELCOME: (_v_welcome, "none"),
    P.REJECT: (_v_reject, "none"),
    P.PING: (_v_ping, "none"),
    P.PONG: (_v_pong, "none"),
    P.PROBE: (_v_probe, "any"),
    P.PROBE_ACK: (_v_probe_ack, "none"),
    P.TASK_ASSIGN: (_v_task_assign, "array"),
    P.TASK_RESULT: (_v_task_result, "array"),
    P.TASK_CANCEL: (_v_task_cancel, "none"),
    P.CONTROL: (_v_control, "none"),
    P.BYE: (_v_bye, "none"),
}

KNOWN_FIELDS: dict[str, frozenset[str]] = {
    P.CHALLENGE: frozenset({"nonce", "server_version"}),
    P.HELLO: frozenset(
        {
            "worker_id",
            "name",
            "platform",
            "device_class",
            "cpu_count",
            "slots",
            "bench_ms_per_unit",
            "battery",
            "version",
            "trusted_local",
            "nonce_w",
            "auth",
        }
    ),
    P.WELCOME: frozenset({"session_id", "ping_interval_s", "accepted_slots", "proof", "limits"}),
    P.REJECT: frozenset({"reason"}),
    P.PING: frozenset({"seq", "t_c"}),
    P.PONG: frozenset({"seq", "t_c", "active_tasks", "queued_tasks", "battery"}),
    P.PROBE: frozenset({"probe_id", "t_c"}),
    P.PROBE_ACK: frozenset({"probe_id", "t_c", "received_bytes"}),
    P.TASK_ASSIGN: frozenset({"task_id", "attempt_id", "kind", "params", *P.ARRAY_FIELDS}),
    P.TASK_RESULT: frozenset(
        {
            "task_id",
            "attempt_id",
            "status",
            "error",
            "compute_ms",
            "queue_ms",
            "t_recv_w",
            "t_send_w",
            "json_result",
            *P.ARRAY_FIELDS,
        }
    ),
    P.TASK_CANCEL: frozenset({"attempt_id"}),
    P.CONTROL: frozenset({"action", "value"}),
    P.BYE: frozenset({"reason"}),
}


def validate_message(msg: P.Message) -> P.Message:
    """Validate a decoded message's header and payload; return it unchanged.

    Raises:
        UnknownMessageType: syntactically fine but unknown type (caller decides).
        ValidationError: anything else that is wrong.
    """
    entry = VALIDATORS.get(msg.type)
    if entry is None:
        raise UnknownMessageType(f"unknown message type {msg.type!r}")
    validator, payload_rule = entry
    validator(msg.header)
    if payload_rule == "none" and msg.payload:
        raise ValidationError(f"{msg.type} must not carry a payload")
    if payload_rule == "array":
        has_array = "shape" in msg.header
        if has_array and not msg.payload:
            raise ValidationError(f"{msg.type} announces an array but has no payload")
        if not has_array and msg.payload:
            raise ValidationError(f"{msg.type} carries a payload without array fields")
    unknown = set(msg.header) - KNOWN_FIELDS[msg.type] - {"type", "v"}
    if unknown:
        log.debug("ignoring unknown fields in %s: %s", msg.type, sorted(unknown)[:10])
    return msg


# --------------------------------------------------------------------------
# Clamping of self-reported values (spec S.6.8, S.8.1)
# --------------------------------------------------------------------------


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a finite number into ``[lo, hi]``; NaN becomes ``lo``."""
    if math.isnan(value):
        return lo
    return max(lo, min(hi, value))


def clamp_battery(battery: dict[str, Any]) -> dict[str, Any]:
    """A cleaned copy of a validated battery object with ``pct`` in [0, 100]."""
    pct = battery.get("pct")
    return {
        "pct": None if pct is None else clamp(float(pct), 0.0, PERCENT),
        "charging": battery.get("charging"),
        "current_ma": battery.get("current_ma"),
        "voltage_v": battery.get("voltage_v"),
        "source": battery.get("source"),
        "ts_w": battery.get("ts_w"),
    }


def clamp_hello(header: dict[str, Any]) -> dict[str, Any]:
    """Self-reported HELLO values, clamped: they only steer estimates (spec S.8.1)."""
    return {
        "slots": int(clamp(int(header["slots"]), *SLOTS_CLAMP)),
        "bench_ms_per_unit": clamp(float(header["bench_ms_per_unit"]), *BENCH_CLAMP),
        "battery": clamp_battery(header["battery"]),
    }


def clamp_durations(
    compute_ms: float, queue_ms: float, measured_total_s: float
) -> tuple[float, float]:
    """Clamp worker-reported ``compute_ms``/``queue_ms`` to [0, measured total].

    The coordinator measured the whole round trip itself; a worker cannot
    honestly claim to have computed for longer than that.
    """
    total_ms = max(0.0, measured_total_s * 1000.0)
    compute = clamp(compute_ms, 0.0, total_ms)
    queue = clamp(queue_ms, 0.0, max(0.0, total_ms - compute))
    return compute, queue


def is_valid_fingerprint_field(value: object) -> bool:
    """True for a ``sha256:<hex>`` fingerprint string (beacon ``fp`` field)."""
    return isinstance(value, str) and FINGERPRINT_RE.fullmatch(value) is not None
