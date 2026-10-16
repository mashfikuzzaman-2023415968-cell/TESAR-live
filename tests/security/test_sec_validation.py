"""Message validation, task-parameter limits and clamping (spec S.5, S.6.8)."""

from __future__ import annotations

import copy
import logging
import math

import pytest

from mdc.common import protocol as P
from mdc.common import validation as V
from tests.helpers.messages import HEX32, HEX64, sample_messages

pytestmark = pytest.mark.security


def _with(msg_type: str, **changes: object) -> P.Message:
    msg = copy.deepcopy(sample_messages()[msg_type])
    for key, value in changes.items():
        if value is _DELETE:
            msg.header.pop(key, None)
        else:
            msg.header[key] = value
    return msg


_DELETE = object()


@pytest.mark.parametrize("msg_type", sorted(P.MESSAGE_TYPES))
def test_every_sample_message_is_valid(msg_type: str) -> None:
    msg = sample_messages()[msg_type]
    assert V.validate_message(msg) is msg


def test_unknown_message_type() -> None:
    with pytest.raises(V.UnknownMessageType):
        V.validate_message(P.make_message("FUTURE", x=1))
    assert issubclass(V.UnknownMessageType, V.ValidationError)


def test_unknown_fields_ignored_but_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="mdc.common.validation"):
        V.validate_message(_with(P.PING, extension="future"))
    assert "extension" in caplog.text


BAD_FIELDS = [
    (P.CHALLENGE, "nonce", "00" * 15),
    (P.CHALLENGE, "nonce", HEX32.upper()),
    (P.CHALLENGE, "server_version", "1 0"),
    (P.HELLO, "worker_id", "w id"),
    (P.HELLO, "worker_id", "w" * 65),
    (P.HELLO, "name", "<script>"),
    (P.HELLO, "platform", "ios"),
    (P.HELLO, "device_class", "tablet"),
    (P.HELLO, "cpu_count", 0),
    (P.HELLO, "cpu_count", True),
    (P.HELLO, "slots", 0),
    (P.HELLO, "slots", 65),
    (P.HELLO, "slots", 2.0),
    (P.HELLO, "bench_ms_per_unit", -1.0),
    (P.HELLO, "bench_ms_per_unit", "fast"),
    (P.HELLO, "battery", None),
    (P.HELLO, "battery", {"pct": "full"}),
    (P.HELLO, "battery", {"source": "magic"}),
    (P.HELLO, "battery", {"charging": 1}),
    (P.HELLO, "battery", {"voltage_v": 1000.0}),
    (P.HELLO, "trusted_local", "yes"),
    (P.HELLO, "nonce_w", "zz" * 16),
    (P.HELLO, "auth", HEX32),
    (P.HELLO, "version", "x" * 33),
    (P.WELCOME, "session_id", ""),
    (P.WELCOME, "ping_interval_s", 0.0),
    (P.WELCOME, "ping_interval_s", 61.0),
    (P.WELCOME, "accepted_slots", 9),
    (P.WELCOME, "proof", "g" * 64),
    (P.WELCOME, "limits", {"blur_radius": 1}),
    (P.WELCOME, "limits", [1, 2]),
    (P.REJECT, "reason", "Traceback (most recent call last)"),
    (P.PING, "seq", -1),
    (P.PING, "seq", 2**53 + 1),
    (P.PING, "t_c", "now"),
    (P.PONG, "active_tasks", -1),
    (P.PONG, "queued_tasks", 10**7),
    (P.PROBE, "probe_id", "p/1"),
    (P.PROBE_ACK, "received_bytes", -5),
    (P.TASK_ASSIGN, "kind", "shell"),
    (P.TASK_ASSIGN, "kind", "os.system"),
    (P.TASK_ASSIGN, "params", "r=2"),
    (P.TASK_ASSIGN, "dtype", "float64"),
    (P.TASK_ASSIGN, "attempt_id", None),
    (P.TASK_RESULT, "status", "maybe"),
    (P.TASK_RESULT, "error", "e" * 257),
    (P.TASK_RESULT, "compute_ms", -1.0),
    (P.TASK_RESULT, "queue_ms", 1e10),
    (P.TASK_RESULT, "json_result", [1]),
    (P.TASK_CANCEL, "attempt_id", 7),
    (P.CONTROL, "action", "rm -rf"),
    (P.CONTROL, "value", 0.0),  # slowdown < 1
    (P.CONTROL, "value", 101.0),
    (P.BYE, "reason", None),
    (P.BYE, "reason", "x" * 257),
]


@pytest.mark.parametrize(("msg_type", "field", "value"), BAD_FIELDS)
def test_bad_field_rejected(msg_type: str, field: str, value: object) -> None:
    with pytest.raises(V.ValidationError):
        V.validate_message(_with(msg_type, **{field: value}))


REQUIRED = [
    (P.HELLO, "auth"),
    (P.HELLO, "worker_id"),
    (P.HELLO, "battery"),
    (P.WELCOME, "proof"),
    (P.CHALLENGE, "nonce"),
    (P.PING, "seq"),
    (P.TASK_RESULT, "attempt_id"),
    (P.TASK_ASSIGN, "kind"),
    (P.CONTROL, "action"),
    (P.REJECT, "reason"),
]


@pytest.mark.parametrize(("msg_type", "field"), REQUIRED)
def test_missing_required_field_rejected(msg_type: str, field: str) -> None:
    with pytest.raises(V.ValidationError):
        V.validate_message(_with(msg_type, **{field: _DELETE}))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_numbers_rejected_by_validators(value: float) -> None:
    # The framing layer already rejects these in JSON; validators re-check
    # (defense in depth, e.g. for in-process messages).
    with pytest.raises(V.ValidationError):
        V.validate_message(_with(P.PING, t_c=value))


def test_oversize_string_rejected() -> None:
    with pytest.raises(V.ValidationError, match="longer than"):
        V.validate_message(_with(P.BYE, reason="x" * 257))
    V.validate_message(_with(P.BYE, reason="x" * 256))  # boundary accepted


@pytest.mark.parametrize(
    "msg_type",
    [
        P.CHALLENGE,
        P.HELLO,
        P.WELCOME,
        P.REJECT,
        P.PING,
        P.PONG,
        P.PROBE_ACK,
        P.TASK_CANCEL,
        P.CONTROL,
        P.BYE,
    ],
)
def test_control_messages_must_not_carry_payload(msg_type: str) -> None:
    msg = sample_messages()[msg_type]
    msg.payload = b"smuggled"
    with pytest.raises(V.ValidationError, match="payload"):
        V.validate_message(msg)


def test_array_messages_payload_rules() -> None:
    msg = sample_messages()[P.TASK_ASSIGN]
    msg.payload = b""
    with pytest.raises(V.ValidationError, match="no payload"):
        V.validate_message(msg)
    synthetic = P.make_message(
        P.TASK_ASSIGN,
        b"stray",
        task_id="t",
        attempt_id="a",
        kind="synthetic",
        params={"seed": 1, "chain_len": 5},
    )
    with pytest.raises(V.ValidationError, match="without array"):
        V.validate_message(synthetic)
    synthetic.payload = b""
    V.validate_message(synthetic)


def test_task_result_output_rules() -> None:
    base = {
        "task_id": "t",
        "attempt_id": "a",
        "compute_ms": 1.0,
        "queue_ms": 0.0,
        "t_recv_w": 1.0,
        "t_send_w": 2.0,
    }
    ok_json = P.make_message(
        P.TASK_RESULT, status="ok", error=None, json_result={"digest": HEX64}, **base
    )
    V.validate_message(ok_json)
    error = P.make_message(P.TASK_RESULT, status="error", error="time_limit", **base)
    V.validate_message(error)
    with pytest.raises(V.ValidationError, match="must carry"):
        V.validate_message(P.make_message(P.TASK_RESULT, status="ok", **base))
    with pytest.raises(V.ValidationError, match="no output"):
        V.validate_message(
            P.make_message(P.TASK_RESULT, status="error", json_result={"digest": HEX64}, **base)
        )
    both = sample_messages()[P.TASK_RESULT]
    both.header["json_result"] = {"digest": HEX64}
    with pytest.raises(V.ValidationError, match="not both"):
        V.validate_message(both)


@pytest.mark.parametrize("action", ["crash", "leave"])
def test_control_without_value(action: str) -> None:
    V.validate_message(P.make_message(P.CONTROL, action=action, value=None))
    V.validate_message(P.make_message(P.CONTROL, action=action))
    with pytest.raises(V.ValidationError):
        V.validate_message(P.make_message(P.CONTROL, action=action, value=1))


@pytest.mark.parametrize(("action", "value"), [("set_malicious", 1.5), ("set_drop", -0.1)])
def test_control_probability_range(action: str, value: float) -> None:
    with pytest.raises(V.ValidationError):
        V.validate_message(P.make_message(P.CONTROL, action=action, value=value))
    V.validate_message(P.make_message(P.CONTROL, action=action, value=0.5))


# ----------------------------------------------------------- task params


LIMITS = V.TaskLimits(blur_radius=3, blur_iterations=10, chain_len=1000, max_tile_pixels=10000)


@pytest.mark.parametrize(
    ("kind", "params"),
    [
        ("img_tile", {"r": 3, "iters": 10}),
        ("img_tile", {"r": 1, "iters": 1}),
        ("row_stats", {"edge_threshold": 0}),
        ("row_stats", {"edge_threshold": 255}),
        ("synthetic", {"seed": 0, "chain_len": 1000}),
    ],
)
def test_task_params_within_limits_accepted(kind: str, params: dict) -> None:
    assert V.validate_task_params(kind, params, LIMITS) is params


@pytest.mark.parametrize(
    ("kind", "params"),
    [
        ("img_tile", {"r": 4, "iters": 1}),
        ("img_tile", {"r": 1, "iters": 11}),
        ("img_tile", {"r": 0, "iters": 1}),
        ("img_tile", {"r": 1.0, "iters": 1}),
        ("img_tile", {"r": 1}),
        ("img_tile", {"r": 1, "iters": 1, "cmd": "x"}),
        ("row_stats", {"edge_threshold": 256}),
        ("synthetic", {"seed": -1, "chain_len": 5}),
        ("synthetic", {"seed": 1, "chain_len": 1001}),
        ("synthetic", "seed=1"),
        ("python_code", {}),
        ("__import__", {}),
    ],
)
def test_task_params_out_of_limits_rejected(kind: str, params: object) -> None:
    with pytest.raises(V.ValidationError):
        V.validate_task_params(kind, params, LIMITS)


def test_task_input_shape_rules() -> None:
    params = {"r": 2, "iters": 2}  # halo = 5 -> input must exceed 10 x 10
    V.validate_task_input("img_tile", params, (11, 11), "uint8", LIMITS)
    with pytest.raises(V.ValidationError, match="2\\*halo"):
        V.validate_task_input("img_tile", params, (10, 50), "uint8", LIMITS)
    with pytest.raises(V.ValidationError, match="max_tile_pixels"):
        V.validate_task_input("img_tile", params, (101, 100), "uint8", LIMITS)
    with pytest.raises(V.ValidationError, match="2-D uint8"):
        V.validate_task_input("img_tile", params, (11, 11), "int32", LIMITS)
    with pytest.raises(V.ValidationError, match="2-D uint8"):
        V.validate_task_input("row_stats", {"edge_threshold": 1}, (2, 3, 4), "uint8", LIMITS)
    with pytest.raises(V.ValidationError, match="requires"):
        V.validate_task_input("row_stats", {"edge_threshold": 1}, None, None, LIMITS)
    V.validate_task_input("synthetic", {"seed": 1, "chain_len": 2}, None, None, LIMITS)
    with pytest.raises(V.ValidationError, match="no input"):
        V.validate_task_input("synthetic", {"seed": 1, "chain_len": 2}, (2, 2), "uint8", LIMITS)


def test_img_tile_halo() -> None:
    assert V.img_tile_halo({"r": 2, "iters": 8}) == 17


# ---------------------------------------------------------- json results


def _row_stats(**changes: object) -> dict:
    hist = [0] * 256
    hist[10] = 60
    hist[200] = 40
    result: dict = {"hist": hist, "edge_pixels": 40, "pixels": 100}
    result.update(changes)
    return result


def test_row_stats_result_valid() -> None:
    V.validate_json_result("row_stats", _row_stats())
    V.validate_json_result("row_stats", _row_stats(edge_pixels=100))  # boundary
    V.validate_json_result("synthetic", {"digest": HEX64})


@pytest.mark.parametrize(
    ("kind", "result"),
    [
        ("row_stats", _row_stats(hist=[0] * 255)),
        ("row_stats", _row_stats(hist=[1] * 256)),  # sums to 256, not 100
        ("row_stats", _row_stats(hist=[-1] + [0] * 254 + [101])),
        ("row_stats", _row_stats(hist=[0.5] * 256)),
        ("row_stats", _row_stats(edge_pixels=101)),
        ("row_stats", _row_stats(edge_pixels=-1)),
        ("row_stats", _row_stats(pixels=-1)),
        ("row_stats", _row_stats(extra=1)),
        ("row_stats", [1, 2]),
        ("synthetic", {"digest": "abc"}),
        ("synthetic", {"digest": HEX64.upper()}),
        ("synthetic", {"digest": HEX64, "x": 1}),
        ("img_tile", {"anything": 1}),
    ],
)
def test_bad_json_results_rejected(kind: str, result: object) -> None:
    with pytest.raises(V.ValidationError):
        V.validate_json_result(kind, result)


def test_result_shape_check() -> None:
    V.validate_result_shape((10, 20), (10, 20))
    with pytest.raises(V.ValidationError):
        V.validate_result_shape((10, 20), (20, 10))


def test_limits_round_trip() -> None:
    assert V.validate_limits(V.DEFAULT_TASK_LIMITS.to_dict()) == V.DEFAULT_TASK_LIMITS


# --------------------------------------------------------------- clamping


def test_clamp_hello_hostile_values() -> None:
    header = sample_messages()[P.HELLO].header
    header.update(slots=64, bench_ms_per_unit=0.0, battery={"pct": 250.0, "source": "mock"})
    clamped = V.clamp_hello(header)
    assert clamped["slots"] == 8
    assert clamped["bench_ms_per_unit"] == 0.01
    assert clamped["battery"]["pct"] == 100.0
    header.update(bench_ms_per_unit=1e11, battery={"pct": -5.0})
    clamped = V.clamp_hello(header)
    assert clamped["bench_ms_per_unit"] == 1e6
    assert clamped["battery"]["pct"] == 0.0


def test_clamp_battery_keeps_nulls() -> None:
    assert V.clamp_battery({})["pct"] is None


def test_clamp_and_durations() -> None:
    assert V.clamp(math.nan, 1.0, 2.0) == 1.0
    assert V.clamp(5.0, 1.0, 2.0) == 2.0
    # Worker claims 10 s of compute for a 2 s round trip -> clamped.
    assert V.clamp_durations(10_000.0, 500.0, 2.0) == (2000.0, 0.0)
    assert V.clamp_durations(-5.0, -1.0, 1.0) == (0.0, 0.0)
    assert V.clamp_durations(300.0, 200.0, 1.0) == (300.0, 200.0)
    assert V.clamp_durations(1.0, 1.0, -3.0) == (0.0, 0.0)


def test_fingerprint_field_helper() -> None:
    assert V.is_valid_fingerprint_field("sha256:" + "a" * 64)
    assert not V.is_valid_fingerprint_field("sha256:a")
