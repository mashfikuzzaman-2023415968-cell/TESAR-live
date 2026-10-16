"""Fuzzing the message validators and join-code decoder (spec G.3 Phase 1).

Property: any header (random, or a valid message with random mutations) is
either accepted or rejected with ValidationError — no other exception.
"""

from __future__ import annotations

import contextlib
import copy

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mdc.common import protocol as P
from mdc.common import validation as V
from mdc.common.credentials import CredentialError, decode_join_code
from tests.helpers.messages import sample_messages

pytestmark = pytest.mark.property

FUZZ = settings(max_examples=250, deadline=None, suppress_health_check=[HealthCheck.too_slow])
SAMPLES = sample_messages()
JSON_SCALARS = (
    st.none()
    | st.booleans()
    | st.integers(-(2**64), 2**64)
    | st.floats(allow_nan=True, allow_infinity=True)
    | st.text(max_size=40)
)
JSON_VALUES = st.recursive(
    JSON_SCALARS,
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=10), children, max_size=4)
    ),
    max_leaves=25,
)


def _validate(msg: P.Message) -> None:
    with contextlib.suppress(V.ValidationError):
        V.validate_message(msg)


@FUZZ
@given(
    st.sampled_from(sorted(P.MESSAGE_TYPES)),
    st.dictionaries(st.text(max_size=12), JSON_VALUES, max_size=10),
    st.binary(max_size=64),
)
def test_random_headers(msg_type: str, header: dict, payload: bytes) -> None:
    _validate(P.Message(msg_type, header, payload))


@FUZZ
@given(st.sampled_from(sorted(P.MESSAGE_TYPES)), st.data())
def test_mutated_valid_headers(msg_type: str, data: st.DataObject) -> None:
    msg = copy.deepcopy(SAMPLES[msg_type])
    keys = sorted(msg.header)
    for _ in range(data.draw(st.integers(1, 3))):
        key = data.draw(st.sampled_from(keys))
        if data.draw(st.booleans()):
            msg.header[key] = data.draw(JSON_VALUES)
        else:
            msg.header.pop(key, None)
    _validate(msg)


@FUZZ
@given(st.sampled_from(sorted(V.TASK_KINDS | {"other"})), JSON_VALUES, JSON_VALUES)
def test_task_params_and_results(kind: str, params: object, result: object) -> None:
    with contextlib.suppress(V.ValidationError):
        V.validate_task_params(kind, params)
    with contextlib.suppress(V.ValidationError):
        V.validate_json_result(kind, result)


@FUZZ
@given(st.text(max_size=1500) | st.from_regex(r"MDC1\.[A-Za-z0-9_-]{0,300}", fullmatch=True))
def test_random_join_codes(text: str) -> None:
    with contextlib.suppress(CredentialError):
        decode_join_code(text)
