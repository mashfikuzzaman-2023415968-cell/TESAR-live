"""Fuzzing the frame decoder, JSON parser and array unpacker (spec G.3 Phase 1).

Property: arbitrary bytes only ever raise ProtocolError/ValidationError/
IntegrityError (never IndexError, KeyError, MemoryError, ...), never hang,
and use bounded memory.
"""

from __future__ import annotations

import asyncio
import contextlib
import struct

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mdc.common import protocol as P
from tests.helpers.memory import peak_extra_bytes
from tests.helpers.messages import sample_messages

pytestmark = pytest.mark.property

ALLOWED = (P.ProtocolError, P.IntegrityError)  # ValidationError is a ProtocolError
LIMITS = P.FrameLimits(max_header_bytes=4096, max_payload_bytes=65536, max_raw_array_bytes=65536)
FUZZ = settings(max_examples=250, deadline=None, suppress_health_check=[HealthCheck.too_slow])
VALID_FRAMES = [P.encode_frame(m) for m in sample_messages().values()]


def _decode(data: bytes) -> None:
    with contextlib.suppress(*ALLOWED):
        P.decode_frame(data, LIMITS)


@FUZZ
@given(st.binary(max_size=2048))
def test_random_bytes_into_decode_frame(data: bytes) -> None:
    _decode(data)


@FUZZ
@given(st.binary(max_size=2048))
def test_random_body_behind_valid_prefix(body: bytes) -> None:
    hlen = min(len(body), 4096)
    _decode(struct.pack("!4sII", P.MAGIC, hlen, len(body) - hlen) + body)


@FUZZ
@given(
    st.sampled_from(VALID_FRAMES),
    st.lists(st.tuples(st.integers(0, 10**6), st.integers(0, 255)), min_size=1, max_size=8),
)
def test_mutated_valid_frames(frame: bytes, flips: list[tuple[int, int]]) -> None:
    data = bytearray(frame)
    for pos, value in flips:
        data[pos % len(data)] = value
    with contextlib.suppress(*ALLOWED):
        P.decode_frame(bytes(data))


@FUZZ
@given(st.text(max_size=500))
def test_random_text_into_strict_json(text: str) -> None:
    with contextlib.suppress(P.ProtocolError):
        P.strict_json_loads(text)


JSON_SCALARS = (
    st.none() | st.booleans() | st.integers(-(2**70), 2**70) | st.floats() | st.text(max_size=20)
)
JSON_VALUES = st.recursive(
    JSON_SCALARS,
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=8), children, max_size=4)
    ),
    max_leaves=30,
)


@FUZZ
@given(
    st.dictionaries(st.sampled_from([*P.ARRAY_FIELDS, "x"]), JSON_VALUES, max_size=6),
    st.binary(max_size=4096),
)
def test_random_array_fields_and_payload(fields: dict, payload: bytes) -> None:
    with contextlib.suppress(*ALLOWED):
        P.unpack_array(fields, payload, max_raw_bytes=65536)


@settings(max_examples=150, deadline=None)
@given(st.lists(st.binary(min_size=1, max_size=64), max_size=40))
def test_random_stream_chunks_into_read_frame(chunks: list[bytes]) -> None:
    async def run() -> None:
        reader = asyncio.StreamReader()
        for chunk in chunks:
            reader.feed_data(chunk)
        reader.feed_eof()
        for _ in range(50):  # bounded: every call consumes input or ends
            try:
                await asyncio.wait_for(P.read_frame(reader, LIMITS), 2)
            except (P.ConnectionClosed, *ALLOWED):
                return

    asyncio.run(run())


def test_fuzzing_memory_is_bounded() -> None:
    """Hostile length fields must not cause large allocations."""
    hostile = [
        struct.pack("!4sII", P.MAGIC, 2**32 - 1, 2**32 - 1),
        struct.pack("!4sII", P.MAGIC, 4096, 2**31) + b"{" * 100,
        struct.pack("!4sII", P.MAGIC, 100, 0) + b"[" * 100,
    ]

    def decode_all() -> None:
        for data in hostile * 200:
            _decode(data)

    peak = peak_extra_bytes(decode_all)
    assert peak < 1024 * 1024
