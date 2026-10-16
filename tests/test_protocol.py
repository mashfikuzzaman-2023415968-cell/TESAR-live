"""Tests for mdc.common.protocol: framing, headers, arrays, integrity."""

from __future__ import annotations

import asyncio
import json
import struct
import zlib

import numpy as np
import pytest

from mdc.common import protocol as P
from tests.helpers.memory import peak_extra_bytes
from tests.helpers.messages import sample_messages, tile_array

SMALL = P.FrameLimits(max_header_bytes=2048, max_payload_bytes=4096, max_raw_array_bytes=8192)


def _frame(header: dict | bytes, payload: bytes = b"", magic: bytes = P.MAGIC) -> bytes:
    hb = header if isinstance(header, bytes) else json.dumps(header).encode()
    return struct.pack("!4sII", magic, len(hb), len(payload)) + hb + payload


def _reader(data: bytes, eof: bool = True) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    if eof:
        reader.feed_eof()
    return reader


# ------------------------------------------------------------- round trips


def test_message_type_constants_complete() -> None:
    assert set(sample_messages()) == P.MESSAGE_TYPES
    assert len(P.MESSAGE_TYPES) == 13


@pytest.mark.parametrize("msg_type", sorted(P.MESSAGE_TYPES))
def test_round_trip_every_type(msg_type: str) -> None:
    msg = sample_messages()[msg_type]
    decoded = P.decode_frame(P.encode_frame(msg))
    assert decoded.type == msg_type
    assert decoded.header == msg.header
    assert decoded.payload == msg.payload


@pytest.mark.parametrize("msg_type", sorted(P.MESSAGE_TYPES))
async def test_round_trip_every_type_via_stream(msg_type: str) -> None:
    msg = sample_messages()[msg_type]
    decoded = await P.read_frame(_reader(P.encode_frame(msg)))
    assert (decoded.type, decoded.header, decoded.payload) == (msg.type, msg.header, msg.payload)


@pytest.mark.parametrize("payload", [b"", b"x", bytes(range(256)) * 10])
def test_round_trip_with_and_without_payload(payload: bytes) -> None:
    msg = P.make_message("BYE", payload, reason="r")
    assert P.decode_frame(P.encode_frame(msg)).payload == payload


def test_encode_sets_type_and_version_from_message() -> None:
    msg = P.Message("PING", {"type": "FORGED", "v": 99, "seq": 1}, b"")
    header = P.decode_frame(P.encode_frame(msg)).header
    assert header["type"] == "PING"
    assert header["v"] == 1


def test_unknown_but_wellformed_type_passes_framing() -> None:
    # Forward compatibility: the validation layer decides about unknown types.
    msg = P.decode_frame(P.encode_frame(P.make_message("FUTURE_THING", x=1)))
    assert msg.type == "FUTURE_THING"


async def test_large_payload_10mb_stream() -> None:
    payload = np.random.default_rng(0).integers(0, 256, 10 * 1024 * 1024, np.uint8).tobytes()
    msg = P.make_message("PROBE", payload, probe_id="p", t_c=1.0)
    decoded = await P.read_frame(_reader(P.encode_frame(msg)))
    assert decoded.payload == payload


@pytest.mark.integration
async def test_large_payload_10mb_over_loopback_tcp() -> None:
    payload = np.random.default_rng(1).integers(0, 256, 10 * 1024 * 1024, np.uint8).tobytes()
    received: asyncio.Future[P.Message] = asyncio.get_running_loop().create_future()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        received.set_result(await P.read_frame(reader))
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    _, writer = await asyncio.open_connection("127.0.0.1", port)
    sent = await P.write_frame(
        writer, P.make_message("PROBE", payload, probe_id="p", t_c=1.0), asyncio.Lock()
    )
    msg = await asyncio.wait_for(received, 30)
    writer.close()
    await writer.wait_closed()
    server.close()
    await server.wait_closed()
    assert msg.payload == payload
    assert sent > len(payload)


async def test_byte_by_byte_delivery() -> None:
    msg = sample_messages()["TASK_ASSIGN"]
    data = P.encode_frame(msg)
    reader = asyncio.StreamReader()

    async def trickle() -> None:
        for i in range(len(data)):
            reader.feed_data(data[i : i + 1])
            if i % 97 == 0:
                await asyncio.sleep(0)
        reader.feed_eof()

    feeder = asyncio.create_task(trickle())
    decoded = await P.read_frame(reader)
    await feeder
    assert decoded.payload == msg.payload
    assert decoded.header == msg.header


async def test_consecutive_frames_in_one_buffer() -> None:
    msgs = list(sample_messages().values())
    reader = _reader(b"".join(P.encode_frame(m) for m in msgs))
    for msg in msgs:
        assert (await P.read_frame(reader)).header == msg.header
    with pytest.raises(P.ConnectionClosed):
        await P.read_frame(reader)


async def test_concurrent_writers_do_not_interleave() -> None:
    received: list[P.Message] = []
    done = asyncio.Event()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        for _ in range(40):
            received.append(await P.read_frame(reader))
        done.set()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    _, writer = await asyncio.open_connection("127.0.0.1", port)
    lock = asyncio.Lock()
    big = b"z" * 200_000
    await asyncio.gather(
        *(
            P.write_frame(writer, P.make_message("PROBE", big, probe_id=f"p{i}", t_c=1.0), lock)
            for i in range(40)
        )
    )
    await asyncio.wait_for(done.wait(), 30)
    writer.close()
    await writer.wait_closed()
    server.close()
    await server.wait_closed()
    assert sorted(m.header["probe_id"] for m in received) == sorted(f"p{i}" for i in range(40))
    assert all(m.payload == big for m in received)


# --------------------------------------------------------------- limits


def test_bad_magic_rejected() -> None:
    with pytest.raises(P.ProtocolError, match="magic"):
        P.decode_frame(_frame({"type": "PING", "v": 1}, magic=b"HTTP"))


def test_oversize_header_rejected_before_reading_body() -> None:
    prefix = struct.pack("!4sII", P.MAGIC, SMALL.max_header_bytes + 1, 0)
    with pytest.raises(P.ProtocolError, match="header length"):
        P.decode_frame(prefix + b"{}" * 2000, SMALL)


def test_oversize_payload_rejected() -> None:
    with pytest.raises(P.ProtocolError, match="payload length"):
        P.decode_frame(_frame({"type": "PING", "v": 1}, b"x" * 4097), SMALL)


def test_limits_boundary_values_accepted() -> None:
    header = {"type": "BYE", "v": 1, "reason": "x"}
    hb = json.dumps(header).encode()
    pad = SMALL.max_header_bytes - len(hb)
    header["reason"] = "x" * (1 + pad)
    frame = _frame(header, b"y" * SMALL.max_payload_bytes)
    assert len(P.decode_frame(frame, SMALL).payload) == SMALL.max_payload_bytes


async def test_stream_rejects_huge_claimed_lengths_without_allocating() -> None:
    reader = _reader(struct.pack("!4sII", P.MAGIC, 16, 2**32 - 1), eof=False)
    with pytest.raises(P.ProtocolError, match="payload length"):
        await P.read_frame(reader)
    reader = _reader(struct.pack("!4sII", P.MAGIC, 2**32 - 1, 0), eof=False)
    with pytest.raises(P.ProtocolError, match="header length"):
        await P.read_frame(reader)


def test_zero_length_header_rejected() -> None:
    with pytest.raises(P.ProtocolError):
        P.decode_frame(struct.pack("!4sII", P.MAGIC, 0, 0))


@pytest.mark.parametrize("cut", [0, 5, 11])
def test_short_frame_rejected(cut: int) -> None:
    with pytest.raises(P.ProtocolError):
        P.decode_frame(P.encode_frame(P.make_message("PING", seq=1, t_c=0.0))[:cut])


def test_length_mismatch_rejected() -> None:
    frame = P.encode_frame(P.make_message("PING", seq=1, t_c=0.0))
    with pytest.raises(P.ProtocolError, match="does not match"):
        P.decode_frame(frame + b"extra")
    with pytest.raises(P.ProtocolError, match="does not match"):
        P.decode_frame(frame[:-1])


async def test_clean_eof_is_connection_closed() -> None:
    with pytest.raises(P.ConnectionClosed):
        await P.read_frame(_reader(b""))


@pytest.mark.parametrize("cut", [3, 12, 20, -2])
async def test_eof_mid_frame_is_protocol_error(cut: int) -> None:
    data = P.encode_frame(P.make_message("BYE", b"payload", reason="x"))
    with pytest.raises(P.ProtocolError, match="inside a frame"):
        await P.read_frame(_reader(data[:cut]))


def test_encode_refuses_oversize_and_bad_messages() -> None:
    with pytest.raises(P.ProtocolError, match="header too large"):
        P.encode_frame(P.make_message("BYE", reason="x" * 5000), SMALL)
    with pytest.raises(P.ProtocolError, match="payload too large"):
        P.encode_frame(P.make_message("BYE", b"x" * 5000, reason="x"), SMALL)
    with pytest.raises(P.ProtocolError, match="invalid message type"):
        P.encode_frame(P.Message("lower", {}))
    with pytest.raises(P.ProtocolError, match="serializable"):
        P.encode_frame(P.make_message("BYE", reason=object()))
    with pytest.raises(P.ProtocolError, match="serializable"):
        P.encode_frame(P.make_message("PING", t_c=float("nan")))


# --------------------------------------------------------------- headers


@pytest.mark.parametrize(
    ("header_bytes", "fragment"),
    [
        (b"[1,2]", "object"),
        (b'"text"', "object"),
        (b'{"v":1}', "type"),
        (b'{"type":"ping","v":1}', "type"),
        (b'{"type":"' + b"A" * 40 + b'","v":1}', "type"),
        (b'{"type":5,"v":1}', "type"),
        (b'{"type":"PING"}', "version"),
        (b'{"type":"PING","v":2}', "version"),
        (b'{"type":"PING","v":true}', "version"),
        (b'{"type":"PING","v":1.0}', "version"),
        (b'{"type":"PING","v":"1"}', "version"),
        (b'\xff\xfe{"type":"PING"}', "invalid JSON"),
        (b'{"type":"PING","v":1', "invalid JSON"),
        (b'{"type":"PING","v":1,"x":NaN}', "non-finite"),
        (b'{"type":"PING","v":1,"x":Infinity}', "non-finite"),
        (b'{"type":"PING","v":1,"x":-Infinity}', "non-finite"),
        (b'{"type":"PING","v":1,"x":1e999}', "non-finite"),
        (b'{"type":"PING","v":1,"x":' + b"9" * 21 + b"}", "too long"),
        (b'{"type":"PING","v":1,"type":"PONG"}', "duplicate"),
    ],
)
def test_bad_headers_rejected(header_bytes: bytes, fragment: str) -> None:
    with pytest.raises(P.ProtocolError, match=fragment):
        P.decode_frame(_frame(header_bytes))


def test_nesting_depth_limit() -> None:
    def nested(depth: int) -> dict:
        value: object = 1
        for _ in range(depth - 1):
            value = {"a": value}
        return {"type": "BYE", "v": 1, "reason": "x", "deep": value}

    P.decode_frame(_frame(nested(8)))  # header object itself is level 1
    with pytest.raises(P.ProtocolError, match="nesting"):
        P.decode_frame(_frame(nested(9)))


def test_extremely_deep_nesting_is_protocol_error() -> None:
    body = b'{"type":"BYE","v":1,"x":' + b"[" * 5000 + b"]" * 5000 + b"}"
    with pytest.raises(P.ProtocolError):
        P.decode_frame(_frame(body), P.FrameLimits(max_header_bytes=20000))


def test_json_depth_helper() -> None:
    assert P.json_depth(1) == 0
    assert P.json_depth({}) == 1
    assert P.json_depth({"a": [1, {"b": []}]}) == 4


def test_strict_json_accepts_normal_values() -> None:
    assert P.strict_json_loads('{"a": -12, "b": 1.5e3, "c": [true, null, "x"]}') == {
        "a": -12,
        "b": 1500.0,
        "c": [True, None, "x"],
    }
    assert P.strict_json_loads(b'"' + "é".encode() + b'"') == "é"


# ---------------------------------------------------------------- arrays


@pytest.mark.parametrize("dtype", ["uint8", "int32", "int64"])
@pytest.mark.parametrize("compression", ["zlib", "none"])
@pytest.mark.parametrize("shape", [(7,), (13, 17), (2, 3, 4), (1, 1)])
def test_pack_unpack_round_trip(dtype: str, compression: str, shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(3)
    info = np.iinfo(dtype)
    array = rng.integers(max(info.min, -(2**40)), min(info.max, 2**40), size=shape, dtype=dtype)
    fields, payload = P.pack_array(array, compression)
    assert fields["shape"] == list(shape)
    assert fields["dtype"] == dtype
    assert fields["raw_len"] == array.nbytes
    out = P.unpack_array(fields, payload)
    assert out.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(out, array)


def test_pack_is_little_endian_and_hash_of_raw_bytes() -> None:
    array = np.array([1, 256], dtype=">i4")  # big-endian input
    fields, payload = P.pack_array(array, "none")
    assert payload == b"\x01\x00\x00\x00\x00\x01\x00\x00"
    assert fields["sha256"] == P.sha256_hex(payload)
    np.testing.assert_array_equal(P.unpack_array(fields, payload), [1, 256])


def test_pack_non_contiguous_array() -> None:
    array = tile_array(5, (20, 30))[:, ::2]
    fields, payload = P.pack_array(array)
    np.testing.assert_array_equal(P.unpack_array(fields, payload), array)


def test_unpacked_array_is_read_only() -> None:
    fields, payload = P.pack_array(tile_array(1))
    out = P.unpack_array(fields, payload)
    with pytest.raises(ValueError):
        out[0, 0] = 1


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((3, 3), dtype=np.float64),
        np.zeros((3, 3), dtype=np.int16),
        np.zeros((2, 2, 2, 2), dtype=np.uint8),
        np.zeros((0, 3), dtype=np.uint8),
        np.zeros((), dtype=np.uint8),
    ],
)
def test_pack_rejects_disallowed_arrays(array: np.ndarray) -> None:
    with pytest.raises(ValueError):
        P.pack_array(array)


def test_pack_rejects_unknown_compression() -> None:
    with pytest.raises(ValueError):
        P.pack_array(tile_array(), "gzip")


def test_tampered_uncompressed_payload_is_integrity_error() -> None:
    fields, payload = P.pack_array(tile_array(), "none")
    tampered = bytes([payload[0] ^ 1]) + payload[1:]
    with pytest.raises(P.IntegrityError):
        P.unpack_array(fields, tampered)


def test_tampered_hash_field_is_integrity_error() -> None:
    fields, payload = P.pack_array(tile_array())
    fields["sha256"] = "0" * 64
    with pytest.raises(P.IntegrityError):
        P.unpack_array(fields, payload)


def test_integrity_error_is_not_a_protocol_error() -> None:
    # Transport corruption must be distinguishable from a protocol violation.
    assert not issubclass(P.IntegrityError, P.ProtocolError)
    assert issubclass(P.ValidationError, P.ProtocolError)


def test_tampered_compressed_payload_rejected() -> None:
    fields, payload = P.pack_array(tile_array())
    tampered = payload[:10] + bytes([payload[10] ^ 0xFF]) + payload[11:]
    with pytest.raises((P.ValidationError, P.IntegrityError)):
        P.unpack_array(fields, tampered)


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        ({"shape": [40]}, "raw_len"),
        ({"shape": []}, "shape"),
        ({"shape": [1, 1, 1, 1]}, "shape"),
        ({"shape": [0, 50]}, "dimension"),
        ({"shape": [40, 16385]}, "dimension"),
        ({"shape": [40, True]}, "dimension"),
        ({"shape": "40x50"}, "shape"),
        ({"dtype": "float64"}, "dtype"),
        ({"dtype": None}, "dtype"),
        ({"comp": "gzip"}, "comp"),
        ({"raw_len": 1}, "raw_len"),
        ({"raw_len": 2000.0}, "raw_len"),
        ({"sha256": "XYZ"}, "sha256"),
        ({"sha256": "A" * 64}, "sha256"),
    ],
)
def test_bad_array_fields_rejected(change: dict, fragment: str) -> None:
    fields, payload = P.pack_array(tile_array())
    fields.update(change)
    with pytest.raises(P.ValidationError, match=fragment):
        P.unpack_array(fields, payload)


def test_raw_len_limit_enforced() -> None:
    fields, payload = P.pack_array(tile_array(shape=(100, 100)))
    with pytest.raises(P.ValidationError, match="exceeds limit"):
        P.unpack_array(fields, payload, max_raw_bytes=9999)
    P.unpack_array(fields, payload, max_raw_bytes=10000)  # boundary accepted


def test_uncompressed_length_mismatch_rejected() -> None:
    fields, payload = P.pack_array(tile_array(), "none")
    with pytest.raises(P.ValidationError, match="length"):
        P.unpack_array(fields, payload + b"\x00")


def test_decompression_bomb_rejected_with_bounded_output() -> None:
    bomb = zlib.compress(b"\x00" * (64 * 1024 * 1024), 9)  # ~64 KB on the wire
    assert len(bomb) < 100_000
    fields = {
        "shape": [10, 10],
        "dtype": "uint8",
        "comp": "zlib",
        "raw_len": 100,
        "sha256": "0" * 64,
    }

    def attempt() -> None:
        with pytest.raises(P.ValidationError, match="raw_len"):
            P.unpack_array(fields, bomb)

    peak = peak_extra_bytes(attempt)
    assert peak < 2 * 1024 * 1024  # never inflated to 64 MB


@pytest.mark.parametrize(
    ("data", "fragment"),
    [
        (zlib.compress(b"a" * 50), "raw_len"),  # too short
        (zlib.compress(b"a" * 100) + b"junk", "extra data"),  # trailing garbage
        (zlib.compress(b"a" * 100)[:-4], "incomplete"),  # truncated (missing checksum)
        (b"not zlib at all", "corrupt"),
    ],
)
def test_safe_decompress_rejects_malformed_streams(data: bytes, fragment: str) -> None:
    with pytest.raises(P.ValidationError, match=fragment):
        P.safe_decompress(data, 100)


def test_safe_decompress_exact() -> None:
    assert P.safe_decompress(zlib.compress(b"q" * 100), 100) == b"q" * 100


def test_sha256_hex_known_value() -> None:
    assert P.sha256_hex(b"") == ("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")


def test_make_message() -> None:
    msg = P.make_message("PING", seq=1, t_c=2.0)
    assert msg.header == {"type": "PING", "v": 1, "seq": 1, "t_c": 2.0}
    assert msg.payload == b""
