"""Tests for small common helpers: EWMA, ids, time, events, logging."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from mdc.common.events import EventLogger, read_events, to_jsonable
from mdc.common.ewma import EWMA
from mdc.common.ids import ID_RE, is_valid_id, new_id
from mdc.common.logging_setup import LOG_FORMAT, parse_level, setup_logging, teardown_logging
from mdc.common.timeutil import elapsed_since, monotonic, wait_event, wall_iso
from tests.helpers.procs import child_env

# ------------------------------------------------------------------- EWMA


def test_ewma_first_sample_initialises() -> None:
    e = EWMA(0.3)
    assert e.value is None
    assert e.count == 0
    assert e.update(10.0) == 10.0
    assert e.count == 1


def test_ewma_formula() -> None:
    e = EWMA(0.25)
    e.update(8.0)
    assert e.update(4.0) == pytest.approx(0.25 * 4 + 0.75 * 8)


def test_ewma_initial_value_is_used_as_prior() -> None:
    e = EWMA(0.5, initial=100.0)
    assert e.value == 100.0
    assert e.count == 0
    assert e.update(0.0) == 50.0


def test_ewma_alpha_one_tracks_last_sample() -> None:
    e = EWMA(1.0)
    for x in (3.0, 7.0, 1.0):
        e.update(x)
    assert e.value == 1.0


def test_ewma_stays_within_sample_bounds() -> None:
    e = EWMA(0.3)
    samples = [5.0, 1.0, 9.0, 3.0, 7.0, 2.0]
    for x in samples:
        e.update(x)
        assert min(samples) <= (e.value or 0) <= max(samples)


def test_ewma_converges_to_constant() -> None:
    e = EWMA(0.3, initial=0.0)
    for _ in range(100):
        e.update(42.0)
    assert e.value == pytest.approx(42.0)


def test_ewma_value_or() -> None:
    assert EWMA(0.3).value_or(7.0) == 7.0
    e = EWMA(0.3)
    e.update(2.0)
    assert e.value_or(7.0) == 2.0
    assert "EWMA(alpha=0.3" in repr(e)


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.0001, math.nan])
def test_ewma_rejects_bad_alpha(alpha: float) -> None:
    with pytest.raises(ValueError):
        EWMA(alpha)


@pytest.mark.parametrize("alpha", [True, "0.3", None])
def test_ewma_rejects_non_number_alpha(alpha: object) -> None:
    with pytest.raises(TypeError):
        EWMA(alpha)  # type: ignore[arg-type]


@pytest.mark.parametrize("x", [math.nan, math.inf, -math.inf])
def test_ewma_rejects_non_finite_samples(x: float) -> None:
    e = EWMA(0.3)
    e.update(1.0)
    with pytest.raises(ValueError):
        e.update(x)
    assert e.value == 1.0  # state unchanged after a rejected sample
    assert e.count == 1


@pytest.mark.parametrize("x", [True, "1", None])
def test_ewma_rejects_non_number_samples(x: object) -> None:
    with pytest.raises(TypeError):
        EWMA(0.3).update(x)  # type: ignore[arg-type]


def test_ewma_rejects_non_finite_initial() -> None:
    with pytest.raises(ValueError):
        EWMA(0.3, initial=math.inf)


# -------------------------------------------------------------------- ids


def test_new_id_format_and_uniqueness() -> None:
    ids = {new_id("att") for _ in range(2000)}
    assert len(ids) == 2000
    for value in ids:
        assert ID_RE.fullmatch(value)
        assert re.fullmatch(r"att_[0-9a-f]{16}", value)


def test_new_id_length_parameter() -> None:
    assert len(new_id("s", nbytes=16)) == len("s_") + 32


@pytest.mark.parametrize("prefix", ["", "has space", "x" * 25, "a_b", "ü"])
def test_new_id_rejects_bad_prefix(prefix: str) -> None:
    with pytest.raises(ValueError):
        new_id(prefix)


@pytest.mark.parametrize("nbytes", [0, 3, 17])
def test_new_id_rejects_bad_nbytes(nbytes: int) -> None:
    with pytest.raises(ValueError):
        new_id("x", nbytes=nbytes)


@pytest.mark.parametrize(
    ("value", "ok"),
    [
        ("abc", True),
        ("a-b_C9", True),
        ("x" * 64, True),
        ("x" * 65, False),
        ("", False),
        ("a b", False),
        ("a\n", False),
        (123, False),
        (None, False),
    ],
)
def test_is_valid_id(value: object, ok: bool) -> None:
    assert is_valid_id(value) is ok


# ------------------------------------------------------------------- time


def test_time_helpers() -> None:
    t0 = monotonic()
    assert elapsed_since(t0) >= 0
    stamp = wall_iso()
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}\+00:00", stamp)


# ----------------------------------------------------------------- events


def test_events_jsonl_format(tmp_path: Path) -> None:
    path = tmp_path / "out" / "events.jsonl"
    with EventLogger(path) as ev:
        ev.emit("worker_connected", worker_id="w1", slots=2, battery={"pct": 80.0})
        ev.emit("job_finished", job_id="j1", makespan_s=1.5, output_correct=True)
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert list(first)[:3] == ["t", "wall", "type"]
    assert first["type"] == "worker_connected"
    assert first["battery"] == {"pct": 80.0}
    assert isinstance(first["t"], float)
    assert first["t"] >= 0
    events = read_events(path)
    assert [e["type"] for e in events] == ["worker_connected", "job_finished"]
    assert events[1]["t"] >= events[0]["t"]


def test_events_appends_to_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    with EventLogger(path) as ev:
        ev.emit("a")
    with EventLogger(path) as ev:
        ev.emit("b")
    assert [e["type"] for e in read_events(path)] == ["a", "b"]


def test_events_memory_only_and_recent() -> None:
    ev = EventLogger(recent_max=3)
    for i in range(5):
        ev.emit("tick", i=i)
    assert [e["i"] for e in ev.recent()] == [2, 3, 4]
    assert [e["i"] for e in ev.recent(2)] == [3, 4]
    assert ev.recent(0) == []
    ev.close()
    ev.close()  # idempotent


def test_events_reset_clock() -> None:
    ev = EventLogger()
    old_t0 = ev.t0
    ev.reset_clock()
    assert ev.t0 >= old_t0
    assert ev.emit("x")["t"] < 1.0


def test_events_open_switches_file(tmp_path: Path) -> None:
    ev = EventLogger(tmp_path / "a.jsonl")
    ev.emit("one")
    ev.open(tmp_path / "b.jsonl")
    ev.emit("two")
    ev.close()
    assert [e["type"] for e in read_events(tmp_path / "a.jsonl")] == ["one"]
    assert [e["type"] for e in read_events(tmp_path / "b.jsonl")] == ["two"]


@pytest.mark.parametrize("bad_type", ["", "Upper", "has space", "1abc", "x" * 65, "a-b", None])
def test_events_reject_bad_type(bad_type: object) -> None:
    with pytest.raises(ValueError):
        EventLogger().emit(bad_type)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["t", "wall"])
def test_events_reject_reserved_fields(field: str) -> None:
    with pytest.raises(ValueError, match="override"):
        EventLogger().emit("x", **{field: 1})


def test_events_sanitize_values(tmp_path: Path) -> None:
    class NumpyLike:
        def item(self) -> float:
            return 2.5

    class BadItem:
        def item(self) -> float:
            raise TypeError("nope")

        def __str__(self) -> str:
            return "bad-item"

    path = tmp_path / "e.jsonl"
    with EventLogger(path) as ev:
        ev.emit(
            "values",
            nan=math.nan,
            inf=math.inf,
            tup=(1, 2),
            st={3},
            np=NumpyLike(),
            bad=BadItem(),
            nested={1: [math.nan]},
            obj=object,
        )
    (event,) = read_events(path)
    assert event["nan"] is None
    assert event["inf"] is None
    assert event["tup"] == [1, 2]
    assert event["st"] == [3]
    assert event["np"] == 2.5
    assert event["bad"] == "bad-item"
    assert event["nested"] == {"1": [None]}
    assert event["obj"].startswith("<class")


def test_to_jsonable_depth_limit() -> None:
    deep: list[object] = []
    node = deep
    for _ in range(50):
        child: list[object] = []
        node.append(child)
        node = child
    text = json.dumps(to_jsonable(deep))
    assert "<too deep>" in text


def test_events_log_injection_is_escaped(tmp_path: Path) -> None:
    path = tmp_path / "e.jsonl"
    with EventLogger(path) as ev:
        ev.emit("worker_connected", name='evil\n{"type":"forged"}')
    lines = path.read_text().splitlines()
    assert len(lines) == 1  # newline in a field cannot create a fake event line
    assert read_events(path)[0]["name"] == 'evil\n{"type":"forged"}'


def test_events_subscribers() -> None:
    ev = EventLogger()
    seen: list[str] = []

    def callback(event: dict) -> None:
        seen.append(event["type"])

    ev.subscribe(callback)
    ev.emit("a")
    ev.unsubscribe(callback)
    ev.unsubscribe(callback)  # unknown callback ignored
    ev.emit("b")
    assert seen == ["a"]


def test_events_broken_subscriber_does_not_break_emit(caplog: pytest.LogCaptureFixture) -> None:
    ev = EventLogger()
    good: list[str] = []

    def broken(event: dict) -> None:
        raise RuntimeError("dashboard socket closed")

    ev.subscribe(broken)
    ev.subscribe(lambda e: good.append(e["type"]))
    with caplog.at_level(logging.ERROR):
        ev.emit("still_works")
    assert good == ["still_works"]
    assert "subscriber" in caplog.text


def test_events_subscriber_may_emit_without_deadlock() -> None:
    ev = EventLogger()

    def echo(event: dict) -> None:
        if event["type"] == "ping":
            ev.emit("pong")

    ev.subscribe(echo)
    ev.emit("ping")
    assert [e["type"] for e in ev.recent()] == ["ping", "pong"]


def test_events_thread_safety(tmp_path: Path) -> None:
    path = tmp_path / "e.jsonl"
    ev = EventLogger(path, recent_max=10000)

    def worker(n: int) -> None:
        for i in range(200):
            ev.emit("tick", thread=n, i=i)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ev.close()
    events = read_events(path)
    assert len(events) == 1600  # no interleaved/corrupt lines
    assert len(ev.recent()) == 1600


def test_read_events_skips_blank_and_non_objects(tmp_path: Path) -> None:
    path = tmp_path / "e.jsonl"
    path.write_text('{"type":"a"}\n\n[1,2]\n{"type":"b"}\n')
    assert [e["type"] for e in read_events(path)] == ["a", "b"]


# ---------------------------------------------------------------- logging


def test_setup_logging_console_and_rotating_file(tmp_path: Path) -> None:
    logfile = tmp_path / "logs" / "coord.log"
    try:
        root = setup_logging("DEBUG", logfile, max_bytes=2000, backups=2)
        assert root.level == logging.DEBUG
        for i in range(200):
            logging.getLogger("mdc.test").info("line %d %s", i, "x" * 40)
        for handler in root.handlers:
            handler.flush()
        files = sorted(p.name for p in logfile.parent.iterdir())
        assert files == ["coord.log", "coord.log.1", "coord.log.2"]  # rotation is bounded
        assert re.search(r" INFO \[mdc\.test\] line 199 ", logfile.read_text())
        # Calling again replaces (does not duplicate) our handlers.
        setup_logging("INFO")
        ours = [h for h in root.handlers if getattr(h, "_mdc_handler", False)]
        assert len(ours) == 1
    finally:
        teardown_logging()
    assert not [h for h in logging.getLogger().handlers if getattr(h, "_mdc_handler", False)]


def test_log_format_matches_spec() -> None:
    assert LOG_FORMAT == "%(asctime)s %(levelname)s [%(name)s] %(message)s"


@pytest.mark.parametrize(("value", "expected"), [("info", 20), (" DEBUG ", 10), (30, 30)])
def test_parse_level(value: str | int, expected: int) -> None:
    assert parse_level(value) == expected


@pytest.mark.parametrize("value", ["verbose", "", True])
def test_parse_level_rejects_junk(value: object) -> None:
    with pytest.raises(ValueError):
        parse_level(value)  # type: ignore[arg-type]


# ------------------------------------------------------ worker-side imports


def test_worker_side_common_modules_do_not_load_yaml() -> None:
    """Phones have only numpy: the shared modules a worker uses must not pull
    in PyYAML (config.py is the documented coordinator-only exception)."""
    code = (
        "import sys\n"
        "import mdc.common.ewma, mdc.common.events, mdc.common.ids\n"
        "import mdc.common.timeutil, mdc.common.logging_setup, mdc.common.statedir\n"
        "import mdc.common.protocol, mdc.common.validation, mdc.common.auth, mdc.common.tls\n"
        "import mdc.common.credentials, mdc.common.redact, mdc.common.safe_names\n"
        "import mdc.tasks.base, mdc.tasks.benchmark, mdc.tasks.tiling\n"
        "bad = [m for m in ('yaml', 'PIL', 'aiohttp', 'cryptography') if m in sys.modules]\n"
        "print(','.join(bad))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
        env=child_env(),
        cwd="/",  # prove it does not rely on the current directory
    )
    assert proc.stdout.strip() == ""


# ------------------------------------------------------ cancellation-safe waits


async def test_wait_event_returns_whether_set() -> None:
    event = asyncio.Event()
    assert await wait_event(event, 0.01) is False
    asyncio.get_running_loop().call_later(0.01, event.set)
    assert await wait_event(event, 5.0) is True


async def test_wait_event_never_swallows_cancellation() -> None:
    """Regression (Python 3.10 hang): cancelling a loop built on wait_event must
    always end it, whatever moment the cancellation hits."""

    async def loop_forever(event: asyncio.Event) -> None:
        while True:
            await wait_event(event, 0.001)

    for i in range(300):
        event = asyncio.Event()
        task = asyncio.create_task(loop_forever(event))
        await asyncio.sleep(0.001 * (i % 3))
        event.set()  # the inner wait completes...
        task.cancel()  # ...and the cancellation arrives right after (the 3.10 trap)
        done, _ = await asyncio.wait({task}, timeout=1.0)
        assert task in done, f"cancellation was swallowed on iteration {i}"
