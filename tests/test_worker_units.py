"""Unit tests for worker components: battery, faults, shaper, discovery, helpers."""

from __future__ import annotations

import asyncio
import json
import random
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mdc.common.validation import TaskLimits, validate_json_result
from mdc.tasks.row_stats import row_stats
from mdc.tasks.synthetic import hash_chain
from mdc.worker import battery as B
from mdc.worker import discovery_listener as D
from mdc.worker.faults import FaultInjector, FaultSettings, corrupt_array, corrupt_json
from mdc.worker.shaper import Shaper
from mdc.worker.worker import detect_platform, stricter_limits

FP = "sha256:" + "ab" * 32

# ---------------------------------------------------------------- Termux


@pytest.mark.parametrize(
    ("output", "pct", "charging", "current_ma"),
    [
        (
            '{"percentage": 87, "status": "DISCHARGING", "plugged": "UNPLUGGED", '
            '"current": -412000}',
            87.0,
            False,
            -412.0,
        ),
        # Some devices report a positive current while discharging: sign normalized.
        (
            '{"percentage": 50, "status": "DISCHARGING", "plugged": "UNPLUGGED", '
            '"current": 300000}',
            50.0,
            False,
            -300.0,
        ),
        (
            '{"percentage": 60, "status": "CHARGING", "plugged": "PLUGGED_AC", "current": -900000}',
            60.0,
            True,
            900.0,
        ),
        ('{"percentage": 100, "status": "FULL", "plugged": "PLUGGED_USB"}', 100.0, True, None),
        (
            '{"percentage": 40, "status": "UNKNOWN", "plugged": "UNPLUGGED", "current": -5000}',
            40.0,
            False,
            -5.0,
        ),
        ('{"percentage": 140, "status": "WEIRD", "plugged": ""}', 100.0, None, None),
        ('{"status": "NOT_CHARGING"}', None, False, None),
    ],
)
def test_parse_termux(
    output: str, pct: float | None, charging: bool | None, current_ma: float | None
) -> None:
    reading = B.parse_termux(output)
    assert reading["pct"] == pct
    assert reading["charging"] is charging
    assert reading["current_ma"] == current_ma
    assert reading["source"] == "termux"


def test_parse_termux_voltage_units() -> None:
    assert B.parse_termux('{"voltage": 3850}')["voltage_v"] == 3.85
    assert B.parse_termux('{"voltage": 3.9}')["voltage_v"] == 3.9


@pytest.mark.parametrize("output", ["[]", "not json", '"x"'])
def test_parse_termux_rejects_garbage(output: str) -> None:
    with pytest.raises(ValueError):
        B.parse_termux(output)


def _fake_runner(stdout: str = "", returncode: int = 0, exc: Exception | None = None):
    calls: list[tuple[list[str], dict]] = []

    def runner(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((args, kwargs))
        if exc is not None:
            raise exc
        return subprocess.CompletedProcess(args, returncode, stdout, "")

    return runner, calls


def test_termux_battery_success_and_cached_fallback(caplog: pytest.LogCaptureFixture) -> None:
    runner, calls = _fake_runner('{"percentage": 77, "status": "DISCHARGING", "current": -1000}')
    bat = B.TermuxBattery("/data/data/com.termux/files/usr/bin/termux-battery-status", runner)
    assert bat.read()["pct"] == 77.0
    args, kwargs = calls[0]
    assert args == ["/data/data/com.termux/files/usr/bin/termux-battery-status"]
    assert kwargs["timeout"] == 5.0 and "shell" not in kwargs
    # Later failures return the last good value (warning logged once).
    for failing in (
        _fake_runner(exc=subprocess.TimeoutExpired("x", 5))[0],
        _fake_runner(returncode=1)[0],
        _fake_runner("garbage")[0],
    ):
        bat._runner = failing
        assert bat.read()["pct"] == 77.0
    assert caplog.text.count("termux battery reading failed") == 1


def test_termux_battery_missing_command() -> None:
    bat = B.TermuxBattery()
    bat.command_path = None
    reading = bat.read()
    assert reading["pct"] is None and reading["source"] == "termux"
    bat2 = B.TermuxBattery("/nope", _fake_runner(exc=FileNotFoundError())[0])
    assert bat2.read()["pct"] is None


# ----------------------------------------------------------------- sysfs


def _fake_sysfs(root: Path, **files: str) -> Path:
    bat = root / "BAT0"
    bat.mkdir(parents=True)
    for name, content in files.items():
        (bat / name).write_text(content + "\n")
    return bat


def test_sysfs_current_now(tmp_path: Path) -> None:
    path = _fake_sysfs(
        tmp_path, capacity="66", status="Discharging", current_now="1545000", voltage_now="11189000"
    )
    reading = B.SysfsBattery(path, root=tmp_path).read()
    assert reading == {
        **reading,
        "pct": 66.0,
        "charging": False,
        "current_ma": -1545.0,
        "voltage_v": 11.189,
        "source": "sysfs",
    }
    assert B.find_sysfs_battery(tmp_path) == path


def test_sysfs_power_now_variant(tmp_path: Path) -> None:
    path = _fake_sysfs(
        tmp_path, capacity="80", status="Charging", power_now="15000000", voltage_now="12000000"
    )
    reading = B.SysfsBattery(path, root=tmp_path).read()
    assert reading["charging"] is True
    assert reading["current_ma"] == pytest.approx(1250.0)  # 15 W / 12 V


def test_sysfs_missing_and_broken(tmp_path: Path) -> None:
    assert B.find_sysfs_battery(tmp_path / "nothing") is None
    assert B.find_sysfs_battery(tmp_path) is None
    path = _fake_sysfs(tmp_path, capacity="garbage")
    reading = B.SysfsBattery(path, root=tmp_path).read()
    assert reading["pct"] is None and reading["charging"] is None
    assert B.SysfsBattery.__new__(B.SysfsBattery).__class__ is B.SysfsBattery
    empty = B.SysfsBattery(None)
    empty.path = None
    assert empty.read()["pct"] is None


# ------------------------------------------------------------------ mock


def test_mock_battery_drains_more_when_busy() -> None:
    now = [0.0]
    bat = B.MockBattery(
        80.0, drain_pct_per_min=1.0, busy_extra_pct_per_min=2.0, clock=lambda: now[0]
    )
    now[0] = 60.0
    assert bat.read()["pct"] == pytest.approx(79.0)
    bat.set_busy(2)
    now[0] = 120.0
    reading = bat.read()
    assert reading["pct"] == pytest.approx(74.0)  # 1 + 2*2 per minute
    assert reading["current_ma"] == -(150.0 + 2 * 600.0)
    bat.charging = True
    now[0] = 600.0
    assert bat.read()["pct"] == pytest.approx(74.0)  # no drain while charging
    assert bat.read()["current_ma"] > 0


def test_mock_battery_never_negative_and_validates() -> None:
    now = [0.0]
    bat = B.MockBattery(1.0, 100.0, clock=lambda: now[0])
    now[0] = 600.0
    assert bat.read()["pct"] == 0.0
    with pytest.raises(ValueError):
        B.MockBattery(101.0)
    with pytest.raises(ValueError):
        B.MockBattery(50.0, -1.0)


def test_make_battery(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(B.make_battery("mock", 55.0), B.MockBattery)
    assert isinstance(B.make_battery("sysfs"), B.SysfsBattery)
    assert isinstance(B.make_battery("termux"), B.TermuxBattery)
    with pytest.raises(ValueError):
        B.make_battery("solar")
    monkeypatch.setattr(B.TermuxBattery, "available", staticmethod(lambda: True))
    assert isinstance(B.make_battery("auto"), B.TermuxBattery)
    monkeypatch.setattr(B.TermuxBattery, "available", staticmethod(lambda: False))
    monkeypatch.setattr(B, "find_sysfs_battery", lambda root=None: None)
    assert isinstance(B.make_battery("auto"), B.MockBattery)


@pytest.mark.parametrize(
    ("reading", "allowed"),
    [
        ({"pct": 50.0, "charging": False}, True),
        ({"pct": 15.0, "charging": False}, True),  # boundary
        ({"pct": 14.9, "charging": False}, False),
        ({"pct": 5.0, "charging": True}, True),
        ({"pct": None, "charging": None}, True),
    ],
)
def test_battery_allows_work(reading: dict, allowed: bool) -> None:
    assert B.battery_allows_work(reading, 15.0) is allowed


# ---------------------------------------------------------------- faults


def test_corrupt_array_is_plausible() -> None:
    rng = random.Random(1)  # noqa: S311 - test data
    for trial in range(30):
        original = np.random.default_rng(trial).integers(0, 256, size=(64, 80), dtype=np.uint8)
        corrupted = corrupt_array(original, rng)
        assert corrupted.shape == original.shape and corrupted.dtype == np.uint8
        changed = np.count_nonzero(corrupted != original)
        fraction = changed / original.size
        assert 0.02 < fraction <= 0.25  # a 5-20 % rectangle (rounding at the edges)
        diff = (corrupted.astype(int) - original.astype(int)) % 256
        assert set(np.unique(diff[diff != 0])).issubset(set(range(20, 81)))
    tiny = np.zeros((1, 1), dtype=np.uint8)
    assert corrupt_array(tiny, rng)[0, 0] != 0


def test_corrupt_array_does_not_modify_input() -> None:
    original = np.zeros((10, 10), dtype=np.uint8)
    corrupt_array(original, random.Random(0))  # noqa: S311
    assert not original.any()


def test_corrupt_json_stays_valid_but_differs() -> None:
    rng = random.Random(2)  # noqa: S311 - test data
    for seed in range(40):
        strip = np.random.default_rng(seed).integers(0, 256, size=(3, seed + 1), dtype=np.uint8)
        good = row_stats(strip, 128)
        bad = corrupt_json("row_stats", good, rng)
        assert bad != good
        validate_json_result("row_stats", bad)  # still passes the schema
    good_digest = {"digest": hash_chain(1, 3)}
    bad_digest = corrupt_json("synthetic", good_digest, rng)
    assert bad_digest != good_digest
    validate_json_result("synthetic", bad_digest)
    assert corrupt_json("other", {"a": 1}, rng) == {"a": 1}


def test_fault_injector_probabilities_are_seeded() -> None:
    a = FaultInjector(FaultSettings(drop_prob=0.3), seed=7)
    b = FaultInjector(FaultSettings(drop_prob=0.3), seed=7)
    seq_a = [a.should_drop() for _ in range(200)]
    assert seq_a == [b.should_drop() for _ in range(200)]
    assert 30 < sum(seq_a) < 90
    never = FaultInjector(FaultSettings())
    assert not any(
        never.should_drop() or never.should_crash() or never.should_corrupt() for _ in range(100)
    )
    always = FaultInjector(FaultSettings(crash_prob=1.0, malicious_prob=1.0))
    assert always.should_crash() and always.should_corrupt()


def test_fault_timing_helpers() -> None:
    inj = FaultInjector(FaultSettings(slowdown=3.0, flaky_mean_s=10.0), seed=3)
    assert inj.slowdown_sleep_s(2.0) == pytest.approx(4.0)
    samples = [inj.flaky_session_s() for _ in range(2000)]
    assert all(s is not None and s > 0 for s in samples)
    assert 8.0 < sum(samples) / len(samples) < 12.0  # mean ~ 10
    assert all(2.0 <= inj.flaky_reconnect_s() <= 10.0 for _ in range(100))
    assert FaultInjector(FaultSettings()).flaky_session_s() is None
    assert FaultInjector(FaultSettings()).slowdown_sleep_s(5.0) == 0.0


@pytest.mark.parametrize(
    "settings",
    [
        FaultSettings(drop_prob=1.5),
        FaultSettings(crash_prob=-0.1),
        FaultSettings(slowdown=0.5),
        FaultSettings(crash_after_s=-1),
        FaultSettings(flaky_mean_s=0),
    ],
)
def test_fault_settings_validated(settings: FaultSettings) -> None:
    with pytest.raises(ValueError):
        FaultInjector(settings)


def test_fault_settings_any_enabled_and_control() -> None:
    assert not FaultSettings().any_enabled()
    assert FaultSettings(slowdown=2.0).any_enabled()
    inj = FaultInjector(FaultSettings())
    inj.apply_control("set_malicious", 0.3)
    inj.apply_control("set_drop", 0.2)
    inj.apply_control("set_slowdown", 3.0)
    assert (inj.settings.malicious_prob, inj.settings.drop_prob, inj.settings.slowdown) == (
        0.3,
        0.2,
        3.0,
    )
    with pytest.raises(ValueError):
        inj.apply_control("set_slowdown", 0.1)


# ---------------------------------------------------------------- shaper


def test_shaper_math() -> None:
    assert Shaper().delay_s(10**6) == 0.0
    assert not Shaper().enabled
    assert Shaper(delay_ms=40).delay_s(0) == pytest.approx(0.04)
    # 2000 kbit/s: 250 000 bytes = 2 000 000 bits -> 1 s
    assert Shaper(bw_kbps=2000).delay_s(250_000) == pytest.approx(1.0)
    assert Shaper(delay_ms=40, bw_kbps=2000).delay_s(250_000) == pytest.approx(1.04)
    assert Shaper(bw_kbps=1).enabled
    with pytest.raises(ValueError):
        Shaper(delay_ms=-1)
    with pytest.raises(ValueError):
        Shaper(bw_kbps=0)
    with pytest.raises(ValueError):
        Shaper().delay_s(-1)


# ------------------------------------------------------------- discovery


def beacon(**changes: object) -> bytes:
    data = {
        "type": "MDC_BEACON",
        "v": 1,
        "host": "192.168.0.10",
        "port": 50010,
        "coord_id": "c1",
        "name": "lab-coordinator",
        "fp": FP,
    }
    data.update(changes)
    data = {k: v for k, v in data.items() if v is not _DROP}
    return json.dumps(data).encode()


_DROP = object()


def test_parse_valid_beacon() -> None:
    parsed = D.parse_beacon(beacon(), FP)
    assert parsed == D.Beacon("192.168.0.10", 50010, "c1", "lab-coordinator")
    assert D.parse_beacon(beacon(coord_id=_DROP, name=_DROP), FP) == D.Beacon(
        "192.168.0.10", 50010, "", ""
    )


@pytest.mark.parametrize(
    "data",
    [
        beacon(fp="sha256:" + "cd" * 32),  # different coordinator
        beacon(fp="not a fingerprint"),
        beacon(fp=_DROP),
        beacon(type="OTHER"),
        beacon(v=2),
        beacon(v=True),
        beacon(host="evil host;rm"),
        beacon(host=5),
        beacon(port=0),
        beacon(port=70000),
        beacon(port="50010"),
        beacon(name="<script>"),
        beacon(coord_id=["x"]),
        b"[1, 2]",
        b"\xff\xfe",
        b'{"type": "MDC_BEACON", "v": NaN}',
        beacon(padding="x" * 1100),  # oversize datagram
    ],
)
def test_bad_beacons_ignored(data: bytes) -> None:
    assert D.parse_beacon(data, FP) is None


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def test_listen_for_beacon_over_udp() -> None:
    port = _free_udp_port()

    async def send_beacons() -> None:
        await asyncio.sleep(0.2)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto(beacon(fp="sha256:" + "cd" * 32, host="10.6.6.6"), ("127.0.0.1", port))
            s.sendto(beacon(padding="x" * 1100), ("127.0.0.1", port))
            s.sendto(beacon(host="10.0.0.42", port=50123), ("127.0.0.1", port))

    sender = asyncio.create_task(send_beacons())
    found = await D.listen_for_beacon(FP, port, None, timeout_s=5)
    await sender
    assert found is not None and (found.host, found.port) == ("10.0.0.42", 50123)


async def test_listen_for_beacon_timeout() -> None:
    assert await D.listen_for_beacon(FP, _free_udp_port(), None, timeout_s=0.2) is None


async def test_beacon_rate_limit() -> None:
    loop = asyncio.get_running_loop()
    found: asyncio.Future[D.Beacon] = loop.create_future()
    proto = D._BeaconProtocol(FP, found)
    for _ in range(25):
        proto.datagram_received(beacon(fp="sha256:" + "cd" * 32), ("1.2.3.4", 1))
    proto.datagram_received(beacon(), ("1.2.3.4", 1))  # 26th in the same second: dropped
    assert not found.done()
    assert proto.ignored == 26


def test_listen_socket_with_multicast() -> None:
    sock = D.make_listen_socket(_free_udp_port(), "239.255.77.77")
    sock.close()
    sock = D.make_listen_socket(_free_udp_port(), "not-an-ip")  # failure is logged, not fatal
    sock.close()


# --------------------------------------------------------------- helpers


def test_stricter_limits() -> None:
    own = TaskLimits(blur_radius=5, blur_iterations=64, chain_len=1000, max_tile_pixels=100)
    theirs = TaskLimits(blur_radius=50, blur_iterations=8, chain_len=10**9, max_tile_pixels=50)
    assert stricter_limits(own, theirs) == TaskLimits(5, 8, 1000, 50)


def test_detect_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANDROID_ROOT", raising=False)
    monkeypatch.setattr(sys, "prefix", "/usr")
    for plat, expected in [("linux", "linux"), ("win32", "windows"), ("darwin", "darwin")]:
        monkeypatch.setattr(sys, "platform", plat)
        assert detect_platform() == expected
    monkeypatch.setenv("ANDROID_ROOT", "/system")
    assert detect_platform() == "android"


def test_sysfs_mains_online_counts_as_charging(tmp_path: Path) -> None:
    """Regression (real laptop): AC online + battery 2 % "Not charging" must not look
    like a dying battery, or the battery floor refuses all work."""
    bat = _fake_sysfs(
        tmp_path, capacity="2", status="Not charging", current_now="1000", voltage_now="11000000"
    )
    reading = B.SysfsBattery(bat, root=tmp_path).read()
    assert reading["charging"] is False  # no AC adapter present yet
    ac = tmp_path / "AC"
    ac.mkdir()
    (ac / "type").write_text("Mains\n")
    (ac / "online").write_text("1\n")
    reading = B.SysfsBattery(bat, root=tmp_path).read()
    assert reading["charging"] is True
    assert reading["current_ma"] == 1.0  # sign follows "externally powered"
    assert B.battery_allows_work(reading, 15.0)
    (ac / "online").write_text("0\n")
    assert B.SysfsBattery(bat, root=tmp_path).read()["charging"] is False
    (tmp_path / "broken").mkdir()  # supplies without type/online files are skipped
    assert B.mains_online(tmp_path) is False
    assert B.mains_online(tmp_path / "missing") is False


def test_colluding_workers_corrupt_identically() -> None:
    """``--collude-key``: same key + same input -> the same wrong result (Phase 10)."""
    tile = np.arange(64 * 64, dtype=np.uint32).reshape(64, 64).astype(np.uint8)
    a = FaultInjector(FaultSettings(malicious_prob=1.0, collude_key="gang"), seed=1)
    b = FaultInjector(FaultSettings(malicious_prob=1.0, collude_key="gang"), seed=2)
    loner = FaultInjector(FaultSettings(malicious_prob=1.0), seed=3)
    out_a = corrupt_array(tile, a.corruption_rng(b"input-1"))
    out_b = corrupt_array(tile, b.corruption_rng(b"input-1"))
    assert np.array_equal(out_a, out_b) and not np.array_equal(out_a, tile)
    other_input = corrupt_array(tile, a.corruption_rng(b"input-2"))
    assert not np.array_equal(out_a, other_input)
    assert loner.corruption_rng(b"input-1") is loner.rng  # no key: its own randomness
    assert FaultSettings(collude_key="gang").any_enabled()
