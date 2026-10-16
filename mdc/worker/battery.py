"""Battery providers: Termux:API, Linux sysfs and a mock (spec Phase 3 item 5, 6.3).

Every provider has ``read() -> dict`` returning the battery object of the
protocol::

    {"pct": 87.0, "charging": False, "current_ma": -412.0, "voltage_v": 3.85,
     "source": "termux" | "sysfs" | "mock", "ts_w": <worker monotonic time>}

Sign convention: ``current_ma`` is **negative while discharging**, whatever
the hardware reports (different phones and laptops disagree on the sign).
Unknown values are ``None``.

Safety (spec S.7.3): the only external command a worker ever runs is
``termux-battery-status``. Its path is resolved once with ``shutil.which``;
it runs with an argument list (no shell) and a 5-second timeout.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)

TERMUX_COMMAND = "termux-battery-status"
TERMUX_TIMEOUT_S = 5.0
SYSFS_ROOT = Path("/sys/class/power_supply")
MICRO = 1_000_000
MILLI_PER_MICRO = 1000
PERCENT = 100.0


class BatteryProvider(Protocol):
    """Anything with ``read() -> dict`` (the protocol's battery object)."""

    source: str

    def read(self) -> dict[str, Any]:
        """Return the current battery reading."""
        ...


def empty_reading(source: str) -> dict[str, Any]:
    """A reading with every value unknown."""
    return {
        "pct": None,
        "charging": None,
        "current_ma": None,
        "voltage_v": None,
        "source": source,
        "ts_w": time.monotonic(),
    }


def _finite(value: object) -> float | None:
    """``float(value)`` if it is a finite number, else ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    as_float = float(value)
    return as_float if math.isfinite(as_float) else None


def _discharge_sign(current_ma: float | None, charging: bool | None) -> float | None:
    """Normalize the sign: negative while discharging, positive while charging."""
    if current_ma is None:
        return None
    magnitude = abs(current_ma)
    if charging is True:
        return magnitude
    if charging is False:
        return -magnitude
    return current_ma


# --------------------------------------------------------------- Termux


def parse_termux(output: str) -> dict[str, Any]:
    """Parse the JSON printed by ``termux-battery-status``.

    Example output::

        {"health": "GOOD", "percentage": 87, "plugged": "UNPLUGGED",
         "status": "DISCHARGING", "temperature": 30.1, "current": -412000}

    ``current`` is in microamperes (µA) on most devices; converted to mA.
    """
    data = json.loads(output)
    if not isinstance(data, dict):
        raise ValueError("termux-battery-status did not print a JSON object")
    status = str(data.get("status", "")).upper()
    plugged = str(data.get("plugged", "")).upper()
    charging: bool | None
    if status in {"CHARGING", "FULL"} or (plugged and plugged != "UNPLUGGED"):
        charging = True
    elif status in {"DISCHARGING", "NOT_CHARGING"} or plugged == "UNPLUGGED":
        charging = False  # unplugged cannot be charging, whatever "status" says
    else:
        charging = None
    pct = _finite(data.get("percentage"))
    current_ua = _finite(data.get("current"))
    current_ma = None if current_ua is None else current_ua / MILLI_PER_MICRO
    voltage = _finite(data.get("voltage"))  # some Termux versions report mV
    reading = empty_reading("termux")
    reading.update(
        pct=None if pct is None else max(0.0, min(PERCENT, pct)),
        charging=charging,
        current_ma=_discharge_sign(current_ma, charging),
        voltage_v=None if voltage is None else (voltage / 1000 if voltage > 100 else voltage),  # noqa: PLR2004
    )
    return reading


class TermuxBattery:
    """Battery via the Termux:API ``termux-battery-status`` command."""

    source = "termux"

    def __init__(
        self,
        command_path: str | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        # Resolve the executable once; never search PATH again on each call.
        self.command_path = command_path or shutil.which(TERMUX_COMMAND)
        self._runner = runner
        self._last: dict[str, Any] | None = None
        self._warned = False

    @staticmethod
    def available() -> bool:
        """True if ``termux-battery-status`` is on PATH."""
        return shutil.which(TERMUX_COMMAND) is not None

    def read(self) -> dict[str, Any]:
        """Run the command (argument list, no shell, 5 s timeout) and parse it.

        On failure, the last good reading is returned (or an empty one) and a
        warning is logged once.
        """
        if self.command_path is None:
            return self._fallback("termux-battery-status not found")
        try:
            proc = self._runner(
                [self.command_path],
                capture_output=True,
                text=True,
                timeout=TERMUX_TIMEOUT_S,
                check=False,
            )
            if proc.returncode != 0:
                return self._fallback(f"exit status {proc.returncode}")
            reading = parse_termux(proc.stdout)
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            return self._fallback(type(exc).__name__)
        self._last = reading
        return reading

    def _fallback(self, why: str) -> dict[str, Any]:
        if not self._warned:
            log.warning("termux battery reading failed (%s); is the Termux:API app installed?", why)
            self._warned = True
        if self._last is not None:
            return dict(self._last, ts_w=time.monotonic())
        return empty_reading("termux")


# ---------------------------------------------------------------- sysfs


def find_sysfs_battery(root: Path = SYSFS_ROOT) -> Path | None:
    """The first ``BAT*`` directory under ``root`` (Linux laptops), if any."""
    if not root.is_dir():
        return None
    for path in sorted(root.glob("BAT*")):
        if (path / "capacity").exists():
            return path
    return None


def _read_number(path: Path) -> float | None:
    try:
        return _finite(float(path.read_text().strip()))
    except (OSError, ValueError):
        return None


def mains_online(root: Path = SYSFS_ROOT) -> bool:
    """True if any ``Mains`` power supply (AC adapter) reports ``online == 1``."""
    if not root.is_dir():
        return False
    for supply in root.iterdir():
        try:
            kind = (supply / "type").read_text().strip()
            online = (supply / "online").read_text().strip()
        except OSError:
            continue
        if kind == "Mains" and online == "1":
            return True
    return False


class SysfsBattery:
    """Battery via ``/sys/class/power_supply/BAT*`` (Linux laptops).

    ``charging`` means *externally powered*: a laptop on AC power counts as
    charging even when the battery reports "Not charging" (charge thresholds,
    battery-care modes). Otherwise a plugged-in laptop with a low battery would
    wrongly refuse work (found on the development laptop: AC online, battery 2 %,
    status "Not charging" — see OBSERVATIONS.md). Termux treats "plugged" the same way.
    """

    source = "sysfs"

    def __init__(self, path: Path | None = None, root: Path = SYSFS_ROOT) -> None:
        self.path = path or find_sysfs_battery(root)
        self.root = root

    def read(self) -> dict[str, Any]:
        """Read capacity, status and current (``current_now`` µA, or ``power_now`` µW / V)."""
        reading = empty_reading("sysfs")
        if self.path is None:
            return reading
        pct = _read_number(self.path / "capacity")
        try:
            status = (self.path / "status").read_text().strip().lower()
        except OSError:
            status = ""
        charging: bool | None = {
            "charging": True,
            "full": True,
            "discharging": False,
            "not charging": False,
        }.get(status)
        if mains_online(self.root):
            charging = True
        voltage_uv = _read_number(self.path / "voltage_now")
        voltage_v = None if voltage_uv is None else voltage_uv / MICRO
        current_ua = _read_number(self.path / "current_now")
        current_ma: float | None
        if current_ua is not None:
            current_ma = current_ua / MILLI_PER_MICRO
        else:
            power_uw = _read_number(self.path / "power_now")
            current_ma = (
                None
                if power_uw is None or not voltage_v
                else (power_uw / MICRO) / voltage_v * 1000.0
            )
        reading.update(
            pct=None if pct is None else max(0.0, min(PERCENT, pct)),
            charging=charging,
            current_ma=_discharge_sign(current_ma, charging),
            voltage_v=voltage_v,
        )
        return reading


# ----------------------------------------------------------------- mock


class MockBattery:
    """Simulated battery for laptop-only clusters and tests.

    Starts at ``start_pct`` and drains linearly at ``drain_pct_per_min``,
    plus extra drain while busy (``busy_extra_pct_per_min`` per busy task).
    ``current_ma`` is synthesized from the busy state.
    """

    source = "mock"

    def __init__(
        self,
        start_pct: float = 80.0,
        drain_pct_per_min: float = 0.5,
        busy_extra_pct_per_min: float = 1.0,
        charging: bool = False,
        voltage_v: float = 3.85,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0 <= start_pct <= PERCENT:
            raise ValueError("start_pct must be in 0..100")
        if drain_pct_per_min < 0 or busy_extra_pct_per_min < 0:
            raise ValueError("drain rates must be >= 0")
        self._clock = clock
        self._lock = threading.Lock()
        self._pct = float(start_pct)
        self._drain = drain_pct_per_min
        self._busy_extra = busy_extra_pct_per_min
        self.charging = charging
        self.voltage_v = voltage_v
        self._busy = 0
        self._last_t = clock()

    def set_busy(self, active_tasks: int) -> None:
        """Tell the mock how many tasks are running (affects drain and current)."""
        with self._lock:
            self._advance()
            self._busy = max(0, active_tasks)

    def _advance(self) -> None:
        now = self._clock()
        minutes = max(0.0, now - self._last_t) / 60.0
        self._last_t = now
        if not self.charging:
            rate = self._drain + self._busy_extra * self._busy
            self._pct = max(0.0, self._pct - rate * minutes)

    def read(self) -> dict[str, Any]:
        """Current simulated reading."""
        with self._lock:
            self._advance()
            idle_ma, busy_ma = 150.0, 600.0
            current = idle_ma + busy_ma * self._busy
            reading = empty_reading("mock")
            reading.update(
                pct=round(self._pct, 3),
                charging=self.charging,
                current_ma=current if self.charging else -current,
                voltage_v=self.voltage_v,
            )
            return reading


def make_battery(
    kind: str,
    mock_pct: float = 80.0,
    mock_drain: float = 0.5,
) -> TermuxBattery | SysfsBattery | MockBattery:
    """Provider for ``--battery auto|termux|sysfs|mock``.

    ``auto``: termux if ``termux-battery-status`` exists, else sysfs if a
    battery exists, else mock.
    """
    if kind == "termux":
        return TermuxBattery()
    if kind == "sysfs":
        return SysfsBattery()
    if kind == "mock":
        return MockBattery(mock_pct, mock_drain)
    if kind != "auto":
        raise ValueError(f"unknown battery provider {kind!r}")
    if TermuxBattery.available():
        return TermuxBattery()
    if find_sysfs_battery() is not None:
        return SysfsBattery()
    return MockBattery(mock_pct, mock_drain)


def battery_allows_work(reading: dict[str, Any], min_pct: float) -> bool:
    """Device-owner protection (S.7.6): work only above ``min_pct`` or while charging.

    Unknown battery level counts as allowed (desktops have no battery).
    """
    if reading.get("charging") is True:
        return True
    pct = reading.get("pct")
    return pct is None or float(pct) >= min_pct
