"""Listen for coordinator UDP beacons (spec Phase 3 item 1, S.3.7, S.6.6).

The coordinator broadcasts (and multicasts) a small JSON datagram every second::

    {"type": "MDC_BEACON", "v": 1, "host": "192.168.0.10", "port": 50010,
     "coord_id": "...", "name": "...", "fp": "sha256:..."}

Discovery is a **convenience, not a trust decision**: a beacon only tells the
worker where to try connecting, and only if its ``fp`` equals the fingerprint
pinned in the worker's join code. Anyone can forge a beacon (the fingerprint
is public), but the TLS pin check is what authenticates the coordinator — a
forged beacon can at worst cause one failed, logged connection attempt.

Defensive limits: datagrams larger than 1024 bytes are ignored, at most 10
beacons per second are processed, and parsing is strict.

If broadcast/multicast is blocked (common on campus WiFi), nothing arrives;
the caller then prints guidance to use ``--coordinator HOST:PORT``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import socket
import struct
from dataclasses import dataclass
from typing import Any

from mdc.common.protocol import ProtocolError, strict_json_loads
from mdc.common.tls import fingerprints_equal, is_valid_fingerprint

log = logging.getLogger(__name__)

MAX_DATAGRAM = 1024
MAX_BEACONS_PER_S = 10
BEACON_TYPE = "MDC_BEACON"
MAX_PORT = 65535
_HOST_RE = re.compile(r"^[A-Za-z0-9.:-]{1,253}$")
_LABEL_RE = re.compile(r"^[A-Za-z0-9_.:-]{0,64}$")


@dataclass(frozen=True)
class Beacon:
    """A validated beacon."""

    host: str
    port: int
    coord_id: str
    name: str


def _beacon_fields_ok(obj: dict[str, Any], expected_fp: str) -> bool:
    """Type/format checks of every beacon field, and the fingerprint match."""
    host, port, fp = obj.get("host"), obj.get("port"), obj.get("fp")
    coord_id, name = obj.get("coord_id", ""), obj.get("name", "")
    return (
        obj.get("type") == BEACON_TYPE
        and type(obj.get("v")) is int
        and obj.get("v") == 1
        and isinstance(host, str)
        and _HOST_RE.fullmatch(host) is not None
        and type(port) is int
        and 1 <= port <= MAX_PORT
        and isinstance(coord_id, str)
        and _LABEL_RE.fullmatch(coord_id) is not None
        and isinstance(name, str)
        and _LABEL_RE.fullmatch(name) is not None
        and is_valid_fingerprint(fp)
        and fingerprints_equal(str(fp), expected_fp)
    )


def parse_beacon(data: bytes, expected_fp: str) -> Beacon | None:
    """Return the beacon if ``data`` is a valid beacon for our pinned coordinator.

    Returns ``None`` (never raises) for anything else: oversize, malformed,
    wrong type/version, invalid host/port, or a different fingerprint.
    """
    if len(data) > MAX_DATAGRAM:
        return None
    try:
        obj = strict_json_loads(data)
    except ProtocolError:
        return None
    if not isinstance(obj, dict) or not _beacon_fields_ok(obj, expected_fp):
        return None
    return Beacon(obj["host"], obj["port"], obj.get("coord_id", ""), obj.get("name", ""))


class _BeaconProtocol(asyncio.DatagramProtocol):
    """Collects the first valid beacon; rate-limits processing."""

    def __init__(self, expected_fp: str, found: asyncio.Future[Beacon]) -> None:
        self.expected_fp = expected_fp
        self.found = found
        self._window_start = 0.0
        self._count = 0
        self.ignored = 0

    def datagram_received(self, data: bytes, addr: tuple[str | Any, int]) -> None:
        loop_time = asyncio.get_running_loop().time()
        if loop_time - self._window_start >= 1.0:
            self._window_start, self._count = loop_time, 0
        self._count += 1
        if self._count > MAX_BEACONS_PER_S or self.found.done():
            self.ignored += 1
            return
        beacon = parse_beacon(data, self.expected_fp)
        if beacon is None:
            self.ignored += 1
            log.debug("ignored datagram from %s (not a valid beacon for us)", addr[0])
            return
        self.found.set_result(beacon)


def make_listen_socket(port: int, multicast_group: str | None) -> socket.socket:
    """UDP socket bound to ``("", port)`` with address reuse and multicast membership.

    ``SO_REUSEADDR``/``SO_REUSEPORT`` let several workers on one machine share
    the beacon port. Joining the multicast group may fail on some Android
    builds; that is logged and broadcast still works.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            with contextlib.suppress(OSError):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.bind(("", port))
        if multicast_group:
            try:
                # ip_mreq: group address + local interface (INADDR_ANY = let the OS choose).
                membership = socket.inet_aton(multicast_group) + struct.pack(
                    "!I", socket.INADDR_ANY
                )
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)
            except OSError as exc:
                log.warning(
                    "could not join multicast group %s (%s); broadcast only", multicast_group, exc
                )
        sock.setblocking(False)
    except BaseException:
        sock.close()
        raise
    return sock


async def listen_for_beacon(
    expected_fp: str,
    port: int = 50000,
    multicast_group: str | None = "239.255.77.77",
    timeout_s: float = 10.0,
) -> Beacon | None:
    """Wait up to ``timeout_s`` for the first valid beacon; ``None`` if none came."""
    loop = asyncio.get_running_loop()
    found: asyncio.Future[Beacon] = loop.create_future()
    sock = make_listen_socket(port, multicast_group)
    transport, _protocol = await loop.create_datagram_endpoint(
        lambda: _BeaconProtocol(expected_fp, found), sock=sock
    )
    try:
        return await asyncio.wait_for(found, timeout_s)
    except asyncio.TimeoutError:
        return None
    finally:
        transport.close()


NO_BEACON_HELP = (
    "No beacon heard. Broadcast may be blocked on this network. "
    "Start the worker with --coordinator HOST:PORT"
)
