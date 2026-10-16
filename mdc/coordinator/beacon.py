"""UDP discovery beacons and LAN address detection (spec Phase 4 item 1).

Every ``network.beacon_interval_s`` the coordinator sends a small JSON
datagram to the beacon port (default 50000)::

    {"type": "MDC_BEACON", "v": 1, "host": "192.168.0.10", "port": 50010,
     "coord_id": "...", "name": "...", "fp": "sha256:..."}

to three destinations (``network.beacon_modes``):

* ``broadcast``: ``255.255.255.255`` *and* the subnet broadcast address
  (computed assuming a /24 netmask — reading the real netmask portably needs
  platform-specific code; see DECISIONS.md);
* ``multicast``: group ``239.255.77.77`` with TTL 1 (never leaves the LAN).

The beacon carries the certificate fingerprint so workers can ignore beacons
of *other* coordinators. It is not authenticated: a forged beacon can at most
cause a failed connection attempt, because the TLS pin check authenticates the
coordinator (spec S.3.7).
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import logging
import socket
from collections.abc import Sequence

log = logging.getLogger(__name__)

FALLBACK_IP = "127.0.0.1"


def detect_lan_ip(probe_host: str = "8.8.8.8", probe_port: int = 80) -> str:
    """Best guess of this machine's primary LAN IPv4 address.

    The "UDP connect trick": ``connect()`` on a UDP socket sends nothing; it
    only asks the OS which local interface/address it *would* use to reach
    ``probe_host``. That is the address other devices on the LAN can reach us
    at. Works offline as long as a default route exists; otherwise falls back
    to ``127.0.0.1`` (then set ``network.advertise_host`` explicitly).
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((probe_host, probe_port))
        address = str(sock.getsockname()[0])
    except OSError:
        return FALLBACK_IP
    finally:
        sock.close()
    try:
        unusable = ipaddress.ip_address(address).is_unspecified
    except ValueError:
        unusable = True
    return FALLBACK_IP if unusable else address


def subnet_broadcast(ip: str) -> str | None:
    """Broadcast address of ``ip`` assuming a /24 network (``None`` for loopback/invalid)."""
    try:
        addr = ipaddress.IPv4Address(ip)
    except ValueError:
        return None
    if addr.is_loopback or addr.is_unspecified:
        return None
    return str(ipaddress.IPv4Network(f"{ip}/24", strict=False).broadcast_address)


def beacon_payload(host: str, port: int, coord_id: str, name: str, fp: str) -> bytes:
    """The beacon datagram (compact JSON, always well under 1024 bytes)."""
    return json.dumps(
        {
            "type": "MDC_BEACON",
            "v": 1,
            "host": host,
            "port": port,
            "coord_id": coord_id,
            "name": name,
            "fp": fp,
        },
        separators=(",", ":"),
    ).encode("ascii")


def beacon_destinations(
    modes: Sequence[str], advertise_ip: str, discovery_port: int, multicast_group: str
) -> list[tuple[str, int]]:
    """Where to send each beacon for the configured modes."""
    targets: list[tuple[str, int]] = []
    if "broadcast" in modes:
        targets.append(("255.255.255.255", discovery_port))
        subnet = subnet_broadcast(advertise_ip)
        if subnet is not None:
            targets.append((subnet, discovery_port))
    if "multicast" in modes:
        targets.append((multicast_group, discovery_port))
    return targets


class BeaconSender:
    """Periodically sends the beacon to every destination."""

    def __init__(
        self,
        payload: bytes,
        destinations: list[tuple[str, int]],
        interval_s: float,
    ) -> None:
        self.payload = payload
        self.destinations = destinations
        self.interval_s = interval_s
        self.sent = 0
        self._failed: set[tuple[str, int]] = set()
        self._sock: socket.socket | None = None
        self._task: asyncio.Task[None] | None = None

    def _open(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)  # stay on the LAN
        sock.setblocking(False)
        return sock

    def send_once(self) -> int:
        """Send one beacon to each destination; return how many sends succeeded."""
        if self._sock is None:
            self._sock = self._open()
        ok = 0
        for dest in self.destinations:
            try:
                self._sock.sendto(self.payload, dest)
                ok += 1
            except OSError as exc:
                if dest not in self._failed:  # log each failing destination once
                    log.warning("beacon to %s:%d failed: %s", dest[0], dest[1], exc)
                    self._failed.add(dest)
        self.sent += 1
        return ok

    async def run(self) -> None:
        """Beacon loop."""
        while True:
            self.send_once()
            await asyncio.sleep(self.interval_s)

    def start(self) -> None:
        """Start beaconing (no-op without destinations)."""
        if self._task is None and self.destinations:
            self._task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        """Stop beaconing and close the socket."""
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None
