"""``python -m mdc.coordinator --diag``: network self-check of the laptop (spec Phase 12).

Prints detected interfaces and the address workers should use, checks that the
TCP, dashboard and discovery ports are free, sends one test beacon, and shows
the certificate fingerprint and enrolled device count. It never prints a secret
(no join codes, device secrets or the dashboard token).
"""

from __future__ import annotations

import platform
import socket
import sys

import mdc
from mdc.common.config import Config
from mdc.coordinator.beacon import (
    BeaconSender,
    beacon_destinations,
    beacon_payload,
    detect_lan_ip,
)
from mdc.coordinator.certs import ensure_certificate
from mdc.coordinator.enrollment import DeviceRegistry

FIREWALL_HELP = (
    "allow inbound TCP {port} (Linux: sudo ufw allow {port}/tcp; Windows: allow Python on "
    "Private networks)"
)


def local_addresses() -> list[str]:
    """IPv4 addresses of this host (best effort, stdlib only)."""
    found = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.add(str(info[4][0]))
    except OSError:
        pass
    found.add(detect_lan_ip())
    return sorted(found)


def port_free(host: str, port: int, kind: int = socket.SOCK_STREAM) -> str:
    """``free``, ``in use`` or the error from trying to bind it."""
    if port == 0:
        return "free (ephemeral)"
    with socket.socket(socket.AF_INET, kind) as sock:
        try:
            sock.bind((host, port))
        except OSError as exc:
            return "in use" if exc.errno in {98, 48, 10048} else f"cannot bind: {exc.strerror}"
    return "free"


def run_diag(cfg: Config) -> int:
    """Print the diagnostics; return 0."""
    net = cfg.network
    print("TESAR-Live coordinator diagnostics")
    print(
        f"  version:    mdc {mdc.__version__}, Python {platform.python_version()}, {sys.platform}"
    )
    print(f"  hostname:   {socket.gethostname()}")
    lan = net.advertise_host or detect_lan_ip()
    print(f"  addresses:  {', '.join(local_addresses())}")
    print(f"  advertise:  {lan}  (workers: --coordinator {lan}:{net.tcp_port})")
    if lan.startswith("127."):
        print("              no LAN address found: connect to WiFi or set network.advertise_host")
    tcp = port_free(net.coordinator_bind, net.tcp_port)
    print(f"  tcp {net.tcp_port}:   {tcp} on {net.coordinator_bind}")
    if tcp == "free":
        print(f"              {FIREWALL_HELP.format(port=net.tcp_port)}")
    dash = port_free(cfg.dashboard.host, cfg.dashboard.port)
    print(f"  dashboard:  {cfg.dashboard.host}:{cfg.dashboard.port} {dash}")
    print(f"  discovery:  UDP {net.discovery_port}, modes {net.beacon_modes or ['none']}")
    cert = ensure_certificate(cfg.security.state_dir, cfg.security.cert_days)
    print(f"  cert fp:    {cert.fingerprint}")
    devices = DeviceRegistry(cfg.security.state_dir).list_devices()
    active = sum(1 for d in devices if not d.get("revoked"))
    print(f"  devices:    {len(devices)} enrolled ({active} active)")
    destinations = beacon_destinations(
        net.beacon_modes, lan, net.discovery_port, net.multicast_group
    )
    if destinations:
        payload = beacon_payload(lan, net.tcp_port, "diag", "diag", cert.fingerprint)
        sender = BeaconSender(payload, destinations, 1.0)
        ok = sender.send_once()
        print(f"  beacon:     test beacon sent to {ok}/{len(destinations)} destination(s)")
        if ok < len(destinations):
            print("              some destinations failed: client isolation or no route;")
            print("              workers can still use --coordinator HOST:PORT")
    else:
        print("  beacon:     disabled (network.beacon_modes is empty)")
    print("  tip:        from the phone, test reachability with: curl http://LAPTOP_IP:8000")
    print("              (while python -m http.server 8000 runs here); if it fails, the WiFi")
    print("              isolates clients: use a personal router or a phone hotspot")
    return 0
