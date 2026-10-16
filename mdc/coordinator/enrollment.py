"""Device enrollment registry: ``state_dir/devices.json`` (spec S.4).

Why enrollment instead of a shared password? With one shared token, anyone who
learned it could create unlimited identities (a *Sybil* attack) and no single
phone could be locked out. Here the operator enrolls each device explicitly;
each gets its own ``worker_id`` and 32-byte secret, and can be revoked or
re-keyed individually.

File format (mode 0600)::

    {"v": 1, "devices": [{"worker_id", "name", "secret_b64", "created",
                          "revoked": false, "revoked_at": null, "last_seen": null}]}

Secrets must be stored retrievably (HMAC verification needs the key itself);
their protection is the file mode — a documented residual risk (SECURITY.md).

Concurrency: the CLI and a running coordinator may touch the file at the same
time, so every read-modify-write holds an exclusive lock on
``devices.json.lock`` and writes atomically (temp file + ``os.replace``).
The coordinator re-reads the file when its mtime changes
(:meth:`DeviceRegistry.reload_if_changed`), which is how revocation takes
effect immediately.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from mdc.common.auth import SECRET_BYTES, make_secret
from mdc.common.credentials import JoinCode, encode_join_code
from mdc.common.ids import new_id
from mdc.common.safe_names import is_valid_id, validate_name
from mdc.common.statedir import (
    check_private_file,
    ensure_private_dir,
    posix_permissions_supported,
    write_private_file,
)
from mdc.coordinator.security_log import SecurityLog

REGISTRY_FILE = "devices.json"
REGISTRY_VERSION = 1
MAX_DEVICES = 256
_DEVICE_FIELDS = frozenset(
    {"worker_id", "name", "secret_b64", "created", "revoked", "revoked_at", "last_seen"}
)


class EnrollmentError(ValueError):
    """Invalid enrollment request or corrupt registry."""


class LookupStatus(Enum):
    """Result of looking up a worker id during authentication."""

    OK = "ok"
    UNKNOWN = "unknown"
    REVOKED = "revoked"


@dataclass
class Device:
    """One enrolled device. ``secret`` is excluded from ``repr``."""

    worker_id: str
    name: str
    secret: bytes
    created: str
    revoked: bool = False
    revoked_at: str | None = None
    last_seen: str | None = None

    def __repr__(self) -> str:
        return f"Device(worker_id={self.worker_id!r}, name={self.name!r}, revoked={self.revoked})"

    def public_info(self) -> dict[str, Any]:
        """Everything except the secret (safe to print or show in the dashboard)."""
        return {
            "worker_id": self.worker_id,
            "name": self.name,
            "created": self.created,
            "revoked": self.revoked,
            "revoked_at": self.revoked_at,
            "last_seen": self.last_seen,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _device_from_json(obj: object) -> Device:
    """Strictly parse one device record."""
    if not isinstance(obj, dict) or set(obj) != _DEVICE_FIELDS:
        raise EnrollmentError("registry: malformed device record")
    worker_id, name = obj["worker_id"], obj["name"]
    if not is_valid_id(worker_id):
        raise EnrollmentError("registry: invalid worker_id")
    try:
        validate_name(name)
        secret = base64.b64decode(obj["secret_b64"], validate=True)
    except (ValueError, TypeError, binascii.Error):
        raise EnrollmentError("registry: invalid name or secret") from None
    if len(secret) != SECRET_BYTES:
        raise EnrollmentError("registry: secret has the wrong length")
    if not isinstance(obj["revoked"], bool) or not isinstance(obj["created"], str):
        raise EnrollmentError("registry: invalid revoked/created field")
    for key in ("revoked_at", "last_seen"):
        if obj[key] is not None and not isinstance(obj[key], str):
            raise EnrollmentError(f"registry: invalid {key}")
    return Device(
        worker_id=str(worker_id),
        name=str(name),
        secret=secret,
        created=obj["created"],
        revoked=obj["revoked"],
        revoked_at=obj["revoked_at"],
        last_seen=obj["last_seen"],
    )


def _device_to_json(device: Device) -> dict[str, Any]:
    return {
        "worker_id": device.worker_id,
        "name": device.name,
        "secret_b64": _b64(device.secret),
        "created": device.created,
        "revoked": device.revoked,
        "revoked_at": device.revoked_at,
        "last_seen": device.last_seen,
    }


@contextmanager
def _file_lock(lock_path: Path) -> Iterator[None]:
    """Exclusive inter-process lock (``flock`` on POSIX, ``msvcrt`` on Windows).

    ``sys.platform`` checks (rather than ``os.name``) let mypy skip the branch
    for the other operating system.
    """
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if sys.platform == "win32":  # pragma: no cover - exercised on Windows only
            import msvcrt  # noqa: PLC0415 - Windows-only module

            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl  # noqa: PLC0415 - POSIX-only module

            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


class DeviceRegistry:
    """The enrolled devices of one coordinator state directory."""

    def __init__(self, state_dir: str | Path, security_log: SecurityLog | None = None) -> None:
        self.state_dir = ensure_private_dir(state_dir)
        self.path = self.state_dir / REGISTRY_FILE
        self.lock_path = self.state_dir / (REGISTRY_FILE + ".lock")
        self.security_log = security_log
        self._devices: list[Device] = []
        self._mtime_ns: int | None = None
        self.reload()

    # ------------------------------------------------------------ persistence

    def _read(self) -> list[Device]:
        if not self.path.exists():
            return []
        if posix_permissions_supported():
            try:
                check_private_file(self.path)
            except PermissionError as exc:
                raise EnrollmentError(str(exc)) from None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise EnrollmentError(f"registry {self.path} unreadable: {exc}") from None
        if not isinstance(data, dict) or data.get("v") != REGISTRY_VERSION:
            raise EnrollmentError("registry: unsupported format version")
        records = data.get("devices")
        if not isinstance(records, list) or len(records) > MAX_DEVICES:
            raise EnrollmentError("registry: 'devices' must be a list of at most 256 records")
        devices = [_device_from_json(r) for r in records]
        if len({d.worker_id for d in devices}) != len(devices):
            raise EnrollmentError("registry: duplicate worker_id")
        active_names = [d.name for d in devices if not d.revoked]
        if len(set(active_names)) != len(active_names):
            raise EnrollmentError("registry: duplicate active device name")
        return devices

    def _write(self, devices: list[Device]) -> None:
        payload = {"v": REGISTRY_VERSION, "devices": [_device_to_json(d) for d in devices]}
        write_private_file(self.path, json.dumps(payload, indent=1) + "\n")
        self._devices = devices
        self._mtime_ns = self.path.stat().st_mtime_ns

    def reload(self) -> None:
        """Re-read the registry from disk."""
        with _file_lock(self.lock_path):
            self._devices = self._read()
            self._mtime_ns = self.path.stat().st_mtime_ns if self.path.exists() else None

    def reload_if_changed(self) -> bool:
        """Re-read if the file's mtime changed (another process edited it)."""
        current = self.path.stat().st_mtime_ns if self.path.exists() else None
        if current == self._mtime_ns:
            return False
        self.reload()
        return True

    @contextmanager
    def _transaction(self) -> Iterator[list[Device]]:
        """Lock, re-read, let the caller modify the list, then write it atomically."""
        with _file_lock(self.lock_path):
            devices = self._read()
            yield devices
            self._write(devices)

    def _log(self, event: str, **fields: Any) -> None:
        if self.security_log is not None:
            self.security_log.emit(event, **fields)

    # ---------------------------------------------------------------- queries

    def devices(self) -> list[Device]:
        """All device records (including revoked ones)."""
        return list(self._devices)

    def list_devices(self) -> list[dict[str, Any]]:
        """Public info of every device — never contains secrets."""
        return [d.public_info() for d in self._devices]

    def find_active(self, name: str) -> Device | None:
        """The non-revoked device with this name, if any."""
        for device in self._devices:
            if device.name == name and not device.revoked:
                return device
        return None

    def lookup(self, worker_id: str) -> tuple[LookupStatus, bytes | None]:
        """Authentication lookup: (status, secret if OK)."""
        for device in self._devices:
            if device.worker_id == worker_id:
                if device.revoked:
                    return LookupStatus.REVOKED, None
                return LookupStatus.OK, device.secret
        return LookupStatus.UNKNOWN, None

    # ------------------------------------------------------------ mutations

    def enroll(self, name: str, host: str, port: int, fp: str) -> JoinCode:
        """Enroll a new device and return its join code."""
        validate_name(name)
        secret = make_secret()
        worker_id = new_id("w")
        code = JoinCode(host=host, port=port, worker_id=worker_id, name=name, secret=secret, fp=fp)
        encode_join_code(code)  # validates host/port/fp before anything is stored
        with self._transaction() as devices:
            if any(d.name == name and not d.revoked for d in devices):
                raise EnrollmentError(f"a device named {name!r} is already enrolled")
            if len(devices) >= MAX_DEVICES:
                raise EnrollmentError(f"at most {MAX_DEVICES} devices can be enrolled")
            devices.append(Device(worker_id, name, secret, _now()))
        self._log("device_enrolled", name=name, worker_id=worker_id)
        return code

    def revoke(self, name: str) -> Device:
        """Revoke the active device ``name``; its secret stops working at once."""
        with self._transaction() as devices:
            for device in devices:
                if device.name == name and not device.revoked:
                    device.revoked = True
                    device.revoked_at = _now()
                    break
            else:
                raise EnrollmentError(f"no active device named {name!r}")
        self._log("device_revoked", name=name, worker_id=device.worker_id)
        return device

    def rotate_secret(self, name: str, host: str, port: int, fp: str) -> JoinCode:
        """Give ``name`` a new secret (same worker_id); the old secret stops working."""
        new_secret = make_secret()
        with self._transaction() as devices:
            for device in devices:
                if device.name == name and not device.revoked:
                    device.secret = new_secret
                    break
            else:
                raise EnrollmentError(f"no active device named {name!r}")
        self._log("device_secret_rotated", name=name, worker_id=device.worker_id)
        return JoinCode(host, port, device.worker_id, name, new_secret, fp)

    def join_codes(self, host: str, port: int, fp: str) -> list[JoinCode]:
        """Join codes for every active device (used after a certificate rotation)."""
        return [
            JoinCode(host, port, d.worker_id, d.name, d.secret, fp)
            for d in self._devices
            if not d.revoked
        ]

    def mark_seen(self, worker_id: str) -> None:
        """Record when a device last authenticated successfully."""
        with self._transaction() as devices:
            for device in devices:
                if device.worker_id == worker_id:
                    device.last_seen = _now()


def enroll(
    name: str,
    state_dir: str | Path,
    host: str,
    port: int,
    fp: str,
    security_log: SecurityLog | None = None,
) -> JoinCode:
    """Convenience wrapper used by scripts (e.g. ``local_cluster.py``)."""
    return DeviceRegistry(state_dir, security_log).enroll(name, host, port, fp)
