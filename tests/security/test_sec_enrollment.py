"""Enrollment, join codes, certificates and the enrollment CLI (spec S.3.2, S.4)."""

from __future__ import annotations

import base64
import datetime as dt
import json
import os
import stat
import threading
from collections.abc import Callable
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from mdc import cli
from mdc.common import credentials as C
from mdc.common.auth import make_secret
from mdc.coordinator import certs
from mdc.coordinator.enrollment import (
    MAX_DEVICES,
    DeviceRegistry,
    EnrollmentError,
    LookupStatus,
    enroll,
)
from mdc.coordinator.security_log import SecurityLog, read_security_log

pytestmark = pytest.mark.security

FP = "sha256:" + "ab" * 32


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _code(**overrides: object) -> C.JoinCode:
    fields: dict[str, object] = {
        "host": "192.168.0.10",
        "port": 50010,
        "worker_id": "w_abc",
        "name": "phone-a",
        "secret": bytes(range(32)),
        "fp": FP,
    }
    fields.update(overrides)
    return C.JoinCode(**fields)  # type: ignore[arg-type]


def _reencode(payload: dict) -> str:
    body = json.dumps(payload).encode()
    return "MDC1." + base64.urlsafe_b64encode(body).decode().rstrip("=")


# ------------------------------------------------------------- join codes


def test_join_code_round_trip() -> None:
    code = _code()
    text = C.encode_join_code(code)
    assert text.startswith("MDC1.")
    assert len(text) <= C.MAX_JOIN_CODE_BYTES
    assert C.decode_join_code(text) == code
    assert C.decode_join_code(f"  {text}\n") == code


def test_join_code_repr_hides_secret() -> None:
    code = _code()
    assert code.secret.hex() not in repr(code)
    assert "secret" not in repr(code)


def test_tampered_join_code_rejected() -> None:
    text = C.encode_join_code(_code())
    body = text[5:]
    # Flip characters across the code: every variant must be rejected or decode
    # to a different (still validated) code — never crash or be silently accepted
    # as the original with altered fields that skipped validation.
    rejected = 0
    for i in range(0, len(body), 3):
        swapped = "A" if body[i] != "A" else "B"
        tampered = "MDC1." + body[:i] + swapped + body[i + 1 :]
        try:
            decoded = C.decode_join_code(tampered)
        except C.CredentialError:
            rejected += 1
        else:
            assert decoded != _code()
    assert rejected > 0


@pytest.mark.parametrize(
    "text",
    [
        "",
        "MDC2.abc",
        "MDC1.",
        "MDC1.***",
        "MDC1.a",
        "MDC1." + "A" * 2000,
        "MDC1." + base64.urlsafe_b64encode(b"[1,2]").decode(),
        "MDC1." + base64.urlsafe_b64encode(b"not json").decode(),
        "MDC1." + base64.urlsafe_b64encode(b'{"v":1,"v":1}').decode(),
        123,
        None,
    ],
)
def test_malformed_join_codes_rejected(text: object) -> None:
    with pytest.raises(C.CredentialError):
        C.decode_join_code(text)


VALID = {
    "v": 1,
    "host": "10.0.0.1",
    "port": 50010,
    "worker_id": "w_x",
    "name": "p",
    "secret": base64.urlsafe_b64encode(bytes(32)).decode().rstrip("="),
    "fp": FP,
}


@pytest.mark.parametrize(
    "change",
    [
        {"v": 2},
        {"v": True},
        {"host": "evil host"},
        {"host": ""},
        {"host": "a" * 254},
        {"port": 0},
        {"port": 70000},
        {"port": "50010"},
        {"port": True},
        {"worker_id": "bad id!"},
        {"name": "has space"},
        {"name": "x" * 33},
        {"name": ".."},
        {"secret": base64.urlsafe_b64encode(bytes(16)).decode()},
        {"secret": "!!!"},
        {"secret": 5},
        {"fp": "sha256:XYZ"},
        {"extra": 1},
    ],
)
def test_join_code_fields_validated(change: dict) -> None:
    C.decode_join_code(_reencode(VALID))  # baseline is valid
    with pytest.raises(C.CredentialError):
        C.decode_join_code(_reencode({**VALID, **change}))


def test_join_code_missing_field_rejected() -> None:
    payload = dict(VALID)
    del payload["fp"]
    with pytest.raises(C.CredentialError):
        C.decode_join_code(_reencode(payload))


def test_encode_refuses_invalid_code() -> None:
    with pytest.raises(C.CredentialError):
        C.encode_join_code(_code(secret=b"short"))
    with pytest.raises(C.CredentialError):
        C.encode_join_code(_code(host="bad host"))


def test_longest_possible_join_code_fits_limit() -> None:
    code = _code(host="a" * 253, name="n" * 32, worker_id="w" * 64)
    text = C.encode_join_code(code)
    assert len(text) <= C.MAX_JOIN_CODE_BYTES
    assert C.decode_join_code(text) == code


def test_join_file_written_0600_and_loose_file_refused(tmp_path: Path) -> None:
    path = C.write_join_file(tmp_path / "phone.join", _code())
    assert _mode(path) == 0o600
    assert C.read_join_file(path) == _code()
    os.chmod(path, 0o644)
    with pytest.raises(PermissionError, match="chmod 600"):
        C.read_join_file(path)


def test_missing_and_binary_join_files(tmp_path: Path) -> None:
    with pytest.raises(C.CredentialError, match="not found"):
        C.read_join_file(tmp_path / "none.join")
    bad = tmp_path / "bin.join"
    bad.write_bytes(b"\xff\xfe")
    os.chmod(bad, 0o600)
    with pytest.raises(C.CredentialError, match="UTF-8"):
        C.read_join_file(bad)


def test_join_code_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(C.ENV_VAR, raising=False)
    assert C.join_code_from_env() is None
    monkeypatch.setenv(C.ENV_VAR, "  ")
    assert C.join_code_from_env() is None
    monkeypatch.setenv(C.ENV_VAR, C.encode_join_code(_code()))
    assert C.join_code_from_env() == _code()
    monkeypatch.setenv(C.ENV_VAR, "garbage")
    with pytest.raises(C.CredentialError):
        C.join_code_from_env()


# --------------------------------------------------------------- registry


def test_enroll_writes_0600_registry_with_unique_ids(
    registry: DeviceRegistry, tmp_state_dir: Path
) -> None:
    a = registry.enroll("phone-a", "10.0.0.1", 50010, FP)
    b = registry.enroll("phone-b", "10.0.0.1", 50010, FP)
    assert a.worker_id != b.worker_id
    assert a.secret != b.secret
    assert len(a.secret) == 32
    path = tmp_state_dir / "devices.json"
    assert _mode(path) == 0o600
    data = json.loads(path.read_text())
    assert data["v"] == 1
    assert [d["name"] for d in data["devices"]] == ["phone-a", "phone-b"]
    assert registry.lookup(a.worker_id) == (LookupStatus.OK, a.secret)
    assert registry.lookup("w_nobody") == (LookupStatus.UNKNOWN, None)
    # No stray temp files from the atomic writes.
    assert sorted(p.name for p in tmp_state_dir.iterdir() if p.name.startswith(".")) == []


def test_registry_persists_across_instances(registry: DeviceRegistry, tmp_state_dir: Path) -> None:
    code = registry.enroll("phone-a", "10.0.0.1", 50010, FP)
    again = DeviceRegistry(tmp_state_dir)
    assert again.lookup(code.worker_id) == (LookupStatus.OK, code.secret)


@pytest.mark.parametrize("name", ["", "bad name", "x" * 33, "../etc", "é"])
def test_enroll_rejects_bad_names(registry: DeviceRegistry, name: str) -> None:
    with pytest.raises(ValueError):
        registry.enroll(name, "10.0.0.1", 50010, FP)


def test_enroll_rejects_bad_endpoint(registry: DeviceRegistry) -> None:
    with pytest.raises(ValueError):
        registry.enroll("phone-a", "10.0.0.1", 0, FP)
    with pytest.raises(ValueError):
        registry.enroll("phone-a", "10.0.0.1", 50010, "sha256:bad")
    assert registry.devices() == []  # nothing stored on failure


def test_duplicate_active_name_rejected_but_reuse_after_revoke(registry: DeviceRegistry) -> None:
    first = registry.enroll("phone-a", "10.0.0.1", 50010, FP)
    with pytest.raises(EnrollmentError, match="already enrolled"):
        registry.enroll("phone-a", "10.0.0.1", 50010, FP)
    registry.revoke("phone-a")
    second = registry.enroll("phone-a", "10.0.0.1", 50010, FP)
    assert second.worker_id != first.worker_id
    assert registry.lookup(first.worker_id)[0] is LookupStatus.REVOKED


def test_revoke(registry: DeviceRegistry, security_log: SecurityLog) -> None:
    code = registry.enroll("phone-a", "10.0.0.1", 50010, FP)
    device = registry.revoke("phone-a")
    assert device.revoked
    assert device.revoked_at is not None
    assert registry.lookup(code.worker_id) == (LookupStatus.REVOKED, None)
    assert registry.find_active("phone-a") is None
    with pytest.raises(EnrollmentError, match="no active device"):
        registry.revoke("phone-a")
    types = [r["type"] for r in security_log.records]
    assert types == ["device_enrolled", "device_revoked"]


def test_rotate_secret(registry: DeviceRegistry) -> None:
    old = registry.enroll("phone-a", "10.0.0.1", 50010, FP)
    new = registry.rotate_secret("phone-a", "10.0.0.2", 50011, FP)
    assert new.worker_id == old.worker_id
    assert new.secret != old.secret
    assert (new.host, new.port) == ("10.0.0.2", 50011)
    assert registry.lookup(old.worker_id) == (LookupStatus.OK, new.secret)
    with pytest.raises(EnrollmentError):
        registry.rotate_secret("ghost", "h", 1, FP)


def test_list_devices_hides_secrets(registry: DeviceRegistry) -> None:
    code = registry.enroll("phone-a", "10.0.0.1", 50010, FP)
    listing = registry.list_devices()
    text = json.dumps(listing)
    assert "secret" not in text
    assert base64.b64encode(code.secret).decode() not in text
    assert code.secret.hex() not in text
    assert listing[0]["name"] == "phone-a"
    assert code.secret.hex() not in repr(registry.devices())


def test_max_devices(registry: DeviceRegistry) -> None:
    for i in range(MAX_DEVICES):
        registry.enroll(f"d{i}", "10.0.0.1", 50010, FP)
    with pytest.raises(EnrollmentError, match="at most"):
        registry.enroll("one-more", "10.0.0.1", 50010, FP)


def test_concurrent_enrollments_are_not_lost(tmp_state_dir: Path) -> None:
    errors: list[Exception] = []

    def worker(i: int) -> None:
        try:
            DeviceRegistry(tmp_state_dir).enroll(f"t{i}", "10.0.0.1", 50010, FP)
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(DeviceRegistry(tmp_state_dir).devices()) == 16


def test_reload_if_changed_sees_other_process_edits(tmp_state_dir: Path) -> None:
    coordinator_view = DeviceRegistry(tmp_state_dir)
    assert coordinator_view.reload_if_changed() is False
    cli_view = DeviceRegistry(tmp_state_dir)
    code = cli_view.enroll("phone-a", "10.0.0.1", 50010, FP)
    assert coordinator_view.lookup(code.worker_id)[0] is LookupStatus.UNKNOWN
    assert coordinator_view.reload_if_changed() is True
    assert coordinator_view.lookup(code.worker_id)[0] is LookupStatus.OK
    cli_view.revoke("phone-a")
    os.utime(tmp_state_dir / "devices.json", ns=(1, 1))  # force an mtime change
    assert coordinator_view.reload_if_changed() is True
    assert coordinator_view.lookup(code.worker_id)[0] is LookupStatus.REVOKED


def test_mark_seen(registry: DeviceRegistry) -> None:
    code = registry.enroll("phone-a", "10.0.0.1", 50010, FP)
    registry.mark_seen(code.worker_id)
    assert registry.list_devices()[0]["last_seen"] is not None


def test_join_codes_for_active_devices(registry: DeviceRegistry) -> None:
    registry.enroll("a", "h", 1, FP)
    registry.enroll("b", "h", 1, FP)
    registry.revoke("a")
    new_fp = "sha256:" + "cd" * 32
    codes = registry.join_codes("10.0.0.9", 50010, new_fp)
    assert [(c.name, c.fp, c.host) for c in codes] == [("b", new_fp, "10.0.0.9")]


def test_loose_registry_permissions_refused(registry: DeviceRegistry, tmp_state_dir: Path) -> None:
    registry.enroll("phone-a", "10.0.0.1", 50010, FP)
    os.chmod(tmp_state_dir / "devices.json", 0o644)
    with pytest.raises(EnrollmentError, match="chmod 600"):
        DeviceRegistry(tmp_state_dir)


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        "[]",
        '{"v": 2, "devices": []}',
        '{"v": 1, "devices": {}}',
        '{"v": 1, "devices": [{"name": "x"}]}',
    ],
)
def test_corrupt_registry_refused(tmp_state_dir: Path, content: str) -> None:
    path = tmp_state_dir / "devices.json"
    path.write_text(content)
    os.chmod(path, 0o600)
    with pytest.raises(EnrollmentError):
        DeviceRegistry(tmp_state_dir)


def _record(**overrides: object) -> dict:
    record = {
        "worker_id": "w_1",
        "name": "a",
        "secret_b64": base64.b64encode(bytes(32)).decode(),
        "created": "2026-01-01",
        "revoked": False,
        "revoked_at": None,
        "last_seen": None,
    }
    record.update(overrides)
    return record


@pytest.mark.parametrize(
    "devices",
    [
        [_record(worker_id="bad id")],
        [_record(name="bad name")],
        [_record(secret_b64="!!!")],  # noqa: S106 - invalid value under test
        [_record(secret_b64=base64.b64encode(bytes(8)).decode())],
        [_record(revoked="no")],
        [_record(last_seen=5)],
        [_record(), _record(name="b")],  # duplicate worker_id
        [_record(), _record(worker_id="w_2")],  # duplicate active name
        [_record()] * (MAX_DEVICES + 1),
    ],
)
def test_invalid_registry_records_refused(tmp_state_dir: Path, devices: list) -> None:
    path = tmp_state_dir / "devices.json"
    path.write_text(json.dumps({"v": 1, "devices": devices}))
    os.chmod(path, 0o600)
    with pytest.raises(EnrollmentError):
        DeviceRegistry(tmp_state_dir)


def test_enroll_wrapper(tmp_state_dir: Path) -> None:
    code = enroll("phone-z", tmp_state_dir, "10.0.0.1", 50010, FP)
    assert DeviceRegistry(tmp_state_dir).lookup(code.worker_id)[0] is LookupStatus.OK


def test_secret_never_reused() -> None:
    assert len({make_secret() for _ in range(500)}) == 500


# ------------------------------------------------------------ certificates


def test_certificate_generated_with_secure_permissions(tmp_state_dir: Path) -> None:
    info = certs.ensure_certificate(tmp_state_dir, 30)
    assert _mode(info.key_path) == 0o600
    assert _mode(info.cert_path) == 0o644
    assert _mode(info.key_path.parent) == 0o700
    cert = x509.load_pem_x509_certificate(info.cert_path.read_bytes())
    assert isinstance(cert.public_key(), ec.EllipticCurvePublicKey)
    assert cert.public_key().curve.name == "secp256r1"
    assert info.fingerprint.startswith("sha256:")
    # ensure_certificate is idempotent: same cert on the second call.
    assert certs.ensure_certificate(tmp_state_dir, 30).fingerprint == info.fingerprint


def test_rotate_certificate_changes_fingerprint(tmp_state_dir: Path) -> None:
    first = certs.ensure_certificate(tmp_state_dir, 30)
    second = certs.rotate_certificate(tmp_state_dir, 30)
    assert first.fingerprint != second.fingerprint


def test_loose_key_permissions_refused(tls_cert: certs.CertInfo, tmp_state_dir: Path) -> None:
    os.chmod(tls_cert.key_path, 0o644)
    with pytest.raises(certs.CertificateError, match="chmod 600"):
        certs.load_certificate(tmp_state_dir)


def test_missing_half_refused(tls_cert: certs.CertInfo, tmp_state_dir: Path) -> None:
    tls_cert.key_path.unlink()
    with pytest.raises(certs.CertificateError, match="missing"):
        certs.ensure_certificate(tmp_state_dir, 30)


def test_garbage_certificate_refused(tls_cert: certs.CertInfo, tmp_state_dir: Path) -> None:
    tls_cert.cert_path.write_text("garbage")
    with pytest.raises(certs.CertificateError, match="parse"):
        certs.load_certificate(tmp_state_dir)


def _write_cert(
    state: Path,
    key: ec.EllipticCurvePrivateKey,
    not_after: dt.datetime,
    not_before: dt.datetime | None = None,
) -> None:
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "x")])
    now = dt.datetime.now(dt.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(not_before or now - dt.timedelta(days=10))
        .not_valid_after(not_after)
        .sign(key, hashes.SHA256())
    )
    cert_path, _ = certs.tls_paths(state)
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def test_expired_certificate_refused_not_silently_replaced(
    tls_cert: certs.CertInfo, tmp_state_dir: Path
) -> None:
    key = serialization.load_pem_private_key(tls_cert.key_path.read_bytes(), None)
    assert isinstance(key, ec.EllipticCurvePrivateKey)
    _write_cert(
        tmp_state_dir,
        key,
        dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1),
        dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=10),
    )
    with pytest.raises(certs.CertificateError, match="expired"):
        certs.ensure_certificate(tmp_state_dir, 30)


def test_certificate_key_mismatch_refused(tls_cert: certs.CertInfo, tmp_state_dir: Path) -> None:
    other_key = ec.generate_private_key(ec.SECP256R1())
    _write_cert(tmp_state_dir, other_key, dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=5))
    with pytest.raises(certs.CertificateError, match="do not match"):
        certs.load_certificate(tmp_state_dir)


@pytest.mark.parametrize("days", [0, 3651])
def test_certificate_validity_bounds(tmp_state_dir: Path, days: int) -> None:
    with pytest.raises(ValueError):
        certs.generate_certificate(tmp_state_dir, days)


# --------------------------------------------------------------------- CLI


@pytest.fixture
def cli_args(tmp_state_dir: Path) -> Callable[..., list[str]]:
    def make(*args: str) -> list[str]:
        return [*args, "--set", f"security.state_dir={tmp_state_dir}"]

    return make


def test_cli_enroll_list_revoke_flow(
    cli_args: Callable[..., list[str]],
    tmp_path: Path,
    tmp_state_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "phone-a.join"
    assert (
        cli.main(
            cli_args(
                "enroll",
                "--name",
                "phone-a",
                "--out",
                str(out),
                "--host",
                "10.1.2.3",
                "--port",
                "50010",
            )
        )
        == 0
    )
    assert _mode(out) == 0o600
    code = C.read_join_file(out)
    assert (code.host, code.port, code.name) == ("10.1.2.3", 50010, "phone-a")
    printed = capsys.readouterr().out
    assert "MDC1." not in printed  # the code itself is never echoed
    assert code.secret.hex() not in printed

    assert cli.main(cli_args("show-fingerprint")) == 0
    assert capsys.readouterr().out.strip() == code.fp

    assert cli.main(cli_args("list-devices")) == 0
    listing = capsys.readouterr().out
    assert "phone-a" in listing and "active" in listing
    assert base64.b64encode(code.secret).decode() not in listing

    assert (
        cli.main(
            cli_args("rotate-secret", "--name", "phone-a", "--out", str(tmp_path / "new.join"))
        )
        == 0
    )
    new = C.read_join_file(tmp_path / "new.join")
    assert new.secret != code.secret

    assert cli.main(cli_args("revoke", "--name", "phone-a")) == 0
    capsys.readouterr()
    assert cli.main(cli_args("list-devices")) == 0
    assert "revoked" in capsys.readouterr().out
    log = read_security_log(tmp_state_dir / "logs" / "security.jsonl")
    assert [r["type"] for r in log] == [
        "device_enrolled",
        "device_secret_rotated",
        "device_revoked",
    ]


def test_cli_enroll_default_out_and_errors(
    cli_args: Callable[..., list[str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    assert cli.main(cli_args("enroll", "--name", "lap-b", "--host", "10.0.0.5")) == 0
    assert (tmp_path / "lap-b.join").exists()
    assert cli.main(cli_args("enroll", "--name", "lap-b", "--host", "10.0.0.5")) == 1
    assert "already enrolled" in capsys.readouterr().err
    assert cli.main(cli_args("revoke", "--name", "ghost")) == 1
    assert cli.main(cli_args("enroll", "--name", "bad name")) == 1
    assert cli.main(["list-devices", "--set", "bogus.key=1"]) == 1


def test_cli_list_empty(
    cli_args: Callable[..., list[str]], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(cli_args("list-devices")) == 0
    assert "No devices" in capsys.readouterr().out


def test_cli_enroll_qr(
    cli_args: Callable[..., list[str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    assert cli.main(cli_args("enroll", "--name", "qr-phone", "--qr", "--host", "10.0.0.5")) == 0
    out = capsys.readouterr().out
    assert "Scan this QR code" in out
    assert "MDC1." not in out  # QR blocks only, not the code text
    assert not (tmp_path / "qr-phone.join").exists()


def test_cli_rotate_cert_requires_confirmation_and_reissues(
    cli_args: Callable[..., list[str]], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.main(
        cli_args("enroll", "--name", "p1", "--out", str(tmp_path / "p1.join"), "--host", "10.0.0.5")
    )
    old = C.read_join_file(tmp_path / "p1.join")
    assert cli.main(cli_args("rotate-cert")) == 2
    assert "--yes" in capsys.readouterr().out
    reissue = tmp_path / "reissued"
    assert (
        cli.main(
            cli_args("rotate-cert", "--yes", "--reissue-dir", str(reissue), "--host", "10.0.0.5")
        )
        == 0
    )
    new = C.read_join_file(reissue / "p1.join")
    assert new.fp != old.fp
    assert (new.worker_id, new.secret) == (old.worker_id, old.secret)
    assert cli.main(cli_args("rotate-cert", "--yes")) == 0
    assert "Re-issue join codes" in capsys.readouterr().out
