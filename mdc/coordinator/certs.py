"""Coordinator TLS certificate: generation, loading, fingerprint, rotation (spec S.3.2).

* ECDSA P-256 key, self-signed certificate valid for ``security.cert_days``.
* Stored in ``state_dir/tls/``: ``cert.pem`` (0644 — public), ``key.pem`` (0600).
* Fingerprint = ``sha256:`` + hex SHA-256 of the DER certificate; it goes into
  every join code and workers pin it.

Rotating the certificate changes the fingerprint, so **every join code must
be re-issued** afterwards (``mdc.cli rotate-cert`` warns and can do it).

An expired certificate is not silently replaced (that would silently break
every worker's pin): loading it fails with instructions to rotate.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from mdc.common.statedir import check_private_file, ensure_private_subdir, write_private_file
from mdc.common.tls import fingerprint_file

log = logging.getLogger(__name__)

CERT_NAME = "cert.pem"
KEY_NAME = "key.pem"
TLS_SUBDIR = "tls"
CLOCK_SKEW = dt.timedelta(minutes=5)  # tolerate devices whose clocks are slightly behind


class CertificateError(RuntimeError):
    """The certificate/key is missing, unreadable, insecure or expired."""


@dataclass(frozen=True)
class CertInfo:
    """Paths and identity of the coordinator certificate."""

    cert_path: Path
    key_path: Path
    fingerprint: str
    not_after: dt.datetime


def tls_paths(state_dir: str | Path) -> tuple[Path, Path]:
    """``(cert_path, key_path)`` inside ``state_dir/tls``."""
    tls_dir = Path(state_dir).expanduser() / TLS_SUBDIR
    return tls_dir / CERT_NAME, tls_dir / KEY_NAME


def generate_certificate(
    state_dir: str | Path, days: int, common_name: str = "mdc-coordinator"
) -> CertInfo:
    """Create a new key + self-signed certificate (overwriting atomically)."""
    if not 1 <= days <= 3650:  # noqa: PLR2004 - same bound as the config check
        raise ValueError("certificate validity must be 1..3650 days")
    tls_dir = ensure_private_subdir(state_dir, TLS_SUBDIR)
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    now = dt.datetime.now(dt.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - CLOCK_SKEW)
        .not_valid_after(now + dt.timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    cert_path, key_path = tls_dir / CERT_NAME, tls_dir / KEY_NAME
    # Key first: a crash between the two writes leaves a cert without its key
    # (detected on load) rather than a fresh key paired with a stale cert.
    write_private_file(key_path, key_pem)
    write_private_file(cert_path, cert.public_bytes(serialization.Encoding.PEM), mode=0o644)
    info = load_certificate(state_dir)
    log.info("generated coordinator certificate %s", info.fingerprint)
    return info


def load_certificate(state_dir: str | Path) -> CertInfo:
    """Load and check an existing certificate; raise :class:`CertificateError` if unusable."""
    cert_path, key_path = tls_paths(state_dir)
    if not cert_path.exists() or not key_path.exists():
        raise CertificateError(f"certificate or key missing in {cert_path.parent}")
    try:
        check_private_file(key_path)
    except PermissionError as exc:
        raise CertificateError(str(exc)) from None
    try:
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
    except ValueError as exc:
        raise CertificateError(f"cannot parse certificate/key: {exc}") from None
    if cert.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    ) != key.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    ):
        raise CertificateError("certificate and private key do not match")
    not_after = cert.not_valid_after_utc
    if not_after <= dt.datetime.now(dt.timezone.utc):
        raise CertificateError(
            f"certificate expired on {not_after.isoformat()}; run "
            "'python -m mdc.cli rotate-cert' and re-issue all join codes"
        )
    return CertInfo(cert_path, key_path, fingerprint_file(cert_path), not_after)


def ensure_certificate(state_dir: str | Path, days: int) -> CertInfo:
    """Load the certificate, generating one on first start."""
    cert_path, key_path = tls_paths(state_dir)
    if not cert_path.exists() and not key_path.exists():
        return generate_certificate(state_dir, days)
    return load_certificate(state_dir)


def rotate_certificate(state_dir: str | Path, days: int) -> CertInfo:
    """Replace the certificate and key. All existing join codes stop working."""
    return generate_certificate(state_dir, days)
