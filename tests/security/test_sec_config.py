"""Phase-0 security controls: insecure config combinations and the private
state directory (spec S.3.1, S.4.2, S.10, S.12 row 0)."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

import mdc.common.statedir as sd
from mdc.common.config import ConfigError, load_config
from mdc.common.statedir import (
    InsecurePermissionsError,
    check_private_file,
    ensure_private_dir,
    ensure_private_subdir,
    posix_permissions_supported,
    read_private_file,
    write_private_file,
)

pytestmark = pytest.mark.security

posix_only = pytest.mark.skipif(
    not posix_permissions_supported(), reason="POSIX permissions not available on this OS"
)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


# ------------------------------------------------------ insecure combinations


def test_tls_off_without_insecure_dev_refused() -> None:
    with pytest.raises(ConfigError, match="--insecure-dev"):
        load_config(overrides=["security.tls=false"])


def test_tls_off_with_insecure_dev_requires_loopback_bind() -> None:
    with pytest.raises(ConfigError, match=r"127\.0\.0\.1"):
        load_config(overrides=["security.tls=false"], insecure_dev=True)  # default bind 0.0.0.0
    with pytest.raises(ConfigError):
        load_config(
            overrides=["security.tls=false", "network.coordinator_bind=192.168.0.5"],
            insecure_dev=True,
        )


def test_tls_off_with_insecure_dev_on_loopback_allowed() -> None:
    cfg = load_config(
        overrides=["security.tls=false", "network.coordinator_bind=127.0.0.1"], insecure_dev=True
    )
    assert cfg.security.tls is False


def test_insecure_dev_flag_alone_does_not_disable_tls() -> None:
    cfg = load_config(insecure_dev=True)
    assert cfg.security.tls is True


def test_demo_controls_without_lab_mode_refused() -> None:
    with pytest.raises(ConfigError, match="lab_mode"):
        load_config(overrides=["dashboard.demo_controls=true"])


def test_demo_controls_with_lab_mode_allowed() -> None:
    cfg = load_config(overrides=["dashboard.demo_controls=true", "security.lab_mode=true"])
    assert cfg.dashboard.demo_controls is True


def test_secure_defaults() -> None:
    cfg = load_config()
    assert cfg.security.tls is True
    assert cfg.security.lab_mode is False
    assert cfg.dashboard.demo_controls is False
    assert cfg.dashboard.host == "127.0.0.1"


def test_tls_string_false_is_not_interpreted() -> None:
    # "false" as a quoted string must be rejected, not read as falsy.
    with pytest.raises(ConfigError, match="expected bool"):
        load_config(overrides=["security.tls='false'"])


# -------------------------------------------------------------- state dir


@posix_only
def test_state_dir_created_0700(tmp_path: Path) -> None:
    state = ensure_private_dir(tmp_path / "a" / "b" / ".mdc")
    assert state.is_dir()
    assert _mode(state) == 0o700


@posix_only
def test_existing_state_dir_is_tightened(tmp_path: Path) -> None:
    loose = tmp_path / "loose"
    loose.mkdir()
    os.chmod(loose, 0o750)  # noqa: S103 - deliberately loose: test tightening
    ensure_private_dir(loose)
    assert _mode(loose) == 0o700


def test_state_dir_tilde_expanded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    state = ensure_private_dir("~/.mdc")
    assert state == tmp_path / ".mdc"


def test_state_dir_symlink_refused(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(InsecurePermissionsError, match="symlink"):
        ensure_private_dir(link)


def test_state_path_that_is_a_file_refused(tmp_path: Path) -> None:
    f = tmp_path / "file"
    f.write_text("x")
    with pytest.raises(InsecurePermissionsError, match="not a directory"):
        ensure_private_dir(f)


@posix_only
def test_state_dir_owned_by_other_user_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = ensure_private_dir(tmp_path / "s")
    monkeypatch.setattr(os, "getuid", lambda: state.stat().st_uid + 1)
    with pytest.raises(InsecurePermissionsError, match="owned by"):
        ensure_private_dir(state)


@posix_only
def test_subdir_created_0700(tmp_state_dir: Path) -> None:
    tls = ensure_private_subdir(tmp_state_dir, "tls")
    assert tls == tmp_state_dir / "tls"
    assert _mode(tls) == 0o700


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "a\\b"])
def test_subdir_rejects_traversal_names(tmp_state_dir: Path, name: str) -> None:
    with pytest.raises(ValueError):
        ensure_private_subdir(tmp_state_dir, name)


# ----------------------------------------------------------- secret files


@posix_only
def test_secret_file_written_0600_atomically(tmp_state_dir: Path) -> None:
    path = write_private_file(tmp_state_dir / "devices.json", '{"v":1}')
    assert _mode(path) == 0o600
    assert path.read_text() == '{"v":1}'
    write_private_file(path, b'{"v":1,"devices":[]}')
    assert path.read_bytes() == b'{"v":1,"devices":[]}'
    assert _mode(path) == 0o600
    # No temp files are left behind.
    assert sorted(p.name for p in tmp_state_dir.iterdir()) == ["devices.json"]


def test_failed_write_leaves_old_file_and_no_temp(
    tmp_state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_private_file(tmp_state_dir / "devices.json", "old")

    def boom(src: str, dst: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        write_private_file(path, "new")
    assert path.read_text() == "old"
    assert sorted(p.name for p in tmp_state_dir.iterdir()) == ["devices.json"]


@posix_only
@pytest.mark.parametrize("mode", [0o644, 0o640, 0o604, 0o660, 0o606, 0o610])
def test_group_or_world_accessible_secret_refused(tmp_path: Path, mode: int) -> None:
    secret = tmp_path / "phone.join"
    secret.write_text("secret")
    os.chmod(secret, mode)
    with pytest.raises(InsecurePermissionsError, match="chmod 600"):
        check_private_file(secret)
    with pytest.raises(InsecurePermissionsError):
        read_private_file(secret)


@posix_only
@pytest.mark.parametrize("mode", [0o600, 0o400])
def test_owner_only_secret_accepted(tmp_path: Path, mode: int) -> None:
    secret = tmp_path / "phone.join"
    secret.write_text("secret")
    os.chmod(secret, mode)
    check_private_file(secret)
    assert read_private_file(secret) == b"secret"


def test_missing_secret_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        check_private_file(tmp_path / "missing.join")


def test_non_posix_skips_permission_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    secret = tmp_path / "x"
    secret.write_text("s")
    os.chmod(secret, 0o644)
    monkeypatch.setattr(sd, "posix_permissions_supported", lambda: False)
    check_private_file(secret)  # Windows: cannot express owner-only, so not enforced
    write_private_file(tmp_path / "y", "z")
    ensure_private_dir(tmp_path / "d")
