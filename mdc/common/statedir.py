"""Secure handling of the private state directory and secret files (spec S.4, S.10).

The coordinator keeps its TLS key, the device registry (with HMAC secrets) and
the dashboard admin token in ``security.state_dir`` (default ``~/.mdc``).
Workers keep their join file somewhere similar. Rules enforced here:

* the directory is created with mode ``0700`` (only the owner may list/enter it);
  an existing directory with group/other permissions is tightened to ``0700``;
* secret files are written with mode ``0600`` and **atomically**: data goes to
  a temporary file in the same directory, is fsync'ed, then ``os.replace``'d
  over the target. A crash mid-write therefore leaves either the old or the
  new file, never a half-written registry;
* a secret file that is readable by group/others is refused
  (:func:`check_private_file`), with a message explaining how to fix it.

On platforms without POSIX permissions (Windows) the permission checks are
skipped, because ``os.chmod`` there cannot express owner-only access.
"""

from __future__ import annotations

import logging
import os
import stat
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

DIR_MODE = 0o700
FILE_MODE = 0o600


class InsecurePermissionsError(PermissionError):
    """A secret file or directory is accessible by other users."""


def posix_permissions_supported() -> bool:
    """True where file modes mean what POSIX says (Linux, Android, macOS)."""
    return os.name == "posix"


def ensure_private_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and make sure it is ``0700``.

    Returns the absolute, user-expanded path.

    Raises:
        InsecurePermissionsError: if the path exists but is not a directory,
            is a symlink, or is owned by another user.
    """
    target = Path(path).expanduser().absolute()
    if target.is_symlink():
        # A symlink could point the secrets at a directory an attacker controls.
        raise InsecurePermissionsError(f"state directory {target} must not be a symlink")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.mkdir(mode=DIR_MODE)
    if not target.is_dir():
        raise InsecurePermissionsError(f"state path {target} exists but is not a directory")
    if posix_permissions_supported():
        st = target.stat()
        if st.st_uid != os.getuid():
            raise InsecurePermissionsError(
                f"state directory {target} is owned by uid {st.st_uid}, not by you"
            )
        if stat.S_IMODE(st.st_mode) != DIR_MODE:
            log.warning(
                "tightening permissions of %s from %o to %o",
                target,
                stat.S_IMODE(st.st_mode),
                DIR_MODE,
            )
            os.chmod(target, DIR_MODE)
    return target


def ensure_private_subdir(state_dir: str | Path, name: str) -> Path:
    """Create ``state_dir/name`` with mode ``0700`` (e.g. ``tls``, ``logs``)."""
    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError(f"invalid subdirectory name {name!r}")
    ensure_private_dir(state_dir)
    return ensure_private_dir(Path(state_dir).expanduser() / name)


def write_private_file(path: str | Path, data: bytes | str, mode: int = FILE_MODE) -> Path:
    """Atomically write ``data`` to ``path`` with permissions ``mode`` (default 0600).

    ``tempfile.mkstemp`` creates the temporary file with mode 0600 from the
    start, so the secret is never readable by others, not even for a moment.
    """
    target = Path(path).expanduser().absolute()
    payload = data.encode("utf-8") if isinstance(data, str) else data
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        if posix_permissions_supported():
            os.chmod(tmp_name, mode)
        os.replace(tmp_name, target)
    except BaseException:
        # Never leave a stray temp file containing a secret behind.
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return target


def check_private_file(path: str | Path) -> None:
    """Refuse a secret file that group or others can read or write.

    Raises:
        FileNotFoundError: if it does not exist.
        InsecurePermissionsError: if its mode has any group/other bit set.
    """
    target = Path(path).expanduser()
    st = target.stat()
    if not posix_permissions_supported():
        return
    if stat.S_IMODE(st.st_mode) & 0o077:
        raise InsecurePermissionsError(
            f"{target} is accessible by other users (mode "
            f"{stat.S_IMODE(st.st_mode):o}). It contains a secret; fix it with: "
            f"chmod 600 {target}"
        )


def read_private_file(path: str | Path) -> bytes:
    """Read a secret file after checking its permissions."""
    check_private_file(path)
    return Path(path).expanduser().read_bytes()
