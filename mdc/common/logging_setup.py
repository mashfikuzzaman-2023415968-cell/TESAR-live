"""Console + rotating-file logging setup.

Format (spec Phase 0): ``%(asctime)s %(levelname)s [%(name)s] %(message)s``.
Log files are created with mode 0600 (:class:`PrivateRotatingFileHandler`).

The log file rotates at 5 MB keeping 3 backups (spec S.7.4: a worker must not
fill a phone's storage). Every handler gets a :class:`RedactingFilter`, so a
join code or private key can never reach a log line, even by accident.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path

from mdc.common.redact import (
    RedactingFilter,
    install_record_redaction,
    uninstall_record_redaction,
)

LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_MARKER = "_mdc_handler"
_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class PrivateRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """``RotatingFileHandler`` whose files are created owner-only (0600).

    The standard handler creates files according to the process umask
    (often 0644/0664). Logs can contain peer addresses and device names, so
    every file — including the new one opened after each rotation — is
    created with mode 0600 from the start.
    """

    def _open(self):  # type: ignore[no-untyped-def]  # signature of the stdlib private hook
        fd = os.open(self.baseFilename, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        if os.name == "posix":
            os.chmod(self.baseFilename, 0o600)  # also tighten a pre-existing file
        return open(fd, self.mode, encoding=self.encoding, errors=self.errors)


def parse_level(level: str | int) -> int:
    """Turn ``"INFO"``/``20`` into a logging level number, rejecting junk."""
    if isinstance(level, bool):
        raise ValueError(f"invalid log level {level!r}")
    if isinstance(level, int):
        return level
    name = level.strip().upper()
    if name not in _LEVELS:
        raise ValueError(f"invalid log level {level!r}; use one of {sorted(_LEVELS)}")
    return int(getattr(logging, name))


def setup_logging(
    level: str | int = "INFO",
    logfile: str | Path | None = None,
    max_bytes: int = 5 * 1024 * 1024,
    backups: int = 3,
) -> logging.Logger:
    """Configure the root logger with a console handler and optional file.

    Calling it again replaces the handlers it installed earlier (handlers that
    other code added, such as pytest's capture handler, are left alone).
    """
    root = logging.getLogger()
    root.setLevel(parse_level(level))
    install_record_redaction()
    for handler in list(root.handlers):
        if getattr(handler, _MARKER, False):
            root.removeHandler(handler)
            handler.close()
    formatter = logging.Formatter(LOG_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.addFilter(RedactingFilter())
    setattr(console, _MARKER, True)
    root.addHandler(console)

    if logfile is not None:
        path = Path(logfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = PrivateRotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(RedactingFilter())
        setattr(file_handler, _MARKER, True)
        root.addHandler(file_handler)
    return root


def teardown_logging() -> None:
    """Remove and close the handlers installed by :func:`setup_logging`."""
    uninstall_record_redaction()
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _MARKER, False):
            root.removeHandler(handler)
            handler.close()
