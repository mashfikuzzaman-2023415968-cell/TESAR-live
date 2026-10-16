"""Errors raised by compute kernels.

Kept in their own module so that the kernels and the registry
(``base.py``) can both import them without an import cycle.
"""

from __future__ import annotations

import time


class TaskError(Exception):
    """A task could not be executed. ``str(exc)`` is sent as the result's ``error``."""


class TaskTimeLimitError(TaskError):
    """The kernel ran past its deadline (spec S.6.5: ``error: "time_limit"``)."""

    def __init__(self) -> None:
        super().__init__("time_limit")


def check_deadline(deadline: float | None) -> None:
    """Raise :class:`TaskTimeLimitError` if ``time.monotonic()`` is past ``deadline``.

    Kernels call this between iterations. Python threads cannot be killed, so
    a cooperative check is the only way to stop a runaway computation and
    keep a phone's CPU (and battery) from being burned indefinitely.
    """
    if deadline is not None and time.monotonic() > deadline:
        raise TaskTimeLimitError
