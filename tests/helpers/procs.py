"""Environment for subprocesses started by tests.

Tests must not depend on ``pip install -e .`` (CI installs only the locked
dependencies), so child processes get the repository root on ``PYTHONPATH``.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def child_env(**overrides: str) -> dict[str, str]:
    """``os.environ`` + repository on PYTHONPATH + overrides (MDC_JOIN_CODE removed)."""
    env = dict(os.environ)
    env.pop("MDC_JOIN_CODE", None)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + existing if existing else "")
    env.update(overrides)
    return env
