"""TESAR-Live: reliability-aware compute offloading in a Mobile Device Cloud.

Package layout:

* ``mdc.common``      — code shared by coordinator and workers (stdlib + numpy only,
                        except ``config.py`` which is coordinator-only).
* ``mdc.tasks``       — deterministic compute kernels (the only code a worker runs).
* ``mdc.worker``      — the worker process (runs on phones via Termux).
* ``mdc.coordinator`` — the coordinator (laptop): scheduling, verification, etc.
* ``mdc.dashboard``   — the live web dashboard.
"""

__version__ = "0.1.0"
