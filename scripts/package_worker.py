#!/usr/bin/env python3
"""Build ``dist/mdc-worker.zip`` for phones (spec Phase 12 item 1).

Contents: ``mdc/__init__.py``, ``mdc/common/``, ``mdc/tasks/``, ``mdc/worker/``
(Python sources only), ``requirements-worker.txt``, ``scripts/termux_setup.sh``
and a generated ``WORKER_README.txt``. Nothing from ``mdc/coordinator`` or
``mdc/dashboard`` — the worker never imports them (``check_worker_isolation``).

The zip is **deterministic** (sorted entries, fixed timestamps and modes), so
the same sources always give the same SHA-256, which is written next to it as
``mdc-worker.zip.sha256`` for ``sha256sum -c`` on the phone. Before writing,
every file is scanned for secrets (join codes, private keys); the build fails if
one is found.

Usage: ``python scripts/package_worker.py [--out-dir dist]``
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZIP_NAME = "mdc-worker.zip"
PACKAGES = ("mdc/common", "mdc/tasks", "mdc/worker")
EXTRA_FILES = ("mdc/__init__.py", "requirements-worker.txt", "scripts/termux_setup.sh")
FIXED_TIME = (2020, 1, 1, 0, 0, 0)
SECRET_PATTERNS = (
    re.compile(r"MDC1\.[A-Za-z0-9_\-]{20,}"),  # a join code
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)
README = """TESAR-Live worker (mdc-worker.zip)
==================================

This archive contains only the worker: no coordinator code and no secrets.

On the phone (Termux):
  1. sha256sum -c mdc-worker.zip.sha256      # compare with the hash shown on the laptop
  2. unzip mdc-worker.zip -d mdc-worker && cd mdc-worker
  3. sh scripts/termux_setup.sh               # installs python, numpy, termux-api
  4. get your join code by QR scan (never over plain HTTP):
       termux-clipboard-get > phone-a.join && chmod 600 phone-a.join
  5. python -m mdc.worker --join-file phone-a.join --device-class phone --battery termux
     (add --coordinator LAPTOP_IP:50010 if discovery fails)
  6. python -m mdc.worker --join-file phone-a.join --diag   # troubleshooting

Full guide: docs/DEPLOYMENT_TERMUX.md in the project repository.
"""


def collect(root: Path = ROOT) -> list[Path]:
    """Relative paths of every file that goes into the zip (sorted)."""
    files = {Path(f) for f in EXTRA_FILES}
    for package in PACKAGES:
        for path in (root / package).rglob("*.py"):
            if "__pycache__" not in path.parts:
                files.add(path.relative_to(root))
    return sorted(files)


def find_secrets(root: Path, files: list[Path]) -> list[str]:
    """``path: pattern`` for every file that looks like it contains a secret."""
    hits = []
    for rel in files:
        text = (root / rel).read_text(encoding="utf-8", errors="replace")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                hits.append(f"{rel}: {pattern.pattern}")
    return hits


def build(out_dir: Path, root: Path = ROOT) -> tuple[Path, str]:
    """Write the zip and its ``.sha256``; return (zip path, hex digest)."""
    files = collect(root)
    secrets_found = find_secrets(root, files)
    if secrets_found:
        raise ValueError("refusing to package secrets:\n  " + "\n  ".join(secrets_found))
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / ZIP_NAME
    entries = [(str(rel), (root / rel).read_bytes()) for rel in files]
    entries.append(("WORKER_README.txt", README.encode("utf-8")))
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in sorted(entries):
            info = zipfile.ZipInfo(name, FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if name.endswith(".sh") else 0o644) << 16
            zf.writestr(info, data)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (out_dir / f"{ZIP_NAME}.sha256").write_text(f"{digest}  {ZIP_NAME}\n")
    return target, digest


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Build dist/mdc-worker.zip")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "dist")
    args = parser.parse_args(argv)
    try:
        target, digest = build(args.out_dir)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {target} ({target.stat().st_size} bytes)")
    print(f"SHA-256 {digest}")
    print("Serve the zip only (it contains no secrets), then stop the server:")
    print(f"  python -m http.server 8000 --directory {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
