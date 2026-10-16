#!/bin/sh
# Build dist/mdc-worker.zip (+ .sha256) for phones. See scripts/package_worker.py.
set -eu
cd "$(dirname "$0")/.."
exec "${PYTHON:-python3}" scripts/package_worker.py "$@"
