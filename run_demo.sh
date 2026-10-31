#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PYTHON="${PYTHON:-$HERE/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON=python3
fi
OUTPUT="${OUTPUT:-$HERE/outputs/demo_$(date +%Y%m%d_%H%M%S)}"
exec "$PYTHON" run.py demo --output "$OUTPUT" "$@"
