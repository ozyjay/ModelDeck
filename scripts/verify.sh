#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -x .venv/bin/python ]] || { echo "ERROR: run ./scripts/setup_fedora.sh first" >&2; exit 1; }
.venv/bin/python -m ruff check backend tests
.venv/bin/python -m pytest

