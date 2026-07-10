#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${MODELDECK_PYTHON:-python3.12}"
if ! "$PYTHON" --version >/dev/null 2>&1; then
  PYENV_312="$(find "$HOME/.pyenv/versions" -mindepth 3 -maxdepth 3 -type f -path '*/bin/python3.12' 2>/dev/null | sort -V | tail -1)"
  [[ -n "$PYENV_312" ]] || { echo "ERROR: Python 3.12 is required." >&2; exit 1; }
  PYTHON="$PYENV_312"
fi
if [[ ! -d .venv ]]; then "$PYTHON" -m venv .venv; fi
[[ "$(.venv/bin/python -c 'import sys; print(sys.prefix)')" == "$PWD/.venv" ]] || { echo "ERROR: install target is not the project .venv" >&2; exit 1; }
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
echo "ModelDeck environment ready at $PWD/.venv"
