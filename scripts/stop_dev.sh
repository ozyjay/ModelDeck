#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for name in gateway management; do
  file="var/run/$name.pid"
  if [[ -f "$file" ]]; then
    pid="$(cat "$file")"
    kill "$pid" 2>/dev/null || true
    rm -f "$file"
  fi
done
echo "ModelDeck services stopped. Managed workers receive graceful shutdown from the management service."

