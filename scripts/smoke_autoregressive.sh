#!/usr/bin/env bash
set -euo pipefail
curl -fsS -X POST http://127.0.0.1:3600/api/workers/mock-ar/start >/dev/null
curl -fsS -H 'content-type: application/json' -d '{"model":"fast-chat","prompt":"Open Day smoke test"}' http://127.0.0.1:8600/v1/completions

