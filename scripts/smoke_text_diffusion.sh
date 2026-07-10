#!/usr/bin/env bash
set -euo pipefail
curl -fsS -X POST http://127.0.0.1:3600/api/workers/mock-diffusion/start >/dev/null
curl -fsS -H 'content-type: application/json' -d '{"model":"text-diffusion","prompt":"A robot arrives at orientation.","seed":11}' http://127.0.0.1:8600/v1/refine

