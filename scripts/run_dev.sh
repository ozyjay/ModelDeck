#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -x .venv/bin/modeldeck ]] || { echo "ERROR: run ./scripts/setup_fedora.sh first" >&2; exit 1; }
./scripts/check_ports.sh
mkdir -p var/log var/run
nohup .venv/bin/modeldeck >var/log/management.log 2>&1 & echo $! >var/run/management.pid
nohup .venv/bin/modeldeck-gateway >var/log/gateway.log 2>&1 & echo $! >var/run/gateway.pid
echo "Management: http://127.0.0.1:3600"
echo "Gateway:    http://127.0.0.1:8600/v1/health"
echo "Workers:    http://127.0.0.1:8610 and http://127.0.0.1:8611 (stopped until requested)"

