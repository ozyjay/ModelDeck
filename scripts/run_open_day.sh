#!/usr/bin/env bash
set -euo pipefail
export MODELDECK_OPEN_DAY=1
export MODELDECK_ALLOW_DOWNLOADS=0
exec "$(dirname "$0")/run_dev.sh"

