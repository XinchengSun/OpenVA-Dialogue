#!/usr/bin/env bash
set -euo pipefail
port="${PIPECAT_PORT:-7860}"
curl --fail --silent --show-error "http://127.0.0.1:${port}/health"
printf '\n'
