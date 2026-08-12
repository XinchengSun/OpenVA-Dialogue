#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f logs/server_realtime_6008.pid ]]; then
  ps -p "$(cat logs/server_realtime_6008.pid)" -o pid,etime,cmd || true
fi

curl -fsS "http://127.0.0.1:${PORT:-6008}/health" | python -m json.tool
