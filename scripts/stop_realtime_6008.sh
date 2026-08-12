#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PID_FILE="logs/server_realtime_6008.pid"
if [[ ! -f "$PID_FILE" ]]; then
  echo "no pid file: $PID_FILE"
  exit 0
fi

pid="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
  echo "not running"
  exit 0
fi

echo "stopping pid=$pid"
kill "$pid" || true
sleep 2
if kill -0 "$pid" 2>/dev/null; then
  kill -9 "$pid" || true
fi
echo "stopped"
