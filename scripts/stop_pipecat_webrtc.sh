#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT_DIR/logs/pipecat_webrtc.pid"
if [[ ! -f "$PID_FILE" ]]; then
  echo "Pipecat server is not running"
  exit 0
fi
pid="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  for _ in {1..20}; do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.25
  done
fi
rm -f "$PID_FILE"

turn_pid_file="$ROOT_DIR/logs/pipecat_turn.pid"
turn_pid=""
if [[ -f "$turn_pid_file" ]]; then
  turn_pid="$(cat "$turn_pid_file" 2>/dev/null || true)"
  if [[ -n "$turn_pid" ]] && kill -0 "$turn_pid" 2>/dev/null; then
    kill "$turn_pid"
  fi
  rm -f "$turn_pid_file"
fi

echo "stopped pid=$pid"
