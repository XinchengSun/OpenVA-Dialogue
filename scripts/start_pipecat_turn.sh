#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"
PID_FILE="$ROOT_DIR/logs/pipecat_turn.pid"
LOG_FILE="$ROOT_DIR/logs/pipecat_turn.log"
CONFIG_FILE="$ROOT_DIR/logs/pipecat_turn.conf"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "missing env file: $ENV_FILE" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a

if [[ -z "${PIPECAT_TURN_URL:-}" ]]; then
  exit 0
fi
if ! command -v turnserver >/dev/null 2>&1; then
  echo "PIPECAT_TURN_URL is set but coturn is not installed" >&2
  exit 1
fi
if [[ -z "${PIPECAT_TURN_USERNAME:-}" || -z "${PIPECAT_TURN_CREDENTIAL:-}" ]]; then
  echo "PIPECAT_TURN_USERNAME and PIPECAT_TURN_CREDENTIAL are required" >&2
  exit 1
fi

mkdir -p "$ROOT_DIR/logs"
if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "TURN server already running pid=$old_pid"
    exit 0
  fi
fi

relay_ip="${PIPECAT_TURN_RELAY_IP:-$(hostname -I | awk '{print $1}')}"
cat >"$CONFIG_FILE" <<EOF
listening-port=3478
listening-ip=127.0.0.1
relay-ip=$relay_ip
min-port=${PIPECAT_TURN_MIN_PORT:-49160}
max-port=${PIPECAT_TURN_MAX_PORT:-49200}
fingerprint
lt-cred-mech
realm=dystream.local
user=$PIPECAT_TURN_USERNAME:$PIPECAT_TURN_CREDENTIAL
no-cli
no-tls
no-dtls
allow-loopback-peers
no-multicast-peers
EOF
chmod 600 "$CONFIG_FILE"

nohup turnserver -c "$CONFIG_FILE" \
  --pidfile "$PID_FILE" \
  --log-file stdout \
  </dev/null >"$LOG_FILE" 2>&1 &

for _ in {1..20}; do
  [[ -s "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null && break
  sleep 0.25
done
pid="$(cat "$PID_FILE" 2>/dev/null || true)"
[[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null || { tail -80 "$LOG_FILE" >&2; exit 1; }
echo "started TURN pid=$pid log=$LOG_FILE"
