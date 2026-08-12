#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-status}"
RUNTIME_ROOT="${CUSTOM_CASCADE_RUNTIME:-${FLASHAV2AV_DATA_ROOT:-$HOME/.local/share/flashav2av}/runtime}"
ENV_FILE="${ENV_FILE:-$RUNTIME_ROOT/config/custom_cascade.env}"
STATE_DIR="${PUBLIC_TUNNEL_STATE_DIR:-$RUNTIME_ROOT/public_tunnel}"
CLOUDFLARED_BIN="${CLOUDFLARED_BIN:-$RUNTIME_ROOT/tools/cloudflared}"
PORT_VALUE="${PORT:-7860}"
ORIGIN_URL="${PUBLIC_TUNNEL_ORIGIN:-http://127.0.0.1:$PORT_VALUE}"
PID_FILE="$STATE_DIR/cloudflared.pid"
LOG_FILE="$STATE_DIR/cloudflared.log"
URL_FILE="$STATE_DIR/public_url"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

load_runtime_env() {
  [[ -f "$ENV_FILE" ]] || die "missing runtime env: $ENV_FILE"
  set +u
  set +x
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
  set +x
  set -u
}

read_pid() {
  [[ -f "$PID_FILE" ]] || return 1
  tr -d '[:space:]' < "$PID_FILE"
}

is_managed_tunnel() {
  local pid="$1"
  local cmdline=""
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
  [[ "$cmdline" == *"$CLOUDFLARED_BIN"* ]]
  [[ "$cmdline" == *"tunnel"* ]]
  [[ "$cmdline" == *"$ORIGIN_URL"* ]]
}

stop_tunnel() {
  local pid=""
  pid="$(read_pid 2>/dev/null || true)"
  if [[ -z "$pid" ]]; then
    echo "public tunnel is not running"
    return 0
  fi
  is_managed_tunnel "$pid" || die "refusing to stop unmanaged/stale PID $pid"
  kill -TERM "$pid"
  for _ in $(seq 1 40); do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f -- "$PID_FILE" "$URL_FILE"
      echo "public tunnel stopped"
      return 0
    fi
    sleep 0.25
  done
  die "cloudflared PID $pid did not stop after 10 seconds"
}

start_tunnel() {
  local pid=""
  local public_url=""

  load_runtime_env
  [[ -n "${PUBLIC_ACCESS_TOKEN:-}" ]] \
    || die "PUBLIC_ACCESS_TOKEN must be set before exposing the frontend"
  [[ -x "$CLOUDFLARED_BIN" ]] \
    || die "missing cloudflared; run scripts/install_cloudflared.sh first"
  curl --fail --silent --show-error --max-time 5 "$ORIGIN_URL/health" >/dev/null \
    || die "origin is not healthy: $ORIGIN_URL"

  mkdir -p "$STATE_DIR"
  chmod 700 "$STATE_DIR"
  pid="$(read_pid 2>/dev/null || true)"
  if [[ -n "$pid" ]] && is_managed_tunnel "$pid"; then
    echo "public tunnel already running: PID $pid"
    [[ -f "$URL_FILE" ]] && cat "$URL_FILE"
    return 0
  fi
  rm -f -- "$PID_FILE" "$URL_FILE" "$LOG_FILE"

  nohup "$CLOUDFLARED_BIN" tunnel --no-autoupdate --url "$ORIGIN_URL" \
    >"$LOG_FILE" 2>&1 &
  pid=$!
  printf '%s\n' "$pid" > "$PID_FILE"
  chmod 600 "$PID_FILE" "$LOG_FILE"

  for _ in $(seq 1 80); do
    if ! is_managed_tunnel "$pid"; then
      tail -n 30 "$LOG_FILE" >&2 || true
      rm -f -- "$PID_FILE"
      die "cloudflared exited before publishing a URL"
    fi
    public_url="$(grep -Eo 'https://[-a-z0-9]+\.trycloudflare\.com' "$LOG_FILE" | tail -n 1 || true)"
    [[ -n "$public_url" ]] && break
    sleep 0.25
  done
  [[ -n "$public_url" ]] || {
    tail -n 30 "$LOG_FILE" >&2 || true
    stop_tunnel || true
    die "timed out waiting for the public URL"
  }

  printf '%s\n' "$public_url" > "$URL_FILE"
  chmod 600 "$URL_FILE"
  echo "public tunnel ready: $public_url"
  echo "run '$0 access-url' when the authenticated link is needed"
}

show_status() {
  local pid=""
  pid="$(read_pid 2>/dev/null || true)"
  if [[ -n "$pid" ]] && is_managed_tunnel "$pid"; then
    echo "public tunnel running: PID $pid"
    [[ -f "$URL_FILE" ]] && cat "$URL_FILE"
    return 0
  fi
  echo "public tunnel is not running"
  return 1
}

show_access_url() {
  load_runtime_env
  [[ -f "$URL_FILE" ]] || die "public URL is unavailable; start the tunnel first"
  [[ -n "${PUBLIC_ACCESS_TOKEN:-}" ]] || die "PUBLIC_ACCESS_TOKEN is not configured"
  printf '%s/?access_token=%s\n' "$(cat "$URL_FILE")" "$PUBLIC_ACCESS_TOKEN"
}

case "$ACTION" in
  start) start_tunnel ;;
  stop) stop_tunnel ;;
  restart) stop_tunnel; start_tunnel ;;
  status) show_status ;;
  url) [[ -f "$URL_FILE" ]] && cat "$URL_FILE" || die "public URL is unavailable" ;;
  access-url) show_access_url ;;
  *) die "usage: $0 {start|stop|restart|status|url|access-url}" ;;
esac
