#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-status}"
RUNTIME_ROOT="${CUSTOM_CASCADE_RUNTIME:-${FLASHAV2AV_DATA_ROOT:-$HOME/.local/share/flashav2av}/runtime}"
ENV_FILE="${ENV_FILE:-$RUNTIME_ROOT/config/custom_cascade.env}"
STATE_DIR="${PUBLIC_TUNNEL_STATE_DIR:-$RUNTIME_ROOT/public_tunnel}"
URL_FILE="$STATE_DIR/public_url"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIGURE="$ROOT_DIR/scripts/configure_customization_access.py"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

public_origin() {
  [[ -f "$URL_FILE" ]] || die "public URL is unavailable; start the tunnel first"
  local value=""
  value="$(tr -d '[:space:]' < "$URL_FILE")"
  [[ "$value" == https://*.trycloudflare.com || "$value" == https://* ]] \
    || die "invalid HTTPS public URL in $URL_FILE"
  printf '%s\n' "$value"
}

configure() {
  [[ -f "$ENV_FILE" ]] || die "missing runtime env: $ENV_FILE"
  "$PYTHON_BIN" "$CONFIGURE" "$@" --env-file "$ENV_FILE"
}

share_url() {
  local token_value=""
  local encoded_value=""
  token_value="$(configure show-token)"
  encoded_value="$(printf '%s' "$token_value" | "$PYTHON_BIN" -c \
    'import sys, urllib.parse; sys.stdout.write(urllib.parse.quote(sys.stdin.read(), safe=""))')"
  token_value=""
  printf '%s/customize/login#token=%s\n' "$(public_origin)" "$encoded_value"
}

case "$ACTION" in
  enable)
    configure enable --public-origin "$(public_origin)"
    echo "restart the MSE service to load the new administrator access settings"
    ;;
  disable)
    configure disable
    echo "restart the MSE service to close remote customization access"
    ;;
  rotate-token)
    configure rotate-token
    echo "restart the MSE service; all existing administrator sessions will be invalid"
    ;;
  status)
    configure status
    ;;
  login-url)
    printf '%s/customize/login\n' "$(public_origin)"
    ;;
  share-url)
    # The fragment is not sent in HTTP requests; the login page removes it before POSTing.
    share_url
    ;;
  show-token)
    # Explicit credential-revealing action for manual login.
    configure show-token
    ;;
  *)
    die "usage: $0 {enable|disable|rotate-token|status|login-url|share-url|show-token}"
    ;;
esac
