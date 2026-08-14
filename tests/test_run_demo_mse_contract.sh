#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT_DIR/scripts/run_demo.sh"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

bash -n "$SCRIPT"

for action in start-mse status-mse stop-mse restart-mse; do
  grep -q "${action})" "$SCRIPT" \
    || fail "missing dispatch for $action"
done

extract_function() {
  local name="$1"
  awk -v signature="${name}() {" '
    $0 == signature { inside=1 }
    inside { print }
    inside && $0 == "}" { exit }
  ' "$SCRIPT"
}

for function_name in start_mse stop_mse show_mse_status cleanup_failed_mse_start; do
  body="$(extract_function "$function_name")"
  [[ -n "$body" ]] || fail "missing function $function_name"
  if grep -Eq 'bridge_env|BRIDGE_|voxcpm2|start_bridge|stop_owned_bridge|status_bridge' \
      <<< "$body"; then
    fail "$function_name references bridge lifecycle state"
  fi
done

start_body="$(extract_function start_mse)"
grep -q 'preflight 0' <<< "$start_body" \
  || fail "start_mse must select bridge-free preflight"
grep -q 'media_smoke' <<< "$start_body" \
  || fail "start_mse must retain media smoke validation"
grep -q 'write_ready_marker' <<< "$start_body" \
  || fail "start_mse must retain the ready marker"

status_body="$(extract_function show_mse_status)"
grep -q 'health_summary' <<< "$status_body" \
  || fail "status-mse must retain full health validation"
grep -q 'ready_marker_matches' <<< "$status_body" \
  || fail "status-mse must validate the ready marker"

stop_body="$(extract_function stop_mse)"
grep -q 'pid_is_managed_mse_instance' <<< "$stop_body" \
  || fail "stop_mse must validate process identity before signaling"
grep -q 'kill -TERM' <<< "$stop_body" \
  || fail "stop_mse must use graceful termination"
if grep -q 'kill -KILL' <<< "$stop_body"; then
  fail "stop_mse must never force-kill a process"
fi

base_identity_body="$(extract_function pid_is_managed_server)"
strict_identity_body="$(extract_function pid_is_managed_mse_instance)"
for contract in process_cwd process_exe script_matches port_matches; do
  grep -q "$contract" <<< "$base_identity_body" \
    || fail "base process identity is missing $contract"
done
grep -q 'pid_is_managed_server' <<< "$strict_identity_body" \
  || fail "strict MSE identity must include the base process identity"
for contract in CUSTOMIZATION_MAIN_ENV_FILE CUSTOMIZATION_SERVER_PORT; do
  grep -q "$contract" <<< "$strict_identity_body" \
    || fail "MSE identity check is missing $contract"
done

echo "PASS: MSE-only lifecycle is syntax-valid, bridge-isolated, and retains identity/health/media checks"
