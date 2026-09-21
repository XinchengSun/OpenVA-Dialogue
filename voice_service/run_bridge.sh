#!/usr/bin/env bash
# Explicit lifecycle manager for the isolated VoxCPM2 bridge.
# It is intentionally not called by the native_s2s startup path.

set -Eeuo pipefail
set +x
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
readonly REPO_ROOT
readonly LOG_DIR="${REPO_ROOT}/logs"
readonly PID_FILE="${LOG_DIR}/voxcpm2_bridge.pid"
readonly LOG_FILE="${LOG_DIR}/voxcpm2_bridge.log"
readonly DEFAULT_ENV_FILE="${SCRIPT_DIR}/.env.voxcpm2"
readonly SERVER_MODULE="voice_service.voxcpm2_server"

ACTION="${1:-status}"
ENV_FILE="${2:-${VOXCPM2_ENV_FILE:-${DEFAULT_ENV_FILE}}}"
ENV_FILE_WAS_EXPLICIT=0
if [[ $# -ge 2 || -n "${VOXCPM2_ENV_FILE:-}" ]]; then
  ENV_FILE_WAS_EXPLICIT=1
fi

info() {
  printf '[voxcpm2-bridge] %s\n' "$*"
}

fail() {
  printf '[voxcpm2-bridge] ERROR: %s\n' "$*" >&2
  exit 1
}

load_env() {
  # Route any inherited or env-file-enabled xtrace to /dev/null. Environment
  # values (including API keys) are never printed by this manager.
  exec 9>/dev/null
  BASH_XTRACEFD=9
  set +x
  if [[ -f "${ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    if ! source "${ENV_FILE}" >/dev/null 2>&1; then
      set +a
      set +x
      unset BASH_XTRACEFD
      exec 9>&-
      fail "could not load configured env file"
    fi
    set +a
  elif (( ENV_FILE_WAS_EXPLICIT )) && [[ "${ACTION}" != "stop" ]]; then
    fail "configured env file does not exist: ${ENV_FILE}"
  fi
  set +x
  unset BASH_XTRACEFD
  exec 9>&-

  : "${VOXCPM2_BRIDGE_HOST:=127.0.0.1}"
  : "${VOXCPM2_BRIDGE_PORT:=8770}"
  : "${VOXCPM2_DEVICES:=0}"
  : "${VOXCPM2_PYTHON:=${FLASHAV2AV_DATA_ROOT:-$HOME/.local/share/flashav2av}/venvs/voxcpm2-nano-2.0.3/bin/python}"
  : "${VOXCPM2_START_TIMEOUT_SEC:=300}"
  : "${VOXCPM2_STOP_TIMEOUT_SEC:=60}"
  export VOXCPM2_BRIDGE_HOST VOXCPM2_BRIDGE_PORT VOXCPM2_DEVICES
}

validate_localhost() {
  case "${VOXCPM2_BRIDGE_HOST}" in
    127.0.0.1|localhost|::1) ;;
    *) fail "VOXCPM2_BRIDGE_HOST must be localhost" ;;
  esac
  [[ "${VOXCPM2_BRIDGE_PORT}" =~ ^[0-9]+$ ]] || fail "invalid bridge port"
  (( VOXCPM2_BRIDGE_PORT >= 1 && VOXCPM2_BRIDGE_PORT <= 65535 )) || \
    fail "invalid bridge port"
}

validate_gpu_mapping() {
  [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || \
    fail "set CUDA_VISIBLE_DEVICES to the physical GPU(s) reserved for VoxCPM2"

  local -a physical_devices logical_devices
  local item
  IFS=',' read -r -a physical_devices <<<"${CUDA_VISIBLE_DEVICES}"
  IFS=',' read -r -a logical_devices <<<"${VOXCPM2_DEVICES}"
  (( ${#physical_devices[@]} > 0 )) || fail "CUDA_VISIBLE_DEVICES is empty"
  (( ${#logical_devices[@]} > 0 )) || fail "VOXCPM2_DEVICES is empty"
  for item in "${physical_devices[@]}"; do
    [[ -n "${item//[[:space:]]/}" ]] || fail "invalid CUDA_VISIBLE_DEVICES mapping"
  done
  for item in "${logical_devices[@]}"; do
    item="${item//[[:space:]]/}"
    [[ "${item}" =~ ^[0-9]+$ ]] || \
      fail "VOXCPM2_DEVICES must use logical indexes after CUDA mapping"
    (( item < ${#physical_devices[@]} )) || \
      fail "VOXCPM2_DEVICES index is outside CUDA_VISIBLE_DEVICES mapping"
  done
}

read_pid() {
  [[ -f "${PID_FILE}" ]] || return 1
  local pid
  IFS= read -r pid <"${PID_FILE}" || return 1
  [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || return 2
  printf '%s' "${pid}"
}

process_is_ours() {
  local pid="$1"
  [[ -d "/proc/${pid}" ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  [[ -r "/proc/${pid}/stat" ]] || return 2
  [[ "$(awk '{print $3}' "/proc/${pid}/stat")" != Z ]] || return 1

  local process_cwd
  process_cwd="$(readlink -f "/proc/${pid}/cwd" 2>/dev/null)" || return 2
  [[ "${process_cwd}" == "${REPO_ROOT}" ]] || return 2

  local -a argv
  mapfile -d '' -t argv <"/proc/${pid}/cmdline" || return 2
  local i
  for (( i=0; i + 1 < ${#argv[@]}; i++ )); do
    if [[ "${argv[i]}" == "-m" && "${argv[i + 1]}" == "${SERVER_MODULE}" ]]; then
      return 0
    fi
  done
  return 2
}

single_gpu_process_matches() {
  [[ -n "${DYSTREAM_SINGLE_GPU:-}" ]] || return 0
  local pid="$1" entry
  local visible="" order="" backend="" device=""
  [[ -r "/proc/${pid}/environ" ]] || return 1
  # Check the live process, not just the edited env file. Never print its
  # environment: unrelated entries may contain credentials.
  while IFS= read -r -d '' entry; do
    case "$entry" in
      CUDA_VISIBLE_DEVICES=*) visible="${entry#*=}" ;;
      CUDA_DEVICE_ORDER=*) order="${entry#*=}" ;;
      VOXCPM2_BACKEND=*) backend="${entry#*=}" ;;
      VOXCPM2_OFFICIAL_DEVICE=*) device="${entry#*=}" ;;
    esac
  done < "/proc/${pid}/environ" || return 1
  [[ "$visible" == "$DYSTREAM_SINGLE_GPU" \
     && "$order" == "PCI_BUS_ID" \
     && "$backend" == "official_prompt_cache" \
     && "$device" == "cuda:0" ]]
}

health_check() {
  "${VOXCPM2_PYTHON}" - "${VOXCPM2_BRIDGE_HOST}" "${VOXCPM2_BRIDGE_PORT}" <<'PY' \
    >/dev/null 2>&1
import asyncio
import json
import sys

from websockets.asyncio.client import connect


async def check() -> None:
    host, port = sys.argv[1], int(sys.argv[2])
    uri_host = f"[{host}]" if ":" in host else host
    async with asyncio.timeout(2.0):
        async with connect(f"ws://{uri_host}:{port}", open_timeout=2.0) as websocket:
            await websocket.send(json.dumps({"type": "health"}))
            event = json.loads(await websocket.recv())
            if (
                event.get("type") != "health"
                or event.get("status") != "ok"
                or int(event.get("sample_rate", 0)) <= 0
            ):
                raise RuntimeError("bridge health response is invalid")


asyncio.run(check())
PY
}

port_is_open() {
  "${VOXCPM2_PYTHON}" - "${VOXCPM2_BRIDGE_HOST}" "${VOXCPM2_BRIDGE_PORT}" <<'PY' \
    >/dev/null 2>&1
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
with socket.create_connection((host, port), timeout=1.0):
    pass
PY
}

wait_for_exit() {
  local pid="$1" timeout="$2" elapsed=0
  while kill -0 "${pid}" 2>/dev/null; do
    [[ -r "/proc/${pid}/stat" ]] && \
      [[ "$(awk '{print $3}' "/proc/${pid}/stat")" == Z ]] && return 0
    (( elapsed >= timeout )) && return 1
    sleep 1
    (( elapsed += 1 ))
  done
}

stop_owned_pid() {
  local pid="$1"
  local identity_status
  if process_is_ours "${pid}"; then
    identity_status=0
  else
    identity_status=$?
  fi
  if (( identity_status == 1 )); then
    rm -f -- "${PID_FILE}"
    info "bridge already stopped (removed stale PID file)"
    return 0
  fi
  if (( identity_status != 0 )); then
    fail "PID ${pid} does not match this repo/module; refusing to signal it"
  fi

  kill -TERM "${pid}"
  if ! wait_for_exit "${pid}" "${VOXCPM2_STOP_TIMEOUT_SEC}"; then
    fail "graceful shutdown timed out; process was not force-killed (PID ${pid})"
  fi
  rm -f -- "${PID_FILE}"
  info "bridge stopped cleanly"
}

start_bridge() {
  validate_localhost
  validate_gpu_mapping
  [[ -x "${VOXCPM2_PYTHON}" ]] || fail "VoxCPM2 Python is not executable"
  command -v setsid >/dev/null 2>&1 || fail "setsid is required"
  [[ -n "${VOXCPM2_MODEL_PATH:-}" ]] || fail "VOXCPM2_MODEL_PATH is required"
  [[ -n "${VOXCPM2_PROMPT_WAV:-}" ]] || fail "VOXCPM2_PROMPT_WAV is required"
  [[ -d "${VOXCPM2_MODEL_PATH}" ]] || fail "configured model directory is missing"
  [[ -f "${VOXCPM2_PROMPT_WAV}" ]] || fail "configured prompt wav is missing"
  [[ "${VOXCPM2_START_TIMEOUT_SEC}" =~ ^[1-9][0-9]*$ ]] || \
    fail "VOXCPM2_START_TIMEOUT_SEC must be a positive integer"

  mkdir -p -- "${LOG_DIR}"
  local pid identity_status
  if pid="$(read_pid)"; then
    if process_is_ours "${pid}"; then
      identity_status=0
    else
      identity_status=$?
    fi
    if (( identity_status == 0 )); then
      single_gpu_process_matches "${pid}" \
        || fail "single-GPU live bridge configuration cannot be verified or does not match; use restart to replace this repo's bridge"
      if health_check; then
        info "bridge is already ready (PID ${pid})"
        return 0
      fi
      fail "bridge PID ${pid} is running but unhealthy; refusing duplicate start"
    elif (( identity_status == 1 )); then
      rm -f -- "${PID_FILE}"
    else
      fail "PID ${pid} does not match this repo/module; refusing to overwrite it"
    fi
  elif [[ -e "${PID_FILE}" ]]; then
    fail "PID file is malformed; refusing to overwrite it"
  fi

  if port_is_open; then
    fail "bridge port is already occupied by an unmanaged process"
  fi

  cd -- "${REPO_ROOT}"
  nohup setsid "${VOXCPM2_PYTHON}" -m "${SERVER_MODULE}" \
    >>"${LOG_FILE}" 2>&1 </dev/null &
  pid=$!
  printf '%s\n' "${pid}" >"${PID_FILE}.tmp"
  mv -f -- "${PID_FILE}.tmp" "${PID_FILE}"

  local elapsed=0
  while (( elapsed < VOXCPM2_START_TIMEOUT_SEC )); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      rm -f -- "${PID_FILE}"
      fail "bridge exited during startup; inspect ${LOG_FILE}"
    fi
    if health_check; then
      if [[ -n "${DYSTREAM_SINGLE_GPU:-}" ]]; then
        process_is_ours "${pid}" \
          || fail "single-GPU new bridge identity cannot be verified; inspect the managed PID before restart"
        if ! single_gpu_process_matches "${pid}"; then
          stop_owned_pid "${pid}"
          fail "single-GPU new bridge configuration does not match; stopped the new process; correct the runtime and use restart"
        fi
      fi
      info "bridge ready (PID ${pid}; health=ok)"
      return 0
    fi
    sleep 1
    (( elapsed += 1 ))
  done

  if process_is_ours "${pid}"; then
    kill -TERM "${pid}"
    wait_for_exit "${pid}" "${VOXCPM2_STOP_TIMEOUT_SEC}" || true
  fi
  rm -f -- "${PID_FILE}"
  fail "bridge did not become healthy before timeout; inspect ${LOG_FILE}"
}

stop_bridge() {
  validate_localhost
  [[ "${VOXCPM2_STOP_TIMEOUT_SEC}" =~ ^[1-9][0-9]*$ ]] || \
    fail "VOXCPM2_STOP_TIMEOUT_SEC must be a positive integer"
  local pid
  if ! pid="$(read_pid)"; then
    if [[ -e "${PID_FILE}" ]]; then
      fail "PID file is malformed; refusing to signal any process"
    fi
    info "bridge is already stopped"
    return 0
  fi
  stop_owned_pid "${pid}"
}

status_bridge() {
  validate_localhost
  local pid identity_status
  if ! pid="$(read_pid)"; then
    if [[ -e "${PID_FILE}" ]]; then
      fail "PID file is malformed"
    fi
    info "status=stopped"
    return 3
  fi
  if process_is_ours "${pid}"; then
    identity_status=0
  else
    identity_status=$?
  fi
  if (( identity_status == 1 )); then
    info "status=stopped (stale PID file; PID ${pid})"
    return 3
  fi
  if (( identity_status != 0 )); then
    fail "status=foreign-pid (PID ${pid}); no signal was sent"
  fi
  single_gpu_process_matches "${pid}" \
    || fail "single-GPU live bridge configuration cannot be verified or does not match; use restart to replace this repo's bridge"
  if health_check; then
    info "status=ready health=ok PID=${pid}"
    return 0
  fi
  info "status=running health=failed PID=${pid}"
  return 4
}

load_env
case "${ACTION}" in
  start) start_bridge ;;
  stop) stop_bridge ;;
  restart)
    stop_bridge
    start_bridge
    ;;
  status) status_bridge ;;
  *) fail "usage: $0 {start|stop|restart|status} [env-file]" ;;
esac
