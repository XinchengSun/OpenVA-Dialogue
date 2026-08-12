#!/usr/bin/env bash
# Lifecycle manager for the isolated Fish/SGLang-compatible candidate bridge.

set -Eeuo pipefail
set +x
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
readonly REPO_ROOT
INSTANCE="${OPENAI_SPEECH_INSTANCE:-default}"
[[ "${INSTANCE}" =~ ^[A-Za-z0-9_-]+$ ]] || {
  printf '[openai-speech-bridge] ERROR: invalid OPENAI_SPEECH_INSTANCE\n' >&2
  exit 1
}
readonly INSTANCE
if [[ "${INSTANCE}" == "default" ]]; then
  INSTANCE_SUFFIX=""
else
  INSTANCE_SUFFIX=".${INSTANCE}"
fi
readonly INSTANCE_SUFFIX
readonly LOG_DIR="${REPO_ROOT}/logs"
readonly PID_FILE="${LOG_DIR}/openai_speech_bridge${INSTANCE_SUFFIX}.pid"
readonly LOG_FILE="${LOG_DIR}/openai_speech_bridge${INSTANCE_SUFFIX}.log"
readonly LOCK_FILE="${LOG_DIR}/openai_speech_bridge${INSTANCE_SUFFIX}.lock"
readonly SERVER_MODULE="voice_service.openai_speech_server"

ACTION="${1:-status}"
ENV_FILE="${2:-${OPENAI_SPEECH_ENV_FILE:-${SCRIPT_DIR}/.env.openai_speech}}"

info() { printf '[openai-speech-bridge:%s] %s\n' "${INSTANCE}" "$*"; }
fail() { printf '[openai-speech-bridge:%s] ERROR: %s\n' "${INSTANCE}" "$*" >&2; exit 1; }

load_env() {
  exec 9>/dev/null
  BASH_XTRACEFD=9
  set +x
  if [[ -f "${ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}" >/dev/null 2>&1 || fail "could not load env file"
    set +a
  elif [[ "${ACTION}" != "stop" && "${ACTION}" != "status" ]]; then
    fail "env file does not exist: ${ENV_FILE}"
  fi
  unset BASH_XTRACEFD
  exec 9>&-

  : "${OPENAI_SPEECH_BRIDGE_HOST:=127.0.0.1}"
  : "${OPENAI_SPEECH_BRIDGE_PORT:=8771}"
  : "${OPENAI_SPEECH_PYTHON:=${PIPECAT_PYTHON:-python3}}"
  : "${OPENAI_SPEECH_START_TIMEOUT_SEC:=300}"
  : "${OPENAI_SPEECH_STOP_TIMEOUT_SEC:=30}"
  export OPENAI_SPEECH_BRIDGE_HOST OPENAI_SPEECH_BRIDGE_PORT
}

validate_config() {
  case "${OPENAI_SPEECH_BRIDGE_HOST}" in
    127.0.0.1|localhost|::1) ;;
    *) fail "bridge host must be localhost" ;;
  esac
  [[ "${OPENAI_SPEECH_BRIDGE_PORT}" =~ ^[0-9]+$ ]] || fail "invalid port"
  (( OPENAI_SPEECH_BRIDGE_PORT >= 1 && OPENAI_SPEECH_BRIDGE_PORT <= 65535 )) \
    || fail "invalid port"
  [[ -x "$(command -v "${OPENAI_SPEECH_PYTHON}" 2>/dev/null || true)" \
     || -x "${OPENAI_SPEECH_PYTHON}" ]] || fail "Python is not executable"
}

acquire_lock() {
  command -v flock >/dev/null 2>&1 || fail "flock is required"
  mkdir -p -- "${LOG_DIR}"
  exec 8>"${LOCK_FILE}"
  flock -n 8 || fail "another bridge lifecycle action is already running"
}

read_pid() {
  [[ -f "${PID_FILE}" ]] || return 1
  local pid
  IFS= read -r pid <"${PID_FILE}" || return 1
  [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || return 2
  printf '%s' "${pid}"
}

process_is_ours() {
  local pid="$1" process_cwd
  [[ -d "/proc/${pid}" ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  [[ "$(awk '{print $3}' "/proc/${pid}/stat" 2>/dev/null)" != Z ]] || return 1
  process_cwd="$(readlink -f "/proc/${pid}/cwd" 2>/dev/null)" || return 2
  [[ "${process_cwd}" == "${REPO_ROOT}" ]] || return 2
  local -a argv
  local index
  mapfile -d '' -t argv <"/proc/${pid}/cmdline" || return 2
  for (( index=0; index + 1 < ${#argv[@]}; index++ )); do
    [[ "${argv[index]}" == "-m" && "${argv[index + 1]}" == "${SERVER_MODULE}" ]] \
      && return 0
  done
  return 2
}

health_check() {
  "${OPENAI_SPEECH_PYTHON}" - "${OPENAI_SPEECH_BRIDGE_HOST}" \
    "${OPENAI_SPEECH_BRIDGE_PORT}" <<'PY' >/dev/null 2>&1
import asyncio
import json
import sys
from websockets.asyncio.client import connect

async def check():
    host, port = sys.argv[1], int(sys.argv[2])
    uri_host = f"[{host}]" if ":" in host else host
    async with asyncio.timeout(2.0):
        async with connect(f"ws://{uri_host}:{port}", open_timeout=2.0) as ws:
            await ws.send(json.dumps({"type": "health"}))
            event = json.loads(await ws.recv())
            if event.get("status") != "ok" or int(event.get("sample_rate", 0)) <= 0:
                raise RuntimeError("invalid health")

asyncio.run(check())
PY
}

port_is_open() {
  "${OPENAI_SPEECH_PYTHON}" - "${OPENAI_SPEECH_BRIDGE_HOST}" \
    "${OPENAI_SPEECH_BRIDGE_PORT}" <<'PY' >/dev/null 2>&1
import socket
import sys
with socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=1.0):
    pass
PY
}

wait_for_exit() {
  local pid="$1" timeout="$2" elapsed=0
  while kill -0 "${pid}" 2>/dev/null; do
    [[ "$(awk '{print $3}' "/proc/${pid}/stat" 2>/dev/null)" == Z ]] && return 0
    (( elapsed >= timeout )) && return 1
    sleep 1
    (( elapsed += 1 ))
  done
}

start_bridge() {
  validate_config
  [[ -n "${OPENAI_SPEECH_BASE_URL:-}" ]] || fail "OPENAI_SPEECH_BASE_URL is required"
  [[ -n "${OPENAI_SPEECH_MODEL:-}" ]] || fail "OPENAI_SPEECH_MODEL is required"
  command -v setsid >/dev/null 2>&1 || fail "setsid is required"
  mkdir -p -- "${LOG_DIR}"

  local pid identity
  if pid="$(read_pid)"; then
    if process_is_ours "${pid}"; then identity=0; else identity=$?; fi
    if (( identity == 0 )); then
      health_check && { info "already ready PID=${pid}"; return 0; }
      fail "managed process is running but unhealthy; refusing duplicate start"
    elif (( identity == 1 )); then
      rm -f -- "${PID_FILE}"
    else
      fail "PID ${pid} is foreign; refusing to signal it"
    fi
  elif [[ -e "${PID_FILE}" ]]; then
    fail "PID file is malformed"
  fi
  port_is_open && fail "bridge port is occupied by an unmanaged process"

  cd -- "${REPO_ROOT}"
  nohup setsid "${OPENAI_SPEECH_PYTHON}" -m "${SERVER_MODULE}" \
    8>&- >>"${LOG_FILE}" 2>&1 </dev/null &
  pid=$!
  printf '%s\n' "${pid}" >"${PID_FILE}.tmp"
  mv -f -- "${PID_FILE}.tmp" "${PID_FILE}"

  local elapsed=0
  while (( elapsed < OPENAI_SPEECH_START_TIMEOUT_SEC )); do
    kill -0 "${pid}" 2>/dev/null || {
      rm -f -- "${PID_FILE}"
      fail "bridge exited during startup; inspect ${LOG_FILE}"
    }
    health_check && { info "ready PID=${pid} port=${OPENAI_SPEECH_BRIDGE_PORT}"; return 0; }
    sleep 1
    (( elapsed += 1 ))
  done
  kill -TERM "${pid}" 2>/dev/null || true
  if wait_for_exit "${pid}" "${OPENAI_SPEECH_STOP_TIMEOUT_SEC}"; then
    rm -f -- "${PID_FILE}"
    fail "startup timeout; candidate stopped; inspect ${LOG_FILE}"
  fi
  fail "startup and graceful-stop timed out; PID file retained for safe recovery"
}

stop_bridge() {
  validate_config
  local pid identity
  if ! pid="$(read_pid)"; then
    [[ -e "${PID_FILE}" ]] && fail "PID file is malformed"
    info "already stopped"
    return 0
  fi
  if process_is_ours "${pid}"; then identity=0; else identity=$?; fi
  (( identity == 0 )) || fail "PID ${pid} is not the managed bridge"
  kill -TERM "${pid}"
  wait_for_exit "${pid}" "${OPENAI_SPEECH_STOP_TIMEOUT_SEC}" \
    || fail "graceful stop timed out; process was not force-killed"
  rm -f -- "${PID_FILE}"
  info "stopped cleanly"
}

status_bridge() {
  validate_config
  local pid identity
  if ! pid="$(read_pid)"; then info "status=stopped"; return 3; fi
  if process_is_ours "${pid}"; then identity=0; else identity=$?; fi
  (( identity == 0 )) || fail "status=foreign-pid PID=${pid}"
  health_check && { info "status=ready PID=${pid}"; return 0; }
  info "status=running health=failed PID=${pid}"
  return 4
}

load_env
acquire_lock
case "${ACTION}" in
  start) start_bridge ;;
  stop) stop_bridge ;;
  restart) stop_bridge; start_bridge ;;
  status) status_bridge ;;
  *) fail "usage: $0 {start|stop|restart|status} [env-file]" ;;
esac
