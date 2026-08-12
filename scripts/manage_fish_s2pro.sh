#!/usr/bin/env bash
# Lifecycle manager for the Fish Speech S2 Pro SGLang-Omni server.

set -Eeuo pipefail
set +x
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
readonly REPO_ROOT
readonly LAUNCHER="${SCRIPT_DIR}/run_fish_s2pro_dual_gpu.sh"
readonly LOG_DIR="${REPO_ROOT}/logs"
readonly PID_FILE="${LOG_DIR}/fish_s2pro.pid"
readonly IDENTITY_FILE="${LOG_DIR}/fish_s2pro.identity"
readonly PENDING_FILE="${LOG_DIR}/fish_s2pro.pending"
readonly LOG_FILE="${LOG_DIR}/fish_s2pro.log"
readonly LOCK_FILE="${LOG_DIR}/fish_s2pro.lock"

ACTION="${1:-status}"
ENV_FILE="${2:-${FISH_S2PRO_ENV_FILE:-${SCRIPT_DIR}/.env.fish_s2pro}}"
readonly ACTION ENV_FILE

info() { printf '[fish-s2pro] %s\n' "$*"; }
fail() { printf '[fish-s2pro] ERROR: %s\n' "$*" >&2; exit 1; }

load_env() {
  # Never expose values from a private env file, even if xtrace was inherited.
  exec 9>/dev/null
  BASH_XTRACEFD=9
  set +x
  if [[ -f "${ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}" >/dev/null 2>&1 || fail "could not load env file"
    set +a
  elif [[ "${ACTION}" == "start" || "${ACTION}" == "restart" ]]; then
    fail "env file does not exist: ${ENV_FILE}"
  fi
  unset BASH_XTRACEFD
  exec 9>&-

  : "${FISH_PROFILE:=low_ttfa_gapless}"
  : "${FISH_HTTP_HOST:=127.0.0.1}"
  : "${FISH_HTTP_PORT:=8001}"
  : "${FISH_HEALTH_PATH:=/health}"
  : "${FISH_START_TIMEOUT_SEC:=900}"
  : "${FISH_STOP_TIMEOUT_SEC:=120}"
  export FISH_PROFILE FISH_HTTP_HOST FISH_HTTP_PORT FISH_HEALTH_PATH
}

validate_config() {
  local dystream_devices="${CUDA_VISIBLE_DEVICES:-}"
  local fish_devices="${FISH_CUDA_VISIBLE_DEVICES:-}"
  local -a dystream_gpu_ids=() fish_gpu_ids=()
  [[ "${FISH_HTTP_HOST}" == "127.0.0.1" ]] \
    || fail "FISH_HTTP_HOST must be 127.0.0.1"
  [[ "${FISH_HTTP_PORT}" =~ ^[0-9]+$ ]] \
    && (( FISH_HTTP_PORT >= 1 && FISH_HTTP_PORT <= 65535 )) \
    || fail "FISH_HTTP_PORT must be between 1 and 65535"
  [[ "${FISH_HEALTH_PATH}" == /* && "${FISH_HEALTH_PATH}" != *$'\n'* ]] \
    || fail "FISH_HEALTH_PATH must be an absolute single-line URL path"
  [[ "${FISH_START_TIMEOUT_SEC}" =~ ^[1-9][0-9]*$ ]] \
    || fail "FISH_START_TIMEOUT_SEC must be a positive integer"
  [[ "${FISH_STOP_TIMEOUT_SEC}" =~ ^[1-9][0-9]*$ ]] \
    || fail "FISH_STOP_TIMEOUT_SEC must be a positive integer"
  IFS=',' read -r -a fish_gpu_ids <<<"${fish_devices}"
  (( ${#fish_gpu_ids[@]} == 2 )) \
    || fail "FISH_CUDA_VISIBLE_DEVICES must contain exactly two GPU ids"
  fish_gpu_ids[0]="${fish_gpu_ids[0]//[[:space:]]/}"
  fish_gpu_ids[1]="${fish_gpu_ids[1]//[[:space:]]/}"
  [[ "${fish_gpu_ids[0]}" =~ ^[0-9]+$ \
        && "${fish_gpu_ids[1]}" =~ ^[0-9]+$ \
        && "${fish_gpu_ids[0]}" != "${fish_gpu_ids[1]}" ]] \
    || fail "FISH_CUDA_VISIBLE_DEVICES must contain two distinct numeric GPU ids"
  if [[ -n "${dystream_devices}" ]]; then
    IFS=',' read -r -a dystream_gpu_ids <<<"${dystream_devices}"
    for fish_gpu in "${fish_gpu_ids[@]}"; do
      for dystream_gpu in "${dystream_gpu_ids[@]}"; do
        dystream_gpu="${dystream_gpu//[[:space:]]/}"
        [[ "${fish_gpu}" != "${dystream_gpu}" ]] \
          || fail "Fish and DyStream CUDA_VISIBLE_DEVICES must not overlap"
      done
    done
  fi
}

acquire_lock() {
  command -v flock >/dev/null 2>&1 || fail "flock is required"
  mkdir -p -- "${LOG_DIR}"
  exec 8>"${LOCK_FILE}"
  flock -n 8 || fail "another Fish lifecycle action is already running"
}

read_pid() {
  [[ -f "${PID_FILE}" ]] || return 1
  local pid
  IFS= read -r pid <"${PID_FILE}" || return 1
  [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || return 2
  printf '%s' "${pid}"
}

read_identity() {
  [[ -f "${IDENTITY_FILE}" ]] || return 1
  local -a fields=()
  mapfile -t fields <"${IDENTITY_FILE}" || return 1
  (( ${#fields[@]} == 6 )) || return 1
  [[ "${fields[0]}" == "v1" ]] || return 1
  [[ "${fields[1]}" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ "${fields[2]}" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ "${fields[3]}" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ "${fields[4]}" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ "${fields[5]}" =~ ^[A-Za-z0-9._-]+$ ]] || return 1
  printf '%s\n' \
    "${fields[1]}" "${fields[2]}" "${fields[3]}" "${fields[4]}" "${fields[5]}"
}

proc_details() {
  local pid="$1" raw rest
  local -a fields=()
  [[ -r "/proc/${pid}/stat" ]] || return 1
  raw="$(<"/proc/${pid}/stat")" || return 1
  rest="${raw##*) }"
  read -r -a fields <<<"${rest}"
  (( ${#fields[@]} >= 20 )) || return 1
  # Fields after comm start at proc field 3: state, pgrp, session, starttime.
  printf '%s\n%s\n%s\n%s\n' \
    "${fields[0]}" "${fields[2]}" "${fields[3]}" "${fields[19]}"
}

process_has_token() {
  local pid="$1" expected="$2" entry
  local -a environment=()
  [[ -r "/proc/${pid}/environ" ]] || return 1
  mapfile -d '' -t environment <"/proc/${pid}/environ" || return 1
  for entry in "${environment[@]}"; do
    [[ "${entry}" == "FLASHAV2AV_FISH_MANAGER_TOKEN=${expected}" ]] && return 0
  done
  return 1
}

# Return 0 for our live leader, 1 for a stopped leader, and 2 for a live PID
# whose identity cannot be proven. Callers never signal status 2.
process_is_ours() {
  local pid="$1"
  local identity_text details_text
  local -a identity=() details=()
  [[ -d "/proc/${pid}" ]] && kill -0 "${pid}" 2>/dev/null || return 1
  details_text="$(proc_details "${pid}")" || return 2
  mapfile -t details <<<"${details_text}"
  [[ "${details[0]}" != "Z" ]] || return 1
  identity_text="$(read_identity)" || return 2
  mapfile -t identity <<<"${identity_text}"
  [[ "${identity[0]}" == "${pid}" ]] || return 2
  [[ "${identity[1]}" == "${details[1]}" ]] || return 2
  [[ "${identity[2]}" == "${details[2]}" ]] || return 2
  [[ "${identity[3]}" == "${details[3]}" ]] || return 2
  [[ "${identity[1]}" == "${pid}" && "${identity[2]}" == "${pid}" ]] \
    || return 2
  process_has_token "${pid}" "${identity[4]}" || return 2
}

health_url() {
  local url_host="${FISH_HTTP_HOST}"
  [[ "${url_host}" == *:* ]] && url_host="[${url_host}]"
  printf 'http://%s:%s%s' "${url_host}" "${FISH_HTTP_PORT}" "${FISH_HEALTH_PATH}"
}

health_check() {
  curl --fail --silent --show-error \
    --connect-timeout 1 --max-time 3 "$(health_url)" >/dev/null 2>&1
}

port_is_open() {
  local url_host="${FISH_HTTP_HOST}"
  [[ "${url_host}" == *:* ]] && url_host="[${url_host}]"
  curl --silent --output /dev/null \
    --connect-timeout 1 --max-time 2 "http://${url_host}:${FISH_HTTP_PORT}/"
}

group_is_alive() {
  local pgid="$1"
  kill -0 -- "-${pgid}" 2>/dev/null
}

wait_for_group_exit() {
  local pgid="$1" timeout="$2" elapsed=0
  while group_is_alive "${pgid}"; do
    (( elapsed >= timeout )) && return 1
    sleep 1
    (( elapsed += 1 ))
  done
}

wait_for_pid_exit() {
  local pid="$1" timeout="$2" elapsed=0 details_text
  while kill -0 "${pid}" 2>/dev/null; do
    details_text="$(proc_details "${pid}" 2>/dev/null || true)"
    [[ "${details_text%%$'\n'*}" == "Z" ]] && break
    (( elapsed >= timeout )) && return 1
    sleep 1
    (( elapsed += 1 ))
  done
  wait "${pid}" 2>/dev/null || true
}

record_pending() {
  local pid="$1" pgid="$2" session="$3" starttime="$4" token="$5" reason="$6"
  printf 'v1\n%s\n%s\n%s\n%s\n%s\n%s\n' \
    "${pid}" "${pgid}" "${session}" "${starttime}" "${token}" "${reason}" \
    >"${PENDING_FILE}.tmp" || return 1
  mv -f -- "${PENDING_FILE}.tmp" "${PENDING_FILE}" || return 1
}

read_pending() {
  [[ -f "${PENDING_FILE}" ]] || return 1
  local -a fields=()
  mapfile -t fields <"${PENDING_FILE}" || return 1
  (( ${#fields[@]} == 7 )) || return 1
  [[ "${fields[0]}" == "v1" ]] || return 1
  [[ "${fields[1]}" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ "${fields[2]}" =~ ^[0-9]+$ ]] || return 1
  [[ "${fields[3]}" =~ ^[0-9]+$ ]] || return 1
  [[ "${fields[4]}" =~ ^[0-9]+$ ]] || return 1
  [[ "${fields[5]}" =~ ^[A-Za-z0-9._-]+$ ]] || return 1
  [[ "${fields[6]}" =~ ^[A-Za-z0-9._-]+$ ]] || return 1
  printf '%s\n' \
    "${fields[1]}" "${fields[2]}" "${fields[3]}" "${fields[4]}" \
    "${fields[5]}" "${fields[6]}"
}

# Return 0 for a verified live pending process, 1 when the recorded process is
# gone or the PID was reused, and 2 when a live process cannot be proven ours.
pending_process_status() {
  local pending_text details_text pid
  local -a pending=() details=()
  pending_text="$(read_pending)" || return 2
  mapfile -t pending <<<"${pending_text}"
  pid="${pending[0]}"
  [[ -d "/proc/${pid}" ]] && kill -0 "${pid}" 2>/dev/null || return 1
  details_text="$(proc_details "${pid}")" || return 2
  mapfile -t details <<<"${details_text}"
  [[ "${details[0]}" != "Z" ]] || return 1
  (( pending[3] > 0 )) || return 2
  [[ "${pending[3]}" == "${details[3]}" ]] || return 1
  process_has_token "${pid}" "${pending[4]}" || return 2
  if (( pending[1] > 0 )); then
    [[ "${pending[1]}" == "${details[1]}" ]] || return 2
  fi
  if (( pending[2] > 0 )); then
    [[ "${pending[2]}" == "${details[2]}" ]] || return 2
  fi
}

recover_pending() {
  local pending_status pending_text pid
  local -a pending=()
  if pending_process_status; then pending_status=0; else pending_status=$?; fi
  if (( pending_status == 1 )); then
    remove_state
    info "cleared stale Fish recovery state"
    return 0
  fi
  (( pending_status == 0 )) \
    || fail "Fish recovery state is live but unverifiable; no signal was sent"
  pending_text="$(read_pending)" || fail "Fish recovery state is malformed"
  mapfile -t pending <<<"${pending_text}"
  pid="${pending[0]}"
  # Revalidate immediately before signalling the recorded process.
  pending_process_status \
    || fail "Fish recovery identity changed; no signal was sent"
  if [[ "${pending[1]}" == "${pid}" && "${pending[2]}" == "${pid}" ]]; then
    kill -TERM -- "-${pending[1]}"
    wait_for_group_exit "${pending[1]}" "${FISH_STOP_TIMEOUT_SEC}" \
      || fail "Fish recovery stop timed out; state was retained"
  else
    kill -TERM "${pid}"
    wait_for_pid_exit "${pid}" "${FISH_STOP_TIMEOUT_SEC}" \
      || fail "Fish recovery stop timed out; state was retained"
  fi
  remove_state
  info "recovered and stopped Fish process PID=${pid}"
}

terminate_new_process() {
  local pid="$1" pgid="$2" session="$3" starttime="$4" token="$5" reason="$6"
  local verified_group=0 details_text pending_written=0
  local -a actual=()
  details_text="$(proc_details "${pid}" 2>/dev/null || true)"
  if [[ -n "${details_text}" ]]; then
    mapfile -t actual <<<"${details_text}"
    pgid="${actual[1]}"
    session="${actual[2]}"
    starttime="${actual[3]}"
  fi
  if (( ${#actual[@]} != 4 )) || ! process_has_token "${pid}" "${token}"; then
    record_pending "${pid}" "${pgid}" "${session}" "${starttime}" "${token}" "${reason}" \
      && pending_written=1
    (( pending_written == 1 )) \
      || fail "could not persist Fish recovery state; existing state files were retained"
    return 1
  fi
  if [[ "${pgid}" == "${pid}" && "${session}" == "${pid}" ]]; then
    verified_group=1
    kill -TERM -- "-${pgid}" 2>/dev/null || true
    if wait_for_group_exit "${pgid}" "${FISH_STOP_TIMEOUT_SEC}"; then
      rm -f -- "${PENDING_FILE}" "${PENDING_FILE}.tmp"
      return 0
    fi
  else
    kill -TERM "${pid}" 2>/dev/null || true
    if wait_for_pid_exit "${pid}" "${FISH_STOP_TIMEOUT_SEC}"; then
      rm -f -- "${PENDING_FILE}" "${PENDING_FILE}.tmp"
      return 0
    fi
  fi
  record_pending "${pid}" "${pgid}" "${session}" "${starttime}" "${token}" "${reason}" \
    && pending_written=1
  (( pending_written == 1 )) \
    || fail "could not persist Fish recovery state; existing state files were retained"
  (( verified_group == 0 )) || return 1
  return 1
}

remove_state() {
  rm -f -- "${PID_FILE}" "${IDENTITY_FILE}" "${PENDING_FILE}" \
    "${PID_FILE}.tmp" "${IDENTITY_FILE}.tmp" "${PENDING_FILE}.tmp"
}

write_state() {
  local pid="$1" pgid="$2" session="$3" starttime="$4" token="$5"
  printf 'v1\n%s\n%s\n%s\n%s\n%s\n' \
    "${pid}" "${pgid}" "${session}" "${starttime}" "${token}" \
    >"${IDENTITY_FILE}.tmp" || return 1
  printf '%s\n' "${pid}" >"${PID_FILE}.tmp" || return 1
  mv -f -- "${IDENTITY_FILE}.tmp" "${IDENTITY_FILE}" || return 1
  mv -f -- "${PID_FILE}.tmp" "${PID_FILE}" || return 1
}

stop_owned_process() {
  local pid="$1" identity_status identity_text
  local -a identity=()
  if process_is_ours "${pid}"; then identity_status=0; else identity_status=$?; fi
  if (( identity_status == 1 )); then
    remove_state
    info "already stopped (removed stale state)"
    return 0
  fi
  (( identity_status == 0 )) \
    || fail "PID ${pid} is foreign or unverifiable; refusing to signal it"
  identity_text="$(read_identity)" || fail "managed identity is malformed"
  mapfile -t identity <<<"${identity_text}"

  # Revalidate immediately before addressing the whole session process group.
  process_is_ours "${pid}" \
    || fail "PID ${pid} changed identity; refusing to signal its process group"
  kill -TERM -- "-${identity[1]}"
  if ! wait_for_group_exit "${identity[1]}" "${FISH_STOP_TIMEOUT_SEC}"; then
    fail "graceful stop timed out; process group was not force-killed"
  fi
  remove_state
  info "stopped cleanly"
}

start_server() {
  validate_config
  command -v curl >/dev/null 2>&1 || fail "curl is required"
  command -v setsid >/dev/null 2>&1 || fail "setsid is required"
  [[ -f "${LAUNCHER}" ]] || fail "missing launcher: ${LAUNCHER}"

  local pid identity_status token elapsed=0 details_text identity_text
  local -a details=()
  if [[ -e "${PENDING_FILE}" ]]; then
    if pending_process_status; then
      fail "a Fish process needs recovery; run stop before starting"
    else
      identity_status=$?
      if (( identity_status == 1 )); then
        remove_state
        info "cleared stale Fish recovery state before start"
      else
        fail "Fish recovery state is unverifiable; refusing to start"
      fi
    fi
  fi
  if pid="$(read_pid)"; then
    if process_is_ours "${pid}"; then identity_status=0; else identity_status=$?; fi
    if (( identity_status == 0 )); then
      health_check && { info "already ready PID=${pid}"; return 0; }
      fail "managed server is running but unhealthy; refusing duplicate start"
    elif (( identity_status == 1 )); then
      remove_state
    else
      fail "PID ${pid} is foreign or unverifiable; refusing to overwrite it"
    fi
  elif [[ -e "${PID_FILE}" ]]; then
    fail "PID file is malformed; refusing to overwrite it"
  elif [[ -e "${IDENTITY_FILE}" ]]; then
    identity_text="$(read_identity 2>/dev/null || true)"
    [[ -n "${identity_text}" ]] \
      || fail "identity file is malformed; refusing to overwrite it"
    fail "identity exists without a PID file; run stop to recover it"
  fi
  port_is_open && fail "Fish HTTP port is occupied by an unmanaged process"

  token="$(date +%s).$$.${RANDOM}.${RANDOM}"
  cd -- "${REPO_ROOT}"
  FLASHAV2AV_FISH_MANAGER_TOKEN="${token}" \
    nohup setsid bash "${LAUNCHER}" "${FISH_PROFILE}" \
    8>&- >>"${LOG_FILE}" 2>&1 </dev/null &
  pid=$!

  while (( elapsed < 5 )); do
    details=()
    details_text="$(proc_details "${pid}" 2>/dev/null || true)"
    [[ -z "${details_text}" ]] || mapfile -t details <<<"${details_text}"
    if (( ${#details[@]} == 4 )) \
      && [[ "${details[1]}" == "${pid}" && "${details[2]}" == "${pid}" ]] \
      && process_has_token "${pid}" "${token}"; then
      break
    fi
    kill -0 "${pid}" 2>/dev/null || fail "Fish server exited during launch; inspect ${LOG_FILE}"
    sleep 1
    (( elapsed += 1 ))
  done
  if (( ${#details[@]} != 4 )); then
    terminate_new_process "${pid}" "0" "0" "0" "${token}" "inspection-failed" \
      || fail "could not inspect launched Fish process; recovery state was retained"
    fail "could not inspect launched Fish process; direct child was terminated"
  fi
  [[ "${details[1]}" == "${pid}" && "${details[2]}" == "${pid}" ]] || {
    terminate_new_process "${pid}" "${details[1]}" "${details[2]}" \
      "${details[3]}" "${token}" "setsid-validation-failed" \
      || fail "setsid validation failed; recovery state was retained"
    fail "setsid did not create an isolated Fish process group"
  }
  process_has_token "${pid}" "${token}" || {
    terminate_new_process "${pid}" "${details[1]}" "${details[2]}" \
      "${details[3]}" "${token}" "token-validation-failed" \
      || fail "token validation failed; recovery state was retained"
    fail "launched Fish process identity could not be verified"
  }
  if ! write_state "${pid}" "${details[1]}" "${details[2]}" "${details[3]}" "${token}"; then
    terminate_new_process "${pid}" "${details[1]}" "${details[2]}" \
      "${details[3]}" "${token}" "state-write-failed" \
      || fail "state write failed; recovery state was retained"
    remove_state
    fail "state write failed; launched Fish process was stopped"
  fi

  elapsed=0
  while (( elapsed < FISH_START_TIMEOUT_SEC )); do
    local identity_status=0
    if process_is_ours "${pid}"; then
      identity_status=0
    else
      identity_status=$?
    fi
    if (( identity_status == 1 )); then
      remove_state
      fail "Fish server exited during startup; inspect ${LOG_FILE}"
    elif (( identity_status == 2 )); then
      fail "Fish process identity became unverifiable; state was retained and no signal was sent"
    fi
    health_check && { info "ready PID=${pid} endpoint=$(health_url)"; return 0; }
    sleep 1
    (( elapsed += 1 ))
  done
  if process_is_ours "${pid}"; then
    identity_status=0
  else
    identity_status=$?
  fi
  if (( identity_status == 0 )); then
    local -a identity=()
    identity_text="$(read_identity)" || fail "managed identity is malformed"
    mapfile -t identity <<<"${identity_text}"
    kill -TERM -- "-${identity[1]}"
    if ! wait_for_group_exit "${identity[1]}" "${FISH_STOP_TIMEOUT_SEC}"; then
      fail "startup timed out and graceful stop timed out; no SIGKILL was sent"
    fi
    remove_state
    fail "startup timed out; Fish server stopped; inspect ${LOG_FILE}"
  elif (( identity_status == 1 )); then
    remove_state
    fail "Fish server exited at startup timeout; inspect ${LOG_FILE}"
  fi
  fail "Fish identity became unverifiable at startup timeout; state was retained"
}

stop_server() {
  validate_config
  local pid identity_status identity_text pending_status
  local -a identity=()
  if [[ -e "${PENDING_FILE}" ]]; then
    recover_pending
    return 0
  fi
  if ! pid="$(read_pid)"; then
    [[ -e "${PID_FILE}" ]] && fail "PID file is malformed; refusing to signal any process"
    if [[ ! -e "${IDENTITY_FILE}" ]]; then
      info "already stopped"
      return 0
    fi
    identity_text="$(read_identity 2>/dev/null || true)"
    [[ -n "${identity_text}" ]] \
      || fail "identity exists without a valid PID file; state was retained"
    mapfile -t identity <<<"${identity_text}"
    pid="${identity[0]}"
    if process_is_ours "${pid}"; then identity_status=0; else identity_status=$?; fi
    if (( identity_status == 0 )); then
      info "recovering managed Fish process from identity state"
      stop_owned_process "${pid}"
      return 0
    elif (( identity_status == 1 )); then
      remove_state
      info "already stopped (removed stale identity)"
      return 0
    fi
    fail "identity exists without a PID file and cannot be verified; state was retained"
  fi
  stop_owned_process "${pid}"
}

status_server() {
  validate_config
  command -v curl >/dev/null 2>&1 || fail "curl is required"
  local pid identity_status identity_text
  local -a identity=()
  if [[ -e "${PENDING_FILE}" ]]; then
    if pending_process_status; then
      pending_status=0
    else
      pending_status=$?
    fi
    if (( pending_status == 1 )); then
      remove_state
      info "status=stopped stale recovery state was cleared"
      return 3
    elif (( pending_status == 0 )); then
      info "status=recovery-required; run stop"
      return 4
    fi
    fail "status=unverifiable recovery state was retained"
  fi
  if ! pid="$(read_pid)"; then
    [[ -e "${PID_FILE}" ]] && fail "PID file is malformed"
    if [[ -e "${IDENTITY_FILE}" ]]; then
      identity_text="$(read_identity 2>/dev/null || true)"
      [[ -n "${identity_text}" ]] \
        || fail "status=unverifiable identity state was retained"
      mapfile -t identity <<<"${identity_text}"
      pid="${identity[0]}"
      if process_is_ours "${pid}"; then
        info "status=running recovery-required PID=${pid}"
        return 4
      fi
      identity_status=$?
      (( identity_status == 1 )) \
        && { info "status=stopped stale_identity_pid=${pid}"; return 3; }
      fail "status=unverifiable identity state was retained"
    fi
    info "status=stopped"
    return 3
  fi
  if process_is_ours "${pid}"; then identity_status=0; else identity_status=$?; fi
  (( identity_status == 0 )) || {
    (( identity_status == 1 )) && { info "status=stopped stale_pid=${pid}"; return 3; }
    fail "status=foreign-pid PID=${pid}; no signal was sent"
  }
  health_check && { info "status=ready health=ok PID=${pid}"; return 0; }
  info "status=running health=failed PID=${pid}"
  return 4
}

load_env
acquire_lock
case "${ACTION}" in
  start) start_server ;;
  stop) stop_server ;;
  restart) stop_server; start_server ;;
  status) status_server ;;
  *) fail "usage: $0 {start|stop|restart|status} [env-file]" ;;
esac
