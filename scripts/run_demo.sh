#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="${FLASHAV2AV_ROOT:-$SCRIPT_ROOT}"
cd "$ROOT_DIR"

ACTION="${1:-start}"
PORT_VALUE="${PORT:-7860}"
PYTHON_BIN="${PIPECAT_PYTHON:-$HOME/.venvs/flashav2av/bin/python}"
EXPECTED_ENGINE_VERSION="${EXPECTED_ENGINE_VERSION:-FLASHAV2AV_0.1.0}"
LAUNCH_TOKEN="${DEMO_LAUNCH_TOKEN:-manual}"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"
PID_FILE="$ROOT_DIR/logs/pipecat_mse.pid"
LOG_FILE="$ROOT_DIR/logs/pipecat_mse.log"
LOCK_FILE="$ROOT_DIR/logs/run_demo.lock"
READY_FILE="$ROOT_DIR/logs/demo_ready.pid"
START_SCRIPT="$ROOT_DIR/scripts/start_pipecat_mse.sh"
VOX_BRIDGE_SCRIPT="$ROOT_DIR/voice_service/run_bridge.sh"
FISH_BRIDGE_SCRIPT="$ROOT_DIR/voice_service/run_openai_speech_bridge.sh"
FISH_MANAGER_SCRIPT="$ROOT_DIR/scripts/manage_fish_s2pro.sh"
TTS_STATE_FILE="$ROOT_DIR/logs/tts_stack.state"
DIALOG_MODE="native_s2s"
TTS_PROVIDER="none"
TTS_BRIDGE_ENV=""
TTS_UPSTREAM_ENV=""
EXPECTED_TTS_MODEL=""

mkdir -p "$ROOT_DIR/logs"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

read_managed_pid() {
  local pid=""
  if [[ -f "$PID_FILE" ]]; then
    pid="$(tr -d '[:space:]' < "$PID_FILE" 2>/dev/null || true)"
  fi
  printf '%s' "$pid"
}

pid_is_alive() {
  local pid="$1"
  local process_state=""
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  process_state="$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null || true)"
  [[ "$process_state" != "Z" ]]
}

pid_is_managed_server() {
  local pid="$1"
  local process_cwd=""
  local process_exe=""
  local expected_exe=""
  local script_matches=0
  local port_matches=0
  local index=0
  local -a process_args=()
  pid_is_alive "$pid" || return 1
  process_cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
  process_exe="$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)"
  expected_exe="$(readlink -f "$PYTHON_BIN" 2>/dev/null || true)"
  mapfile -d '' -t process_args < "/proc/$pid/cmdline" || return 1
  [[ "$process_cwd" == "$ROOT_DIR" ]] || return 1
  [[ -n "$expected_exe" && "$process_exe" == "$expected_exe" ]] || return 1
  for ((index = 0; index < ${#process_args[@]}; index++)); do
    if [[ "${process_args[index]}" == "server_mse.py" \
          || "${process_args[index]}" == "./server_mse.py" \
          || "${process_args[index]}" == "$ROOT_DIR/server_mse.py" ]]; then
      script_matches=1
    fi
    if [[ "${process_args[index]}" == "--port" \
          && $((index + 1)) -lt ${#process_args[@]} \
          && "${process_args[index + 1]}" == "$PORT_VALUE" ]]; then
      port_matches=1
    fi
  done
  [[ "$script_matches" -eq 1 && "$port_matches" -eq 1 ]]
}

pid_is_managed_mse_instance() {
  local pid="$1"
  local expected_env_file=""
  local process_env_entry=""
  local process_env_file=""
  local process_env_port=""
  pid_is_managed_server "$pid" || return 1
  expected_env_file="$(readlink -f "$ENV_FILE" 2>/dev/null || true)"
  while IFS= read -r -d '' process_env_entry; do
    case "$process_env_entry" in
      CUSTOMIZATION_MAIN_ENV_FILE=*)
        process_env_file="${process_env_entry#*=}"
        ;;
      CUSTOMIZATION_SERVER_PORT=*)
        process_env_port="${process_env_entry#*=}"
        ;;
    esac
  done < "/proc/$pid/environ" || return 1
  process_env_file="$(readlink -f "$process_env_file" 2>/dev/null || true)"
  [[ -n "$expected_env_file" && "$process_env_file" == "$expected_env_file" ]] \
    || return 1
  [[ "$process_env_port" == "$PORT_VALUE" ]]
}

health_json() {
  curl --fail --silent --show-error --max-time 4 \
    "http://127.0.0.1:$PORT_VALUE/health"
}

validate_health_json() {
  "$PYTHON_BIN" -c '
import json
import sys

h = json.load(sys.stdin)
expected_engine_version = sys.argv[1]
expected_dialog_mode = sys.argv[2]
expected_tts_model = sys.argv[3]
dialog = h.get("dialog_session", {})
tts = (dialog.get("custom_cascade") or {}).get("tts") or {}
checks = {
    "status=ok": h.get("status") == "ok",
    "engine_version": h.get("engine_version") == expected_engine_version,
    "dialog_backend=pipecat": h.get("dialog_backend") == "pipecat",
    "feed_thread_alive": h.get("feed_thread_alive") is True,
    "frame_ready": h.get("frame_ready") is True,
    "motion_alive": h.get("workers", {}).get("motion_alive") is True,
    "render_alive": h.get("workers", {}).get("render_alive") is True,
    "dialog_ready": dialog.get("ready") is True,
    "dialog_mode": dialog.get("mode") == expected_dialog_mode,
    "provider_ready": (
        dialog.get("s2s_ready") is True
        if expected_dialog_mode == "native_s2s"
        else dialog.get("custom_cascade_ready") is True
    ),
    "tts_model": (
        True
        if expected_dialog_mode == "native_s2s"
        else tts.get("model") == expected_tts_model
    ),
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("unhealthy fields: " + ", ".join(failed))
print(
    "engine={engine} mode={mode} state={state} turn={turn} generation={generation}".format(
        engine=h.get("engine_version", "unknown"),
        mode=expected_dialog_mode,
        state=h.get("state", "unknown"),
        turn=h.get("turn_id", "unknown"),
        generation=h.get("stream_generation", "unknown"),
    )
)
' "$EXPECTED_ENGINE_VERSION" "$DIALOG_MODE" "$EXPECTED_TTS_MODEL"
}

health_summary() {
  local snapshot=""
  snapshot="$(health_json)" || return 1
  printf '%s' "$snapshot" | validate_health_json
}

require_file() {
  local path="$1"
  local label="$2"
  [[ -f "$path" ]] || die "missing $label: $path"
}

load_dialog_config() {
  if [[ -f "$ENV_FILE" ]]; then
    set +u
    set +x
    set -a
    # shellcheck disable=SC1090
    if ! . "$ENV_FILE"; then
      set +a
      set -u
      die "could not load environment file: $ENV_FILE"
    fi
    set +x
    set +a
    set -u
  fi
  DIALOG_MODE="${PIPECAT_MSE_DIALOG_MODE:-native_s2s}"
  [[ "$DIALOG_MODE" == "native_s2s" || "$DIALOG_MODE" == "custom_cascade" ]] \
    || die "PIPECAT_MSE_DIALOG_MODE must be native_s2s or custom_cascade"
  PORT_VALUE="${PORT:-7860}"
  PYTHON_BIN="${PIPECAT_PYTHON:-$PYTHON_BIN}"
  EXPECTED_ENGINE_VERSION="${EXPECTED_ENGINE_VERSION:-FLASHAV2AV_0.1.0}"
  if [[ "$DIALOG_MODE" == "custom_cascade" ]]; then
    if [[ -n "${PIPECAT_TTS_PROVIDER:-}" ]]; then
      TTS_PROVIDER="$PIPECAT_TTS_PROVIDER"
    elif [[ -n "${VOXCPM2_ENV_FILE:-}" ]]; then
      TTS_PROVIDER="voxcpm2"
    else
      TTS_PROVIDER="fish_s2pro"
    fi
    case "$TTS_PROVIDER" in
      fish_s2pro)
        TTS_BRIDGE_ENV="${PIPECAT_TTS_BRIDGE_ENV_FILE:-}"
        TTS_UPSTREAM_ENV="${FISH_S2PRO_ENV_FILE:-}"
        EXPECTED_TTS_MODEL="${PIPECAT_TTS_MODEL:-fishaudio/s2-pro}"
        ;;
      voxcpm2)
        TTS_BRIDGE_ENV="${PIPECAT_TTS_BRIDGE_ENV_FILE:-${VOXCPM2_ENV_FILE:-}}"
        TTS_UPSTREAM_ENV=""
        EXPECTED_TTS_MODEL="${PIPECAT_TTS_MODEL:-VoxCPM2}"
        ;;
      *) die "PIPECAT_TTS_PROVIDER must be fish_s2pro or voxcpm2" ;;
    esac
  else
    TTS_PROVIDER="none"
  fi
}

bridge_env_file() {
  [[ -n "$TTS_BRIDGE_ENV" ]] || die "custom_cascade requires PIPECAT_TTS_BRIDGE_ENV_FILE"
  printf '%s' "$TTS_BRIDGE_ENV"
}

write_tts_state() {
  local provider="$1" bridge_env="$2" upstream_env="${3:--}"
  local temporary="$TTS_STATE_FILE.tmp.$$"
  [[ "$provider" == "fish_s2pro" || "$provider" == "voxcpm2" ]] || return 1
  [[ "$bridge_env" == /* && "$bridge_env" != *$'\n'* ]] || return 1
  [[ "$upstream_env" == "-" || ( "$upstream_env" == /* && "$upstream_env" != *$'\n'* ) ]] || return 1
  if ! printf 'v1\n%s\n%s\n%s\n' "$provider" "$bridge_env" "$upstream_env" > "$temporary" \
    || ! chmod 600 "$temporary" \
    || ! mv -f -- "$temporary" "$TTS_STATE_FILE"; then
    rm -f -- "$temporary"
    echo "ERROR: could not persist TTS stack state" >&2
    return 1
  fi
}

read_tts_state() {
  local version="" provider="" bridge_env="" upstream_env="" extra=""
  [[ -f "$TTS_STATE_FILE" ]] || return 1
  {
    IFS= read -r version
    IFS= read -r provider
    IFS= read -r bridge_env
    IFS= read -r upstream_env
    IFS= read -r extra || true
  } < "$TTS_STATE_FILE"
  [[ "$version" == "v1" && -z "$extra" ]] || return 1
  [[ "$provider" == "fish_s2pro" || "$provider" == "voxcpm2" ]] || return 1
  [[ "$bridge_env" == /* && "$bridge_env" != *$'\n'* ]] || return 1
  [[ "$upstream_env" == "-" || "$upstream_env" == /* ]] || return 1
  printf '%s\n%s\n%s\n' "$provider" "$bridge_env" "$upstream_env"
}

start_tts_stack_if_custom() {
  local -a existing_state=()
  local state_payload=""
  local state_created=0
  local upstream="-"
  [[ "$DIALOG_MODE" == "custom_cascade" ]] || return 0
  require_file "$(bridge_env_file)" "TTS bridge environment"
  if [[ "$TTS_PROVIDER" == "fish_s2pro" ]]; then
    require_file "$FISH_MANAGER_SCRIPT" "Fish S2 Pro manager"
    require_file "$FISH_BRIDGE_SCRIPT" "Fish PCM bridge manager"
    require_file "$TTS_UPSTREAM_ENV" "Fish S2 Pro environment"
    upstream="$TTS_UPSTREAM_ENV"
  else
    require_file "$VOX_BRIDGE_SCRIPT" "VoxCPM2 bridge manager"
  fi
  if [[ -f "$TTS_STATE_FILE" ]]; then
    state_payload="$(read_tts_state)" \
      || die "existing TTS ownership state is malformed; refusing to overwrite it"
    mapfile -t existing_state <<< "$state_payload"
    [[ ${#existing_state[@]} -eq 3 \
          && "${existing_state[0]}" == "$TTS_PROVIDER" \
          && "${existing_state[1]}" == "$TTS_BRIDGE_ENV" \
          && "${existing_state[2]}" == "$upstream" ]] \
      || die "configured TTS stack differs from the owned stack; use restart"
  else
    write_tts_state "$TTS_PROVIDER" "$TTS_BRIDGE_ENV" "$upstream" \
      || die "could not record TTS ownership before startup"
    state_created=1
  fi
  if [[ "$TTS_PROVIDER" == "fish_s2pro" ]]; then
    if ! bash "$FISH_MANAGER_SCRIPT" start "$TTS_UPSTREAM_ENV"; then
      # The Fish manager may intentionally retain a verified recovery record
      # after a partial start. Keep our ownership state so the unified `stop`
      # command can invoke that recovery path instead of orphaning the stack.
      return 1
    fi
    if ! OPENAI_SPEECH_INSTANCE=flashav2av \
      bash "$FISH_BRIDGE_SCRIPT" start "$TTS_BRIDGE_ENV"; then
      if bash "$FISH_MANAGER_SCRIPT" stop "$TTS_UPSTREAM_ENV"; then
        rm -f -- "$TTS_STATE_FILE"
      fi
      return 1
    fi
  else
    if ! bash "$VOX_BRIDGE_SCRIPT" start "$TTS_BRIDGE_ENV"; then
      if [[ "$state_created" -eq 1 ]]; then
        rm -f -- "$TTS_STATE_FILE"
      fi
      return 1
    fi
  fi
}

stop_owned_tts_stack_if_present() {
  local -a state=()
  local state_payload=""
  [[ -f "$TTS_STATE_FILE" ]] || return 0
  state_payload="$(read_tts_state)" \
    || { echo "ERROR: malformed TTS ownership state" >&2; return 1; }
  mapfile -t state <<< "$state_payload"
  [[ ${#state[@]} -eq 3 ]] \
    || { echo "ERROR: malformed TTS ownership state" >&2; return 1; }
  if [[ "${state[0]}" == "fish_s2pro" ]]; then
    require_file "$FISH_BRIDGE_SCRIPT" "Fish PCM bridge manager"
    require_file "$FISH_MANAGER_SCRIPT" "Fish S2 Pro manager"
    OPENAI_SPEECH_INSTANCE=flashav2av \
      bash "$FISH_BRIDGE_SCRIPT" stop "${state[1]}" || return 1
    bash "$FISH_MANAGER_SCRIPT" stop "${state[2]}" || return 1
  else
    require_file "$VOX_BRIDGE_SCRIPT" "VoxCPM2 bridge manager"
    bash "$VOX_BRIDGE_SCRIPT" stop "${state[1]}" || return 1
  fi
  rm -f -- "$TTS_STATE_FILE"
}

status_tts_stack_if_custom() {
  local -a state=()
  local expected_upstream="-"
  local state_payload=""
  [[ "$DIALOG_MODE" == "custom_cascade" ]] || return 0
  state_payload="$(read_tts_state)" \
    || die "custom cascade has no valid TTS ownership state"
  mapfile -t state <<< "$state_payload"
  [[ ${#state[@]} -eq 3 ]] \
    || die "custom cascade has malformed TTS ownership state"
  if [[ "$TTS_PROVIDER" == "fish_s2pro" ]]; then
    expected_upstream="$TTS_UPSTREAM_ENV"
  fi
  [[ "${state[0]}" == "$TTS_PROVIDER" \
        && "${state[1]}" == "$TTS_BRIDGE_ENV" \
        && "${state[2]}" == "$expected_upstream" ]] \
    || die "running TTS stack differs from the configured stack; use restart"
  if [[ "$TTS_PROVIDER" == "fish_s2pro" ]]; then
    require_file "$FISH_MANAGER_SCRIPT" "Fish S2 Pro manager"
    require_file "$FISH_BRIDGE_SCRIPT" "Fish PCM bridge manager"
    bash "$FISH_MANAGER_SCRIPT" status "${state[2]}"
    OPENAI_SPEECH_INSTANCE=flashav2av \
      bash "$FISH_BRIDGE_SCRIPT" status "${state[1]}"
  else
    require_file "$VOX_BRIDGE_SCRIPT" "VoxCPM2 bridge manager"
    bash "$VOX_BRIDGE_SCRIPT" status "${state[1]}"
  fi
}

cleanup_failed_start() {
  local pid=""
  if stop_demo; then
    return 0
  fi
  pid="$(read_managed_pid)"
  if [[ -z "$pid" ]] || ! pid_is_alive "$pid" || ! pid_is_managed_server "$pid"; then
    stop_owned_tts_stack_if_present || true
  fi
}

cleanup_failed_mse_start() {
  # The MSE-only lifecycle never owns a speech bridge or TTS engine.
  stop_mse || true
}

preflight() {
  local manage_tts="${1:-1}"
  local required_command=""
  local visible_gpus=""
  local gpu_count=""
  local ref_image=""
  local encoder_list=""

  for required_command in curl ffmpeg ffprobe flock nvidia-smi readlink; do
    command -v "$required_command" >/dev/null 2>&1 \
      || die "missing command: $required_command"
  done
  [[ -x "$PYTHON_BIN" ]] || die "missing Pipecat Python: $PYTHON_BIN"
  require_file "$ENV_FILE" "environment file"
  require_file "$START_SCRIPT" "start script"

  [[ -n "${PIPECAT_S2S_API_KEY:-${DASHSCOPE_API_KEY:-${PIPECAT_LLM_API_KEY:-${OPENAI_API_KEY:-}}}}" ]] \
    || die "missing PIPECAT_S2S_API_KEY (or reusable DashScope/LLM key) in $ENV_FILE"

  [[ "$manage_tts" == "0" || "$manage_tts" == "1" ]] \
    || die "internal error: manage_tts must be 0 or 1"
  if [[ "$DIALOG_MODE" == "custom_cascade" ]]; then
    [[ -n "${PIPECAT_LLM_API_KEY:-${OPENAI_API_KEY:-}}" ]] \
      || die "custom_cascade requires PIPECAT_LLM_API_KEY or OPENAI_API_KEY"
    [[ -n "${PIPECAT_LLM_MODEL:-${OPENAI_MODEL:-}}" ]] \
      || die "custom_cascade requires PIPECAT_LLM_MODEL or OPENAI_MODEL"
    if [[ "$manage_tts" == "1" ]]; then
      if [[ "$TTS_PROVIDER" == "fish_s2pro" ]]; then
        require_file "$FISH_MANAGER_SCRIPT" "Fish S2 Pro manager"
        require_file "$FISH_BRIDGE_SCRIPT" "Fish PCM bridge manager"
        require_file "$TTS_UPSTREAM_ENV" "Fish S2 Pro environment"
      else
        require_file "$VOX_BRIDGE_SCRIPT" "VoxCPM2 bridge manager"
      fi
      require_file "$(bridge_env_file)" "TTS bridge environment"
    fi
  fi

  visible_gpus="${CUDA_VISIBLE_DEVICES:-0,1}"
  IFS=',' read -r -a gpu_ids <<< "$visible_gpus"
  [[ ${#gpu_ids[@]} -eq 2 ]] \
    || die "CUDA_VISIBLE_DEVICES must expose exactly two GPUs, got: $visible_gpus"
  gpu_ids[0]="${gpu_ids[0]//[[:space:]]/}"
  gpu_ids[1]="${gpu_ids[1]//[[:space:]]/}"
  [[ -n "${gpu_ids[0]}" && -n "${gpu_ids[1]}" && "${gpu_ids[0]}" != "${gpu_ids[1]}" ]] \
    || die "CUDA_VISIBLE_DEVICES must contain two different GPU ids"
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
  [[ "$gpu_count" -ge 2 ]] || die "at least two NVIDIA GPUs are required"
  for selected_gpu in "${gpu_ids[@]}"; do
    [[ "$selected_gpu" =~ ^[0-9]+$ ]] \
      || die "CUDA_VISIBLE_DEVICES must use numeric GPU ids, got: $selected_gpu"
    (( selected_gpu < gpu_count )) \
      || die "selected GPU id $selected_gpu does not exist; detected count=$gpu_count"
  done

  require_file "$ROOT_DIR/checkpoints/last.ckpt" "motion checkpoint"
  require_file \
    "$ROOT_DIR/tools/pretrained_model/epoch=0-step=312000.ckpt" \
    "renderer checkpoint"
  require_file \
    "$ROOT_DIR/tools/hf_models/wav2vec2-base-960h/pytorch_model.bin" \
    "wav2vec2 checkpoint"
  ref_image="${DYSTREAM_REF_IMAGE:-$ROOT_DIR/assets/demo_avatar/ref.png}"
  require_file "$ref_image" "avatar reference image"

  encoder_list="$(ffmpeg -hide_banner -encoders 2>/dev/null)"
  grep -q 'libx264' <<< "$encoder_list" || die "ffmpeg lacks libx264 encoder"
  grep -qE '(^|[[:space:]])aac([[:space:]]|$)' <<< "$encoder_list" \
    || die "ffmpeg lacks AAC encoder"

  "$PYTHON_BIN" "$ROOT_DIR/scripts/check_pipecat_env.py"
  echo "PREFLIGHT_OK gpus=$visible_gpus motion_gpu=${MOTION_GPU:-0} render_gpu=${RENDER_GPU:-1}"
}

media_smoke() {
  "$PYTHON_BIN" "$ROOT_DIR/scripts/check_demo_ready.py" \
    --port "$PORT_VALUE" \
    --timeout "${DEMO_MEDIA_SMOKE_TIMEOUT_SEC:-30}" \
    --expected-engine-version "$EXPECTED_ENGINE_VERSION"
}

write_ready_marker() {
  local pid="$1"
  local temporary="$READY_FILE.tmp.$$"
  [[ "$LAUNCH_TOKEN" =~ ^[A-Za-z0-9._-]{1,128}$ ]] \
    || die "DEMO_LAUNCH_TOKEN contains unsupported characters"
  printf '%s %s %s %s\n' \
    "$pid" "$EXPECTED_ENGINE_VERSION" "$LAUNCH_TOKEN" "$TTS_PROVIDER" > "$temporary"
  chmod 600 "$temporary"
  mv -f -- "$temporary" "$READY_FILE"
}

ready_marker_matches() {
  local pid="$1"
  local marker_pid=""
  local marker_version=""
  local marker_token=""
  local marker_provider=""
  local marker_extra=""
  [[ -f "$READY_FILE" ]] || return 1
  read -r marker_pid marker_version marker_token marker_provider marker_extra \
    < "$READY_FILE" || return 1
  [[ -z "$marker_extra" \
        && "$marker_pid" == "$pid" \
        && "$marker_version" == "$EXPECTED_ENGINE_VERSION" \
        && "$marker_provider" == "$TTS_PROVIDER" ]]
}

show_status() {
  local pid=""
  local summary=""
  pid="$(read_managed_pid)"
  if [[ -z "$pid" ]]; then
    echo "DEMO_STOPPED pid_file=$PID_FILE"
    return 1
  fi
  if ! pid_is_alive "$pid"; then
    echo "DEMO_STOPPED stale_pid=$pid"
    return 1
  fi
  pid_is_managed_server "$pid" \
    || die "PID file points to an unrelated process; refusing to manage pid=$pid"
  status_tts_stack_if_custom
  summary="$(health_summary)" \
    || die "managed server pid=$pid is running but health validation failed"
  ready_marker_matches "$pid" \
    || die "server pid=$pid is healthy but has not completed the media smoke test"
  echo "DEMO_READY pid=$pid remote_port=$PORT_VALUE $summary"
}

start_demo() {
  local pid=""
  local summary=""
  rm -f -- "$READY_FILE"
  preflight
  pid="$(read_managed_pid)"
  if [[ -n "$pid" ]] && pid_is_alive "$pid"; then
    if pid_is_managed_server "$pid"; then
      status_tts_stack_if_custom
      summary="$(health_summary)" \
        || die "managed server pid=$pid is running but unhealthy; inspect $LOG_FILE"
      media_smoke \
        || die "managed server pid=$pid failed the decodable-media smoke test"
      write_ready_marker "$pid"
      echo "DEMO_READY pid=$pid remote_port=$PORT_VALUE $summary"
      return 0
    fi
    echo "WARNING: removing reused/unrelated stale PID file pid=$pid; process was not killed" >&2
    rm -f -- "$PID_FILE"
    pid=""
  fi
  if [[ -n "$pid" ]]; then
    rm -f -- "$PID_FILE"
  fi

  if "$PYTHON_BIN" -c "
import socket
s = socket.socket()
s.settimeout(0.5)
try:
    occupied = s.connect_ex(('127.0.0.1', int('$PORT_VALUE'))) == 0
finally:
    s.close()
raise SystemExit(0 if occupied else 1)
"; then
    die "port $PORT_VALUE is already occupied by an unmanaged process"
  fi

  start_tts_stack_if_custom

  if ! PIPECAT_PYTHON="$PYTHON_BIN" \
    ENV_FILE="$ENV_FILE" \
    PORT="$PORT_VALUE" \
    PIPECAT_STARTUP_WAIT_SEC="${PIPECAT_STARTUP_WAIT_SEC:-240}" \
    bash "$START_SCRIPT" 9>&-; then
    cleanup_failed_start
    die "MSE start script failed"
  fi

  pid="$(read_managed_pid)"
  if ! pid_is_managed_server "$pid"; then
    cleanup_failed_start
    die "start script returned without a valid managed server"
  fi
  if ! summary="$(health_summary)"; then
    cleanup_failed_start
    die "server started but full health validation failed; inspect $LOG_FILE"
  fi
  if ! media_smoke; then
    cleanup_failed_start
    die "server started but failed the decodable-media smoke test"
  fi
  write_ready_marker "$pid"
  echo "DEMO_READY pid=$pid remote_port=$PORT_VALUE $summary"
}

stop_demo() {
  local pid=""
  local deadline=0
  local mse_stopped=1
  local result=0
  local stop_wait_sec="${PIPECAT_STOP_WAIT_SEC:-240}"
  pid="$(read_managed_pid)"
  if [[ -z "$pid" ]]; then
    rm -f -- "$READY_FILE"
    stop_owned_tts_stack_if_present || result=$?
    echo "DEMO_STOPPED already_stopped=1"
    return "$result"
  fi
  if ! pid_is_alive "$pid"; then
    rm -f -- "$PID_FILE"
    rm -f -- "$READY_FILE"
    stop_owned_tts_stack_if_present || result=$?
    echo "DEMO_STOPPED removed_stale_pid=$pid"
    return "$result"
  fi
  if ! pid_is_managed_server "$pid"; then
    echo "ERROR: PID file points to an unrelated process; refusing to stop pid=$pid" >&2
    return 1
  fi

  if [[ ! "$stop_wait_sec" =~ ^[1-9][0-9]*$ ]]; then
    echo "WARNING: invalid PIPECAT_STOP_WAIT_SEC; using 240 seconds" >&2
    stop_wait_sec=240
  fi
  if ! kill -TERM "$pid"; then
    if pid_is_alive "$pid"; then
      echo "ERROR: could not signal managed server pid=$pid" >&2
      mse_stopped=0
      result=1
    fi
  fi
  if [[ "$mse_stopped" -eq 1 ]] && pid_is_alive "$pid"; then
    deadline=$((SECONDS + stop_wait_sec))
    while pid_is_alive "$pid" && (( SECONDS < deadline )); do
      sleep 1
    done
  fi
  if pid_is_alive "$pid"; then
    echo "ERROR: server pid=$pid did not exit after ${stop_wait_sec}s; no SIGKILL was sent" >&2
    mse_stopped=0
    result=1
  else
    rm -f -- "$PID_FILE"
    rm -f -- "$READY_FILE"
  fi
  if [[ "$mse_stopped" -eq 1 ]]; then
    stop_owned_tts_stack_if_present || result=$?
    echo "DEMO_STOPPED pid=$pid"
  else
    echo "ERROR: TTS stack was left running because its MSE client is still alive" >&2
  fi
  return "$result"
}

show_mse_status() {
  local pid=""
  local summary=""
  pid="$(read_managed_pid)"
  if [[ -z "$pid" ]]; then
    echo "DEMO_STOPPED pid_file=$PID_FILE"
    return 1
  fi
  if ! pid_is_alive "$pid"; then
    echo "DEMO_STOPPED stale_pid=$pid"
    return 1
  fi
  pid_is_managed_mse_instance "$pid" \
    || die "PID file points to an unrelated process; refusing to manage pid=$pid"
  summary="$(health_summary)" \
    || die "managed server pid=$pid is running but health validation failed"
  ready_marker_matches "$pid" \
    || die "server pid=$pid is healthy but has not completed the media smoke test"
  echo "DEMO_READY pid=$pid remote_port=$PORT_VALUE $summary"
}

start_mse() {
  local pid=""
  local summary=""
  rm -f -- "$READY_FILE"
  preflight 0
  pid="$(read_managed_pid)"
  if [[ -n "$pid" ]] && pid_is_alive "$pid"; then
    pid_is_managed_mse_instance "$pid" \
      || die "PID file points to an unrelated process; refusing to manage pid=$pid"
    summary="$(health_summary)" \
      || die "managed server pid=$pid is running but unhealthy; inspect $LOG_FILE"
    media_smoke \
      || die "managed server pid=$pid failed the decodable-media smoke test"
    write_ready_marker "$pid"
    echo "DEMO_READY pid=$pid remote_port=$PORT_VALUE $summary"
    return 0
  fi
  if [[ -n "$pid" ]]; then
    rm -f -- "$PID_FILE"
  fi

  if "$PYTHON_BIN" -c "
import socket
s = socket.socket()
s.settimeout(0.5)
try:
    occupied = s.connect_ex(('127.0.0.1', int('$PORT_VALUE'))) == 0
finally:
    s.close()
raise SystemExit(0 if occupied else 1)
"; then
    die "port $PORT_VALUE is already occupied by an unmanaged process"
  fi

  if ! PIPECAT_PYTHON="$PYTHON_BIN" \
    ENV_FILE="$ENV_FILE" \
    PORT="$PORT_VALUE" \
    PIPECAT_STARTUP_WAIT_SEC="${PIPECAT_STARTUP_WAIT_SEC:-240}" \
    bash "$START_SCRIPT" 9>&-; then
    cleanup_failed_mse_start
    die "MSE start script failed"
  fi

  pid="$(read_managed_pid)"
  if ! pid_is_managed_mse_instance "$pid"; then
    cleanup_failed_mse_start
    die "start script returned without a valid managed server"
  fi
  if ! summary="$(health_summary)"; then
    cleanup_failed_mse_start
    die "server started but full health validation failed; inspect $LOG_FILE"
  fi
  if ! media_smoke; then
    cleanup_failed_mse_start
    die "server started but failed the decodable-media smoke test"
  fi
  write_ready_marker "$pid"
  echo "DEMO_READY pid=$pid remote_port=$PORT_VALUE $summary"
}

stop_mse() {
  local pid=""
  local deadline=0
  local stop_wait_sec="${PIPECAT_STOP_WAIT_SEC:-240}"
  pid="$(read_managed_pid)"
  if [[ -z "$pid" ]]; then
    rm -f -- "$READY_FILE"
    echo "DEMO_STOPPED already_stopped=1"
    return 0
  fi
  if ! pid_is_alive "$pid"; then
    rm -f -- "$PID_FILE"
    rm -f -- "$READY_FILE"
    echo "DEMO_STOPPED removed_stale_pid=$pid"
    return 0
  fi
  if ! pid_is_managed_mse_instance "$pid"; then
    echo "ERROR: PID file points to an unrelated process; refusing to stop pid=$pid" >&2
    return 1
  fi

  if [[ ! "$stop_wait_sec" =~ ^[1-9][0-9]*$ ]]; then
    echo "WARNING: invalid PIPECAT_STOP_WAIT_SEC; using 240 seconds" >&2
    stop_wait_sec=240
  fi
  if ! kill -TERM "$pid" && pid_is_alive "$pid"; then
    echo "ERROR: could not signal managed server pid=$pid" >&2
    return 1
  fi
  if pid_is_alive "$pid"; then
    deadline=$((SECONDS + stop_wait_sec))
    while pid_is_alive "$pid" && (( SECONDS < deadline )); do
      sleep 1
    done
  fi
  if pid_is_alive "$pid"; then
    echo "ERROR: server pid=$pid did not exit after ${stop_wait_sec}s; no SIGKILL was sent" >&2
    return 1
  fi
  rm -f -- "$PID_FILE"
  rm -f -- "$READY_FILE"
  echo "DEMO_STOPPED pid=$pid"
}

case "$ACTION" in
  start | status | stop | restart | start-mse | status-mse | stop-mse | restart-mse)
    ;;
  *)
    die "usage: bash scripts/run_demo.sh [start|status|stop|restart|start-mse|status-mse|stop-mse|restart-mse]"
    ;;
esac

command -v flock >/dev/null 2>&1 || die "missing command: flock"
exec 9>"$LOCK_FILE"
flock -n 9 || die "another run_demo.sh command is already in progress"
load_dialog_config

case "$ACTION" in
  start)
    start_demo
    ;;
  status)
    show_status
    ;;
  stop)
    stop_demo
    ;;
  restart)
    stop_demo
    start_demo
    ;;
  start-mse)
    start_mse
    ;;
  status-mse)
    show_mse_status
    ;;
  stop-mse)
    stop_mse
    ;;
  restart-mse)
    stop_mse
    start_mse
    ;;
esac
