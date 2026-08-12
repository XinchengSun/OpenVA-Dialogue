#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="${FLASHAV2AV_ROOT:-$SCRIPT_ROOT}"
cd "$ROOT_DIR"

ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"
PYTHON_BIN="${PIPECAT_PYTHON:-$HOME/.venvs/flashav2av/bin/python}"
PIPECAT_CACHE_ROOT="${PIPECAT_CACHE_ROOT:-$HOME/.cache/flashav2av/pipecat}"
TORCH_CACHE_ROOT="${TORCH_HOME:-$HOME/.cache/flashav2av/torch}"
NLTK_CACHE_ROOT="${NLTK_DATA:-$HOME/.cache/flashav2av/nltk}"
PID_FILE="logs/pipecat_mse.pid"
LOG_FILE="logs/pipecat_mse.log"
PORT_VALUE="${PORT:-7860}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "missing Pipecat Python: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$ENV_FILE" ]]; then
  echo "missing env file: $ENV_FILE" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a

if [[ -z "${PIPECAT_S2S_API_KEY:-${DASHSCOPE_API_KEY:-${PIPECAT_LLM_API_KEY:-${OPENAI_API_KEY:-}}}}" ]]; then
  echo "missing PIPECAT_S2S_API_KEY (or reusable DashScope/LLM key) in $ENV_FILE" >&2
  exit 1
fi

VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-0,1}"
MOTION_LOGICAL_GPU="${MOTION_GPU:-0}"
RENDER_LOGICAL_GPU="${RENDER_GPU:-1}"
ALLOW_SHARED_DYSTREAM_GPU="${ALLOW_SHARED_DYSTREAM_GPU:-0}"
IFS=',' read -r -a gpu_ids <<< "$VISIBLE_GPUS"
if [[ ${#gpu_ids[@]} -ne 2 || "${gpu_ids[0]}" == "${gpu_ids[1]}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must expose exactly two different GPUs, got: $VISIBLE_GPUS" >&2
  exit 1
fi
if [[ "$ALLOW_SHARED_DYSTREAM_GPU" != "0" && "$ALLOW_SHARED_DYSTREAM_GPU" != "1" ]]; then
  echo "ALLOW_SHARED_DYSTREAM_GPU must be 0 or 1" >&2
  exit 1
fi
if [[ "$MOTION_LOGICAL_GPU,$RENDER_LOGICAL_GPU" != "0,1" && "$MOTION_LOGICAL_GPU,$RENDER_LOGICAL_GPU" != "1,0" && ! ( "$ALLOW_SHARED_DYSTREAM_GPU" == "1" && "$MOTION_LOGICAL_GPU" == "$RENDER_LOGICAL_GPU" && ( "$MOTION_LOGICAL_GPU" == "0" || "$MOTION_LOGICAL_GPU" == "1" ) ) ]]; then
  echo "MOTION_GPU and RENDER_GPU must use logical 0/1; sharing requires ALLOW_SHARED_DYSTREAM_GPU=1" >&2
  exit 1
fi

mkdir -p logs "$PIPECAT_CACHE_ROOT"/{huggingface,modelscope,piper,xdg} \
  "$TORCH_CACHE_ROOT" "$NLTK_CACHE_ROOT"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Pipecat MSE server is already running pid=$old_pid" >&2
    exit 1
  fi
fi

nohup env \
  CUSTOMIZATION_MAIN_ENV_FILE="$ENV_FILE" \
  CUSTOMIZATION_SERVER_PORT="$PORT_VALUE" \
  DIALOG_BACKEND="${DIALOG_BACKEND:-pipecat}" \
  PIPECAT_S2S_MODEL="${PIPECAT_S2S_MODEL:-qwen-audio-3.0-realtime-flash}" \
  PIPECAT_S2S_VOICE="${PIPECAT_S2S_VOICE:-longanqian}" \
  PIPECAT_S2S_TURN_DETECTION="${PIPECAT_S2S_TURN_DETECTION:-smart_turn}" \
  PIPECAT_S2S_VAD_SILENCE_MS="${PIPECAT_S2S_VAD_SILENCE_MS:-500}" \
  PIPECAT_S2S_VAD_THRESHOLD="${PIPECAT_S2S_VAD_THRESHOLD:-0.5}" \
  CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}" \
  CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS" \
  TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  MODELSCOPE_DISABLE_AUTO_UPDATE="${MODELSCOPE_DISABLE_AUTO_UPDATE:-1}" \
  ALLOW_SHARED_DYSTREAM_GPU="$ALLOW_SHARED_DYSTREAM_GPU" \
  MOTION_GPU="$MOTION_LOGICAL_GPU" \
  RENDER_GPU="$RENDER_LOGICAL_GPU" \
  PIPE_FRAME_STRIDE="${PIPE_FRAME_STRIDE:-2}" \
  PIPE_GOP="${PIPE_GOP:-3}" \
  DYSTREAM_AUDIO_Q="${DYSTREAM_AUDIO_Q:-8}" \
  DYSTREAM_MOTION_Q="${DYSTREAM_MOTION_Q:-4}" \
  DYSTREAM_FRAME_Q="${DYSTREAM_FRAME_Q:-20}" \
  ENGINE_IDLE_WARMUP_SEGMENTS="${ENGINE_IDLE_WARMUP_SEGMENTS:-16}" \
  ENGINE_IDLE_CONTINUOUS="${ENGINE_IDLE_CONTINUOUS:-1}" \
  ENGINE_IDLE_VISIBLE="${ENGINE_IDLE_VISIBLE:-1}" \
  ENGINE_IDLE_ONLY_WITH_CLIENT="${ENGINE_IDLE_ONLY_WITH_CLIENT:-1}" \
  ENGINE_IDLE_COMPACT_SEGMENTS="${ENGINE_IDLE_COMPACT_SEGMENTS:-40}" \
  ENGINE_IDLE_INFLIGHT_HIGH_WATER_SEC="${ENGINE_IDLE_INFLIGHT_HIGH_WATER_SEC:-0.32}" \
  ENGINE_IDLE_AUDIO_Q_MAX="${ENGINE_IDLE_AUDIO_Q_MAX:-1}" \
  ENGINE_IDLE_MOTION_Q_MAX="${ENGINE_IDLE_MOTION_Q_MAX:-1}" \
  ENGINE_IDLE_FRAME_Q_MAX="${ENGINE_IDLE_FRAME_Q_MAX:-8}" \
  DYSTREAM_AUDIO_HISTORY_KEEP_SEC="${DYSTREAM_AUDIO_HISTORY_KEEP_SEC:-4.0}" \
  ENGINE_LISTENER_AUDIO="${ENGINE_LISTENER_AUDIO:-1}" \
  ENGINE_LISTENER_VIRTUAL_AUDIO="${ENGINE_LISTENER_VIRTUAL_AUDIO:-$ROOT_DIR/wav_files/_sgIH81kj78-Scene-005+audio_v3_0.wav}" \
  ENGINE_USER_SPEAKING_RMS="${ENGINE_USER_SPEAKING_RMS:-0.006}" \
  ENGINE_USER_SPEAKING_HOLD_SEC="${ENGINE_USER_SPEAKING_HOLD_SEC:-0.6}" \
  ENGINE_MAX_USER_AUDIO_BUF_SEC="${ENGINE_MAX_USER_AUDIO_BUF_SEC:-1.0}" \
  ENGINE_ASSISTANT_MEDIA_DRAIN_SEC="${ENGINE_ASSISTANT_MEDIA_DRAIN_SEC:-1.2}" \
  ENGINE_TTS_STREAM_RESET="${ENGINE_TTS_STREAM_RESET:-0}" \
  ENGINE_INTERRUPT_GRACE_SEC="${ENGINE_INTERRUPT_GRACE_SEC:-0.40}" \
  DYSTREAM_LISTENING_CONTROLLER=0 \
  DYSTREAM_LISTENING_CONTROLLER_DEBUG=0 \
  DYSTREAM_LISTENING_FACE_CONTROL=0 \
  DYSTREAM_LISTENING_NOD=0 \
  DYSTREAM_LISTENING_BLINK=0 \
  DYSTREAM_LISTENING_FACE_CONTROL_DEBUG=0 \
  DYSTREAM_WAV2VEC_DIR="${DYSTREAM_WAV2VEC_DIR:-$ROOT_DIR/tools/hf_models/wav2vec2-base-960h}" \
  HF_HOME="${HF_HOME:-$PIPECAT_CACHE_ROOT/huggingface}" \
  MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$PIPECAT_CACHE_ROOT/modelscope}" \
  NLTK_DATA="$NLTK_CACHE_ROOT" \
  TORCH_HOME="$TORCH_CACHE_ROOT" \
  XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PIPECAT_CACHE_ROOT/xdg}" \
  "$PYTHON_BIN" -u server_mse.py \
    --port "$PORT_VALUE" \
    --sample "${SAMPLE:-1}" \
    --motion_gpu "$MOTION_LOGICAL_GPU" \
    --render_gpu "$RENDER_LOGICAL_GPU" \
    --hop_ms "${HOP_MS:-200}" \
    --feature_lag_frames "${FEATURE_LAG_FRAMES:-3}" \
    --denoising_steps "${DENOISING_STEPS:-1}" \
    --segment_frames "${SEGMENT_FRAMES:-4}" \
  > "$LOG_FILE" 2>&1 &

server_pid=$!
echo "$server_pid" > "$PID_FILE"

startup_wait_sec="${PIPECAT_STARTUP_WAIT_SEC:-120}"
startup_deadline=$((SECONDS + startup_wait_sec))
while (( SECONDS < startup_deadline )); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    exit_code=0
    wait "$server_pid" || exit_code=$?
    echo "Pipecat MSE server exited during startup pid=$server_pid code=$exit_code" >&2
    tail -n 80 "$LOG_FILE" >&2 || true
    exit 1
  fi
  if curl --fail --silent --max-time 2 \
      "http://127.0.0.1:$PORT_VALUE/health" >/dev/null 2>&1; then
    echo "ready pid=$server_pid port=$PORT_VALUE log=$LOG_FILE"
    exit 0
  fi
  sleep 1
done

echo \
  "Pipecat MSE startup timed out after ${startup_wait_sec}s; process remains pid=$server_pid" \
  >&2
tail -n 80 "$LOG_FILE" >&2 || true
exit 1
