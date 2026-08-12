#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"
DATA_ROOT="${FLASHAV2AV_DATA_ROOT:-$HOME/.local/share/flashav2av}"
PYTHON_BIN="${PIPECAT_PYTHON:-$DATA_ROOT/venvs/pipecat/bin/python}"
PID_FILE="logs/pipecat_webrtc.pid"
LOG_FILE="logs/pipecat_webrtc.log"

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
VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-0,1}"
MOTION_LOGICAL_GPU="${MOTION_GPU:-0}"
RENDER_LOGICAL_GPU="${RENDER_GPU:-1}"

if [[ -z "${PIPECAT_LLM_API_KEY:-${OPENAI_API_KEY:-}}" ]]; then
  echo "missing PIPECAT_LLM_API_KEY (or OPENAI_API_KEY) in $ENV_FILE" >&2
  exit 1
fi
if [[ -z "${PIPECAT_LLM_MODEL:-${OPENAI_MODEL:-}}" ]]; then
  echo "missing PIPECAT_LLM_MODEL (or OPENAI_MODEL) in $ENV_FILE" >&2
  exit 1
fi

IFS=',' read -r -a gpu_ids <<< "$VISIBLE_GPUS"
if [[ ${#gpu_ids[@]} -ne 2 ]]; then
  echo "CUDA_VISIBLE_DEVICES must expose exactly two GPUs, got: $VISIBLE_GPUS" >&2
  exit 1
fi
if [[ "${gpu_ids[0]}" == "${gpu_ids[1]}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must name two different physical GPUs" >&2
  exit 1
fi
if [[ "$MOTION_LOGICAL_GPU" == "$RENDER_LOGICAL_GPU" ]]; then
  echo "MOTION_GPU and RENDER_GPU must be different logical devices" >&2
  exit 1
fi
if [[ "$MOTION_LOGICAL_GPU,$RENDER_LOGICAL_GPU" != "0,1"       && "$MOTION_LOGICAL_GPU,$RENDER_LOGICAL_GPU" != "1,0" ]]; then
  echo "MOTION_GPU and RENDER_GPU must be logical devices 0 and 1" >&2
  exit 1
fi

mkdir -p logs "$DATA_ROOT"/cache/pipecat/{huggingface,modelscope,nltk_data,piper,xdg}
if [[ -n "${PIPECAT_TURN_URL:-}" ]]; then
  bash scripts/start_pipecat_turn.sh
fi

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Pipecat server is already running pid=$old_pid" >&2
    exit 1
  fi
fi

NLTK_DATA="${NLTK_DATA:-$DATA_ROOT/cache/pipecat/nltk_data}" \
  CUDA_VISIBLE_DEVICES="" \
  "$PYTHON_BIN" scripts/check_pipecat_env.py

nohup env \
  CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS" \
  MOTION_GPU="$MOTION_LOGICAL_GPU" \
  RENDER_GPU="$RENDER_LOGICAL_GPU" \
  ENGINE_IDLE_VISIBLE="${ENGINE_IDLE_VISIBLE:-0}" \
  ENGINE_LISTENER_AUDIO="${ENGINE_LISTENER_AUDIO:-1}" \
  ENGINE_TTS_STREAM_RESET="${ENGINE_TTS_STREAM_RESET:-0}" \
  PIPE_FRAME_STRIDE="${PIPE_FRAME_STRIDE:-2}" \
  HF_HOME="${HF_HOME:-$DATA_ROOT/cache/pipecat/huggingface}" \
  MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$DATA_ROOT/cache/pipecat/modelscope}" \
  NLTK_DATA="${NLTK_DATA:-$DATA_ROOT/cache/pipecat/nltk_data}" \
  XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DATA_ROOT/cache/pipecat/xdg}" \
  "$PYTHON_BIN" -u pipecat_server.py \
    --host 0.0.0.0 \
    --port "${PIPECAT_PORT:-7860}" \
  > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "started pid=$(cat "$PID_FILE") log=$LOG_FILE"
