#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs
PID_FILE="logs/server_realtime_6008.pid"
LOG_FILE="logs/server_realtime_6008.log"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "stopping old pid=$old_pid"
    kill "$old_pid" || true
    sleep 2
    if kill -0 "$old_pid" 2>/dev/null; then
      kill -9 "$old_pid" || true
    fi
  fi
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "missing env file: $ENV_FILE" >&2
  echo "copy .env.example to .env and fill SEEDUPLEX_APP_ID / SEEDUPLEX_ACCESS_KEY" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a

nohup env \
  TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
  PIPE_FRAME_STRIDE="${PIPE_FRAME_STRIDE:-2}" \
  DYSTREAM_AUDIO_Q="${DYSTREAM_AUDIO_Q:-8}" \
  DYSTREAM_MOTION_Q="${DYSTREAM_MOTION_Q:-4}" \
  DYSTREAM_FRAME_Q="${DYSTREAM_FRAME_Q:-20}" \
  ENGINE_IDLE_WARMUP_SEGMENTS="${ENGINE_IDLE_WARMUP_SEGMENTS:-16}" \
  ENGINE_IDLE_CONTINUOUS="${ENGINE_IDLE_CONTINUOUS:-1}" \
  ENGINE_IDLE_VISIBLE="${ENGINE_IDLE_VISIBLE:-1}" \
  ENGINE_IDLE_ONLY_WITH_CLIENT="${ENGINE_IDLE_ONLY_WITH_CLIENT:-1}" \
  ENGINE_IDLE_RESET_SEGMENTS="${ENGINE_IDLE_RESET_SEGMENTS:-60}" \
  ENGINE_LISTENER_AUDIO="${ENGINE_LISTENER_AUDIO:-0}" \
  ENGINE_USER_SPEAKING_RMS="${ENGINE_USER_SPEAKING_RMS:-0.006}" \
  ENGINE_USER_SPEAKING_HOLD_SEC="${ENGINE_USER_SPEAKING_HOLD_SEC:-0.6}" \
  ENGINE_MAX_USER_AUDIO_BUF_SEC="${ENGINE_MAX_USER_AUDIO_BUF_SEC:-1.0}" \
  ENGINE_ASSISTANT_MEDIA_DRAIN_SEC="${ENGINE_ASSISTANT_MEDIA_DRAIN_SEC:-1.2}" \
  ENGINE_TTS_STREAM_RESET="${ENGINE_TTS_STREAM_RESET:-0}" \
  ENGINE_INTERRUPT_BRIDGE_SEC="${ENGINE_INTERRUPT_BRIDGE_SEC:-0.20}" \
  DYSTREAM_LISTENING_CONTROLLER="${DYSTREAM_LISTENING_CONTROLLER:-0}" \
  DYSTREAM_LISTENING_CONTROLLER_DEBUG="${DYSTREAM_LISTENING_CONTROLLER_DEBUG:-0}" \
  DYSTREAM_LISTENING_FACE_CONTROL="${DYSTREAM_LISTENING_FACE_CONTROL:-0}" \
  DYSTREAM_LISTENING_NOD="${DYSTREAM_LISTENING_NOD:-0}" \
  DYSTREAM_LISTENING_BLINK="${DYSTREAM_LISTENING_BLINK:-0}" \
  DYSTREAM_LISTENING_FACE_CONTROL_DEBUG="${DYSTREAM_LISTENING_FACE_CONTROL_DEBUG:-0}" \
  "${PIPECAT_PYTHON:-python3}" -u server_mse.py \
    --port "${PORT:-6008}" \
    --sample "${SAMPLE:-1}" \
    --motion_gpu "${MOTION_GPU:-0}" \
    --render_gpu "${RENDER_GPU:-1}" \
    --hop_ms "${HOP_MS:-200}" \
    --feature_lag_frames "${FEATURE_LAG_FRAMES:-3}" \
    --denoising_steps "${DENOISING_STEPS:-1}" \
    --segment_frames "${SEGMENT_FRAMES:-4}" \
  > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "started pid=$(cat "$PID_FILE") log=$LOG_FILE"
