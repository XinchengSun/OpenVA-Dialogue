#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${FLASHAV2AV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="${FLASHAV2AV_REPO:-$ROOT}"
RUNTIME="${CUSTOM_CASCADE_RUNTIME:-${FLASHAV2AV_DATA_ROOT:-$HOME/.local/share/flashav2av}/runtime}"
OFFICIAL="$RUNTIME/candidates/prompt_cache_20260803/VoxCPM"
PYTHON="$RUNTIME/venvs/voxcpm2-nano-2.0.3/bin/python"
MODEL="$RUNTIME/models/VoxCPM2"
ENV_FILE="$RUNTIME/config/voxcpm2.env"
PID_FILE="$REPO/logs/voxcpm2_bridge.pid"
BRIDGE_LOG="$REPO/logs/voxcpm2_bridge.log"
BENCHMARK="$REPO/scripts/bench_voxcpm2_prompt_cache.py"
OUTPUT_DIR="${1:-$RUNTIME/candidates/prompt_cache_20260803/results/official_prompt_cache_run1}"
RUN_LOG="$RUNTIME/candidates/prompt_cache_20260803/prompt_cache_ab_safe.log"

exec > >(tee -a "$RUN_LOG") 2>&1

if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite existing output: $OUTPUT_DIR" >&2
    exit 2
fi
for required in "$OFFICIAL/src/voxcpm/model/voxcpm2.py" "$MODEL/config.json" "$ENV_FILE" "$BENCHMARK"; do
    [[ -f "$required" ]] || { echo "Missing required file: $required" >&2; exit 2; }
done

ORIGINAL_PID="$(<"$PID_FILE")"
[[ "$ORIGINAL_PID" =~ ^[0-9]+$ ]] || { echo "Invalid bridge PID: $ORIGINAL_PID" >&2; exit 2; }
kill -0 "$ORIGINAL_PID"
ORIGINAL_COMMAND="$(tr '\0' ' ' < "/proc/$ORIGINAL_PID/cmdline")"
[[ "$ORIGINAL_COMMAND" == *"voice_service.voxcpm2_server"* ]] || {
    echo "PID $ORIGINAL_PID is not the VoxCPM2 bridge: $ORIGINAL_COMMAND" >&2
    exit 2
}

restore_bridge() {
    local original_status=$?
    local ready=0
    trap - EXIT HUP INT TERM
    if [[ -f "$PID_FILE" ]]; then
        local current_pid
        current_pid="$(<"$PID_FILE")"
        if [[ "$current_pid" =~ ^[0-9]+$ ]] && kill -0 "$current_pid" 2>/dev/null; then
            exit "$original_status"
        fi
    fi

    echo "Restoring Nano VoxCPM2 bridge..."
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
    export CUDA_VISIBLE_DEVICES=6
    export PYTHONPATH="$REPO"
    cd "$REPO"
    nohup "$PYTHON" -m voice_service.voxcpm2_server >> "$BRIDGE_LOG" 2>&1 &
    local restored_pid=$!
    printf '%s\n' "$restored_pid" > "$PID_FILE"

    for _ in $(seq 1 300); do
        if "$PYTHON" -c 'import json; from websockets.sync.client import connect; ws=connect("ws://127.0.0.1:8770", open_timeout=2, close_timeout=1); ws.send(json.dumps({"type":"health"})); event=json.loads(ws.recv(timeout=2)); ws.close(); assert event.get("type")=="health" and event.get("status")=="ok" and int(event.get("sample_rate",0))>0' >/dev/null 2>&1; then
            ready=1
            break
        fi
        if ! kill -0 "$restored_pid" 2>/dev/null; then
            break
        fi
        sleep 2
    done
    if [[ "$ready" -ne 1 ]]; then
        echo "Nano bridge restoration failed; inspect $BRIDGE_LOG" >&2
        exit 97
    fi
    echo "Nano bridge restored: PID $restored_pid"
    exit "$original_status"
}
trap restore_bridge EXIT HUP INT TERM

echo "Stopping Nano bridge PID $ORIGINAL_PID for exclusive GPU6 A/B..."
kill -TERM "$ORIGINAL_PID"
for _ in $(seq 1 120); do
    if ! kill -0 "$ORIGINAL_PID" 2>/dev/null; then
        break
    fi
    sleep 1
done
if kill -0 "$ORIGINAL_PID" 2>/dev/null; then
    echo "Nano bridge did not stop cleanly; candidate will not run" >&2
    exit 3
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH="$OFFICIAL/src:$REPO"

echo "Running official VoxCPM2 prompt-cache candidate on physical GPU6..."
"$PYTHON" "$BENCHMARK" official \
    --source "$OFFICIAL/src" \
    --model "$MODEL" \
    --reference "$VOXCPM2_PROMPT_WAV" \
    --rounds 2 \
    --warmups 1 \
    --device cuda:0 \
    --seed 42 \
    --inference-timesteps 10 \
    --cfg-value 2.0 \
    --streaming-prefix-len 4 \
    --optimize \
    --output-dir "$OUTPUT_DIR"

echo "Official candidate finished: $OUTPUT_DIR"
