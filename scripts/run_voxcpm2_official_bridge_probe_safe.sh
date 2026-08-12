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
PROBE="$REPO/scripts/probe_voxcpm2_turn_protocol.py"
OUTPUT_DIR="${1:?usage: $0 OUTPUT_DIR}"
CANDIDATE_LOG="$OUTPUT_DIR.bridge.log"
CANDIDATE_PID=""
ORIGINAL_PID=""

if [[ -e "$OUTPUT_DIR" || -e "$CANDIDATE_LOG" ]]; then
    echo "Refusing to overwrite candidate output: $OUTPUT_DIR" >&2
    exit 2
fi
for required in \
    "$OFFICIAL/src/voxcpm/model/voxcpm2.py" \
    "$MODEL/config.json" \
    "$ENV_FILE" \
    "$PROBE" \
    "$REPO/voice_service/official_voxcpm2_backend.py"; do
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

bridge_healthy() {
    "$PYTHON" -c 'import json; from websockets.sync.client import connect; ws=connect("ws://127.0.0.1:8770", open_timeout=2, close_timeout=1); ws.send(json.dumps({"type":"health"})); event=json.loads(ws.recv(timeout=2)); ws.close(); assert event.get("type")=="health" and event.get("status")=="ok" and int(event.get("sample_rate",0))>0' >/dev/null 2>&1
}

wait_stopped() {
    local process_id="$1"
    local attempts="$2"
    for _ in $(seq 1 "$attempts"); do
        if ! kill -0 "$process_id" 2>/dev/null; then
            return 0
        fi
        sleep 1
    done
    return 1
}

restore_bridge() {
    local original_status=$?
    local restored_pid
    local ready=0
    trap - EXIT HUP INT TERM

    if [[ "$CANDIDATE_PID" =~ ^[0-9]+$ ]] && kill -0 "$CANDIDATE_PID" 2>/dev/null; then
        kill -TERM "$CANDIDATE_PID" 2>/dev/null || true
        wait_stopped "$CANDIDATE_PID" 120 || true
    fi

    if bridge_healthy; then
        exit "$original_status"
    fi

    echo "Restoring configured VoxCPM2 bridge..."
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
    export PYTHONPATH="$REPO"
    cd "$REPO"
    nohup "$PYTHON" -m voice_service.voxcpm2_server >> "$BRIDGE_LOG" 2>&1 &
    restored_pid=$!
    printf '%s\n' "$restored_pid" > "$PID_FILE"
    for _ in $(seq 1 300); do
        if bridge_healthy; then
            ready=1
            break
        fi
        if ! kill -0 "$restored_pid" 2>/dev/null; then
            break
        fi
        sleep 2
    done
    if [[ "$ready" -ne 1 ]]; then
        echo "Configured bridge restoration failed; inspect $BRIDGE_LOG" >&2
        exit 97
    fi
    echo "Configured bridge restored: PID $restored_pid"
    exit "$original_status"
}
trap restore_bridge EXIT HUP INT TERM

echo "Stopping current bridge PID $ORIGINAL_PID for isolated official-backend probe..."
kill -TERM "$ORIGINAL_PID"
if ! wait_stopped "$ORIGINAL_PID" 120; then
    echo "Current bridge did not stop cleanly; candidate will not run" >&2
    exit 3
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH="$REPO"
export VOXCPM2_BACKEND=official_prompt_cache
export VOXCPM2_OFFICIAL_SOURCE="$OFFICIAL/src"
export VOXCPM2_OFFICIAL_DEVICE=cuda
export VOXCPM2_OFFICIAL_OPTIMIZE=1
export VOXCPM2_SEED=42

cd "$REPO"
nohup "$PYTHON" -m voice_service.voxcpm2_server > "$CANDIDATE_LOG" 2>&1 &
CANDIDATE_PID=$!
echo "Official candidate loading as PID $CANDIDATE_PID..."
ready=0
for _ in $(seq 1 300); do
    if bridge_healthy; then
        ready=1
        break
    fi
    if ! kill -0 "$CANDIDATE_PID" 2>/dev/null; then
        break
    fi
    sleep 2
done
if [[ "$ready" -ne 1 ]]; then
    echo "Official candidate failed to become healthy; inspect $CANDIDATE_LOG" >&2
    exit 4
fi

"$PYTHON" "$PROBE" \
    --url ws://127.0.0.1:8770 \
    --timeout 120 \
    --output-dir "$OUTPUT_DIR"

echo "Official bridge protocol probe completed: $OUTPUT_DIR"
