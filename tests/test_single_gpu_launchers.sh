#!/usr/bin/env bash
# GPU-free behavioral checks. No service, bridge, or GPU process is started.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
START_SCRIPT="$ROOT_DIR/scripts/start_pipecat_mse.sh"
RUN_SCRIPT="$ROOT_DIR/scripts/run_demo.sh"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/single-gpu-launchers.XXXXXX")"
cleanup() {
  case "$TEST_ROOT" in
    "${TMPDIR:-/tmp}"/single-gpu-launchers.*) rm -rf -- "$TEST_ROOT" ;;
    *) echo "refusing unsafe test cleanup: $TEST_ROOT" >&2 ;;
  esac
}
trap cleanup EXIT
export FLASHAV2AV_ROOT="$TEST_ROOT/repo"
export ENV_FILE="$TEST_ROOT/main.env"
export PIPECAT_PYTHON="$(command -v python3)"

mkdir -p "$FLASHAV2AV_ROOT/scripts" "$FLASHAV2AV_ROOT/checkpoints" \
  "$FLASHAV2AV_ROOT/tools/pretrained_model" \
  "$FLASHAV2AV_ROOT/tools/hf_models/wav2vec2-base-960h" \
  "$FLASHAV2AV_ROOT/assets/demo_avatar"
ln -s "$START_SCRIPT" "$FLASHAV2AV_ROOT/scripts/start_pipecat_mse.sh"
touch "$FLASHAV2AV_ROOT/scripts/check_pipecat_env.py" \
  "$FLASHAV2AV_ROOT/checkpoints/last.ckpt" \
  "$FLASHAV2AV_ROOT/tools/pretrained_model/epoch=0-step=312000.ckpt" \
  "$FLASHAV2AV_ROOT/tools/hf_models/wav2vec2-base-960h/pytorch_model.bin" \
  "$FLASHAV2AV_ROOT/assets/demo_avatar/ref.png"

write_config() {
  local physical="$1"
  cat > "$ENV_FILE" <<EOF
PIPECAT_LLM_API_KEY=fixture-secret
PIPECAT_LLM_MODEL=fixture-model
PIPECAT_MSE_DIALOG_MODE=custom_cascade
PIPECAT_TTS_PROVIDER=voxcpm2
PIPECAT_ASR_DEVICE=cpu
PIPECAT_TTS_LIFECYCLE=managed
PIPECAT_TTS_BRIDGE_ENV_FILE='$TEST_ROOT/voxcpm2.env'
DYSTREAM_SINGLE_GPU=$physical
CUDA_VISIBLE_DEVICES=$physical
MOTION_GPU=0
RENDER_GPU=0
ALLOW_SHARED_DYSTREAM_GPU=1
EOF
  cat > "$TEST_ROOT/voxcpm2.env" <<EOF
CUDA_VISIBLE_DEVICES=$physical
VOXCPM2_DEVICES=0
VOXCPM2_BACKEND=official_prompt_cache
VOXCPM2_OFFICIAL_DEVICE=cuda:0
EOF
}

expect_success() {
  if ! "$@" > "$TEST_ROOT/output" 2>&1; then
    cat "$TEST_ROOT/output" >&2
    echo "FAIL: expected success: $*" >&2
    exit 1
  fi
  if grep -q 'fixture-secret' "$TEST_ROOT/output"; then
    echo "FAIL: GPU preflight printed credentials" >&2
    exit 1
  fi
}

expect_failure() {
  local expected="$1"
  shift
  if "$@" > "$TEST_ROOT/output" 2>&1; then
    echo "FAIL: expected rejection: $*" >&2
    exit 1
  fi
  grep -q "$expected" "$TEST_ROOT/output" || {
    cat "$TEST_ROOT/output" >&2
    echo "FAIL: missing diagnostic: $expected" >&2
    exit 1
  }
}

run_preflight() (
  # Supply a synthetic host inventory and codec list; retain the real launcher
  # mapping checks. Source function definitions only, never lifecycle dispatch.
  nvidia-smi() { seq 0 "$((FAKE_GPU_COUNT - 1))"; }
  ffmpeg() { printf '%s\n' ' V libx264' ' A aac'; }
  ffprobe() { return 0; }
  curl() { return 0; }
  source <(sed '/^case "$ACTION" in/,$d' "$RUN_SCRIPT")
  load_dialog_config
  preflight 0
)

bash -n "$START_SCRIPT"
bash -n "$RUN_SCRIPT"

write_config 6
expect_success bash "$START_SCRIPT" --check-gpu-config
grep -q 'gpus=6 motion_gpu=0 render_gpu=0' "$TEST_ROOT/output"
FAKE_GPU_COUNT=7 expect_success run_preflight
FAKE_GPU_COUNT=1 expect_failure 'selected GPU id 6 does not exist' run_preflight

write_config 0
FAKE_GPU_COUNT=1 expect_success run_preflight

for invalid in \
  'CUDA_VISIBLE_DEVICES=0,1' \
  'DYSTREAM_SINGLE_GPU=invalid' \
  'CUDA_DEVICE_ORDER=FASTEST_FIRST' \
  'MOTION_GPU=6' \
  'RENDER_GPU=1' \
  'ALLOW_SHARED_DYSTREAM_GPU=0' \
  'PIPECAT_TTS_PROVIDER=fish_s2pro' \
  'PIPECAT_ASR_DEVICE=cuda:0' \
  'PIPECAT_TTS_LIFECYCLE=external'; do
  write_config 6
  printf '%s\n' "$invalid" >> "$ENV_FILE"
  expect_failure 'single-GPU\|DYSTREAM_SINGLE_GPU' bash "$START_SCRIPT" --check-gpu-config
done

for invalid in \
  'CUDA_VISIBLE_DEVICES=5' \
  'CUDA_VISIBLE_DEVICES=5,6' \
  'CUDA_DEVICE_ORDER=FASTEST_FIRST' \
  'VOXCPM2_DEVICES=1' \
  'VOXCPM2_BACKEND=nano' \
  'VOXCPM2_OFFICIAL_DEVICE=cuda:1'; do
  write_config 6
  printf '%s\n' "$invalid" >> "$TEST_ROOT/voxcpm2.env"
  expect_failure 'single-GPU TTS' bash "$START_SCRIPT" --check-gpu-config
done

write_config 6
printf '%s\n' 'DYSTREAM_SINGLE_GPU=' >> "$ENV_FILE"
expect_failure 'two different GPUs' bash "$START_SCRIPT" --check-gpu-config

write_config 6
printf '%s\n' "PIPECAT_TTS_BRIDGE_ENV_FILE='$TEST_ROOT/missing.env'" >> "$ENV_FILE"
expect_failure 'TTS bridge env file' bash "$START_SCRIPT" --check-gpu-config

# Legacy two-card mode stays valid, including its explicit sharing override.
write_config 0
printf '%s\n' 'DYSTREAM_SINGLE_GPU=' 'CUDA_VISIBLE_DEVICES=0,1' \
  'RENDER_GPU=1' 'ALLOW_SHARED_DYSTREAM_GPU=0' >> "$ENV_FILE"
expect_success bash "$START_SCRIPT" --check-gpu-config
FAKE_GPU_COUNT=2 expect_success run_preflight
FAKE_GPU_COUNT=1 expect_failure 'at least 2 NVIDIA' run_preflight
printf '%s\n' 'RENDER_GPU=0' 'ALLOW_SHARED_DYSTREAM_GPU=1' >> "$ENV_FILE"
expect_success bash "$START_SCRIPT" --check-gpu-config
printf '%s\n' 'CUDA_VISIBLE_DEVICES=0, 0' >> "$ENV_FILE"
# Quote whitespace so this remains a valid shell assignment.
sed -i "s/CUDA_VISIBLE_DEVICES=0, 0/CUDA_VISIBLE_DEVICES='0, 0'/" "$ENV_FILE"
expect_failure 'two different GPUs' bash "$START_SCRIPT" --check-gpu-config

# No PID or lifecycle marker may be created by either read-only check.
[[ ! -e "$FLASHAV2AV_ROOT/logs/pipecat_mse.pid" ]]
[[ ! -e "$FLASHAV2AV_ROOT/logs/tts_stack.state" ]]
echo 'PASS: single-GPU mapping, TTS budget, physical inventory, conflicts, and legacy two-card preflight'
