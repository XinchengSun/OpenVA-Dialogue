#!/usr/bin/env bash
# All dependencies are tiny local fixtures. No package/model download occurs.
set -euo pipefail
SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/single-gpu-setup.XXXXXX")"
cleanup() {
  case "$TEST_ROOT" in
    "${TMPDIR:-/tmp}"/single-gpu-setup.*) rm -rf -- "$TEST_ROOT" ;;
  esac
}
trap cleanup EXIT
REPO="$TEST_ROOT/repo"
export FLASHAV2AV_DATA_ROOT="$TEST_ROOT/runtime"
export PIPECAT_VENV="$FLASHAV2AV_DATA_ROOT/venvs/pipecat"
unset ENV_FILE FLASHAV2AV_ROOT
OFFICIAL="$TEST_ROOT/official/src"
MODEL="$FLASHAV2AV_DATA_ROOT/models/VoxCPM2"
mkdir -p "$REPO/scripts" "$REPO/checkpoints" "$REPO/tools/pretrained_model" \
  "$REPO/tools/hf_models/wav2vec2-base-960h" "$FLASHAV2AV_DATA_ROOT/config" \
  "$PIPECAT_VENV/bin" "$OFFICIAL/voxcpm/model" "$MODEL" "$TEST_ROOT/bin"
cp "$SOURCE_ROOT/scripts/flashav2av" "$SOURCE_ROOT/scripts/setup_single_gpu.sh" \
  "$SOURCE_ROOT/scripts/start_pipecat_mse.sh" "$REPO/scripts/"
ln -s "$(command -v python3)" "$PIPECAT_VENV/bin/python"
for command in curl ffmpeg ffprobe nvidia-smi; do
  printf '#!/usr/bin/env bash\nexit 99\n' > "$TEST_ROOT/bin/$command"
  chmod +x "$TEST_ROOT/bin/$command"
done
export PATH="$TEST_ROOT/bin:$PATH"
touch "$REPO/checkpoints/last.ckpt" \
  "$REPO/tools/pretrained_model/epoch=0-step=312000.ckpt" \
  "$REPO/tools/hf_models/wav2vec2-base-960h/pytorch_model.bin" "$TEST_ROOT/avatar.png"
cat > "$OFFICIAL/voxcpm/__init__.py" <<'PY'
class VoxCPM:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise AssertionError("check-only must not construct a model")
PY
touch "$OFFICIAL/voxcpm/model/__init__.py"
cat > "$OFFICIAL/voxcpm/model/voxcpm2.py" <<'PY'
class VoxCPM2Model:
    def build_prompt_cache(self): pass
    def merge_prompt_cache(self): pass
    def generate_with_prompt_cache_streaming(self): pass
PY
for module in torch torchaudio websockets soundfile numpy; do
  touch "$OFFICIAL/$module.py"
done
for file in config.json audiovae.pth tokenizer.json tokenizer_config.json model.safetensors; do
  printf '{}\n' > "$MODEL/$file"
done
cat > "$FLASHAV2AV_DATA_ROOT/config/custom_cascade.env" <<EOF
PIPECAT_LLM_API_KEY=fixture-key
PIPECAT_MSE_DIALOG_MODE=custom_cascade
PIPECAT_TTS_PROVIDER=voxcpm2
PIPECAT_ASR_DEVICE=cpu
PIPECAT_TTS_LIFECYCLE=managed
PIPECAT_TTS_BRIDGE_ENV_FILE='$FLASHAV2AV_DATA_ROOT/config/voxcpm2.env'
DYSTREAM_SINGLE_GPU=6
CUDA_VISIBLE_DEVICES=6
MOTION_GPU=0
RENDER_GPU=0
ALLOW_SHARED_DYSTREAM_GPU=1
DYSTREAM_REF_IMAGE='$TEST_ROOT/avatar.png'
EOF
cat > "$FLASHAV2AV_DATA_ROOT/config/voxcpm2.env" <<EOF
CUDA_VISIBLE_DEVICES=6
VOXCPM2_DEVICES=0
VOXCPM2_BACKEND=official_prompt_cache
VOXCPM2_OFFICIAL_DEVICE=cuda:0
VOXCPM2_PYTHON='$PIPECAT_VENV/bin/python'
VOXCPM2_OFFICIAL_SOURCE='$OFFICIAL'
VOXCPM2_MODEL_PATH='$MODEL'
EOF

for command in setup-single-gpu setup; do
  bash "$REPO/scripts/flashav2av" "$command" --check-only > "$TEST_ROOT/output" 2>&1 || {
    cat "$TEST_ROOT/output" >&2
    exit 1
  }
  grep -q 'OFFICIAL_VOXCPM2_PREREQUISITES_OK' "$TEST_ROOT/output"
  grep -q 'SETUP_PROFILE=single_gpu' "$TEST_ROOT/output"
  ! grep -q 'fish-s2-pro\|fixture-key' "$TEST_ROOT/output"
done
[[ ! -d "$REPO/logs" && ! -d "$FLASHAV2AV_DATA_ROOT/cache" ]]
[[ -z "$(find "$OFFICIAL" -name __pycache__ -print -quit)" ]]

if bash "$REPO/scripts/flashav2av" setup-single-gpu > "$TEST_ROOT/output" 2>&1; then
  echo 'FAIL: unimplemented fresh-host setup must not report success' >&2
  exit 1
fi
grep -q 'requires --check-only' "$TEST_ROOT/output"
rm "$MODEL/audiovae.pth"
if bash "$REPO/scripts/flashav2av" setup-single-gpu --check-only > "$TEST_ROOT/output" 2>&1; then
  echo 'FAIL: incomplete official weights must be rejected' >&2
  exit 1
fi
grep -q 'VOXCPM2_MODEL_MISSING_OR_INVALID audiovae.pth' "$TEST_ROOT/output"
! grep -q 'fish-s2-pro' "$TEST_ROOT/output"

printf '%s\n' 'DYSTREAM_SINGLE_GPU=' >> "$FLASHAV2AV_DATA_ROOT/config/custom_cascade.env"
if bash "$REPO/scripts/flashav2av" setup --check-only > "$TEST_ROOT/output" 2>&1; then
  echo 'FAIL: default multi-GPU profile must still require Fish assets' >&2
  exit 1
fi
grep -q 'fish-s2-pro' "$TEST_ROOT/output"
if bash "$REPO/scripts/flashav2av" setup-single-gpu --check-only > "$TEST_ROOT/output" 2>&1; then
  echo 'FAIL: explicit single-GPU setup must require a single-GPU configuration' >&2
  exit 1
fi
grep -q 'SINGLE_GPU_CONFIG_MISSING' "$TEST_ROOT/output"
echo 'PASS: single-GPU setup is read-only, checks official prerequisites, skips Fish, and preserves the multi-GPU profile'
