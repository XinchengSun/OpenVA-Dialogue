#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
profile="${1:-low_ttfa_gapless}"

case "$profile" in
  balanced)
    config="$repo_root/configs/fish_s2pro_4090_2gpu_balanced.yaml"
    ;;
  low_ttfa_gapless)
    config="$repo_root/configs/fish_s2pro_4090_2gpu_low_ttfa_gapless.yaml"
    ;;
  *)
    echo "usage: $0 [balanced|low_ttfa_gapless]" >&2
    exit 2
    ;;
esac

fish_root="${FISH_ROOT:-${FLASHAV2AV_DATA_ROOT:-$HOME/.local/share/flashav2av}/runtime/fish-s2-pro}"
sglang_src="${SGLANG_OMNI_SRC:-$fish_root/src/sglang-omni-ghproxy}"
venv="${FISH_VENV:-$fish_root/venv}"
model="${FISH_MODEL:-$fish_root/models/fishaudio-s2-pro}"
reference_dir="${FISH_REFERENCE_DIR:-$fish_root/references}"
cuda_devices="${FISH_CUDA_VISIBLE_DEVICES:-5,6}"
host="${FISH_HTTP_HOST:-127.0.0.1}"
port="${FISH_HTTP_PORT:-8001}"
patch_file="$repo_root/patches/sglang_omni_fish_cross_gpu_host_staging.patch"
expected_sglang_commit="2e607bc005c1a33801d60ef8f48f2a29d8b18aa5"

for required in \
  "$config" \
  "$patch_file" \
  "$venv/bin/sgl-omni" \
  "$model/config.json" \
  "$model/codec.pth" \
  "$model/model.safetensors.index.json" \
  "$model/model-00001-of-00002.safetensors" \
  "$model/model-00002-of-00002.safetensors" \
  "$reference_dir"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required path: $required" >&2
    exit 1
  fi
done

actual_sglang_commit="$(git -C "$sglang_src" rev-parse HEAD)"
if [[ "$actual_sglang_commit" != "$expected_sglang_commit" ]]; then
  echo "unsupported SGLang-Omni commit: $actual_sglang_commit" >&2
  echo "expected: $expected_sglang_commit" >&2
  exit 1
fi

# SGLang-Omni 2e607bc routes every GPU-to-GPU stream through CUDA IPC.
# This 8x4090 host exposes no P2P pairs, so apply the recorded SHM fallback once.
if git -C "$sglang_src" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
  : # already applied
elif git -C "$sglang_src" apply --check "$patch_file" >/dev/null 2>&1; then
  git -C "$sglang_src" apply "$patch_file"
else
  echo "cross-GPU host-staging patch is neither cleanly applied nor applicable" >&2
  echo "inspect: $sglang_src" >&2
  exit 1
fi

mkdir -p "$fish_root/cache/huggingface" "$fish_root/cache/triton"

echo "Fish S2 Pro profile: $profile"
echo "physical GPUs: $cuda_devices (logical tts_engine=0, vocoder=1)"
echo "HTTP endpoint: http://$host:$port"

exec env \
  PATH="$venv/bin:$PATH" \
  PYTHONPATH="$sglang_src${PYTHONPATH:+:$PYTHONPATH}" \
  CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}" \
  CUDA_VISIBLE_DEVICES="$cuda_devices" \
  HF_HOME="$fish_root/cache/huggingface" \
  TRITON_CACHE_DIR="$fish_root/cache/triton" \
  "$venv/bin/sgl-omni" serve \
    --model-path "$model" \
    --config "$config" \
    --model-name fishaudio/s2-pro \
    --allowed-local-media-path "$reference_dir" \
    --host "$host" \
    --port "$port"
