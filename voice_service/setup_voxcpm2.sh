#!/usr/bin/env bash
set -Eeuo pipefail

# This installer is deliberately isolated from the DyStream/Pipecat runtime.
# It writes only below DATA_ROOT and never activates or mutates the caller's venv.

MODE="full"
DATA_ROOT="${VOXCPM2_DATA_ROOT:-${FLASHAV2AV_DATA_ROOT:-$HOME/.local/share/flashav2av}}"
PYTHON_BIN="${VOXCPM2_PYTHON_BOOTSTRAP:-python3.11}"
MIN_FREE_GB="${VOXCPM2_MIN_FREE_GB:-}"

PYPI_INDEX_URL="${VOXCPM2_PYPI_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
PYTORCH_MIRROR_ROOT="${VOXCPM2_PYTORCH_MIRROR_ROOT:-https://mirrors.aliyun.com/pytorch-wheels}"
MODEL_ID="${VOXCPM2_MODELSCOPE_ID:-OpenBMB/VoxCPM2}"

TORCH_VERSION="2.5.1"
TORCHCODEC_VERSION="0.1.1"
FLASH_ATTN_VERSION="2.8.3.post1"
NANO_VERSION="2.0.3"
WEBSOCKETS_VERSION="14.1"
MODELSCOPE_VERSION="1.39.0"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements-voxcpm2.txt"

usage() {
  cat <<'EOF'
Usage: setup_voxcpm2.sh [--check-only | --download-only] [options]

Modes:
  (no mode)          Check host, create an isolated venv, install the pinned
                     runtime, download VoxCPM2, and verify both.
  --check-only       Read-only host/environment/model inspection. No download,
                     directory creation, or package installation.
  --download-only    Download/continue the public ModelScope model only. CUDA,
                     GPU and NVCC are still reported but are not required.

Options:
  --data-root PATH   Writable data root (default: ~/.local/share/flashav2av).
  --python PATH      Python 3.11 bootstrap executable (default: python3.11).
  --min-free-gb N    Minimum free space. Defaults: full/check=40, download=20.
  -h, --help         Show this help.

All writable paths are derived from --data-root:
  venv:       DATA_ROOT/venvs/voxcpm2-nano-2.0.3
  downloader: DATA_ROOT/venvs/modelscope-downloader
  model:      DATA_ROOT/models/VoxCPM2
  caches:     DATA_ROOT/cache/{pip,modelscope}
EOF
}

info() { printf '[voxcpm2-setup] %s\n' "$*"; }
warn() { printf '[voxcpm2-setup] WARNING: %s\n' "$*" >&2; }
die() { printf '[voxcpm2-setup] ERROR: %s\n' "$*" >&2; exit 1; }

while (($#)); do
  case "$1" in
    --check-only)
      [[ "$MODE" == "full" ]] || die "choose only one mode"
      MODE="check"
      shift
      ;;
    --download-only)
      [[ "$MODE" == "full" ]] || die "choose only one mode"
      MODE="download"
      shift
      ;;
    --data-root)
      (($# >= 2)) || die "--data-root requires a value"
      DATA_ROOT="$2"
      shift 2
      ;;
    --python)
      (($# >= 2)) || die "--python requires a value"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --min-free-gb)
      (($# >= 2)) || die "--min-free-gb requires a value"
      MIN_FREE_GB="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ "$(uname -s)" == "Linux" ]] || die "this installer only supports Linux"
command -v readlink >/dev/null 2>&1 || die "readlink is required"
[[ "$PYPI_INDEX_URL" != *'@'* && "$PYTORCH_MIRROR_ROOT" != *'@'* ]] || \
  die "credential-bearing mirror URLs are not allowed because installers may log them"

DATA_ROOT="$(readlink -m -- "$DATA_ROOT")"
[[ "$DATA_ROOT" == /* && "$DATA_ROOT" != "/" ]] || die "data root must be an absolute non-root path"

ENV_DIR="$DATA_ROOT/venvs/voxcpm2-nano-$NANO_VERSION"
DOWNLOADER_ENV="$DATA_ROOT/venvs/modelscope-downloader"
MODEL_DIR="$DATA_ROOT/models/VoxCPM2"
PIP_CACHE_DIR="$DATA_ROOT/cache/pip"
MODELSCOPE_CACHE="$DATA_ROOT/cache/modelscope"

path_is_below_root() {
  local candidate
  candidate="$(readlink -m -- "$1")"
  [[ "$candidate" == "$DATA_ROOT"/* ]]
}

for writable_path in "$ENV_DIR" "$DOWNLOADER_ENV" "$MODEL_DIR" "$PIP_CACHE_DIR" "$MODELSCOPE_CACHE"; do
  path_is_below_root "$writable_path" || die "refusing path outside data root: $writable_path"
done

[[ "$ENV_DIR" != "$DATA_ROOT/pipecat-env" ]] || die "refusing to reuse the Pipecat environment"
if [[ -n "${PIPECAT_PYTHON:-}" ]]; then
  pipecat_env="$(readlink -m -- "$(dirname -- "$(dirname -- "$PIPECAT_PYTHON")")")"
  [[ "$ENV_DIR" != "$pipecat_env" ]] || die "VoxCPM2 and Pipecat environments must be separate"
fi

if [[ -z "$MIN_FREE_GB" ]]; then
  if [[ "$MODE" == "download" ]]; then
    MIN_FREE_GB=20
  else
    MIN_FREE_GB=40
  fi
fi
[[ "$MIN_FREE_GB" =~ ^[0-9]+$ ]] || die "--min-free-gb must be a non-negative integer"

existing_parent() {
  local candidate="$1"
  while [[ ! -e "$candidate" && "$candidate" != "/" ]]; do
    candidate="$(dirname -- "$candidate")"
  done
  printf '%s\n' "$candidate"
}

check_disk() {
  local probe available_kb required_kb available_gb
  probe="$(existing_parent "$DATA_ROOT")"
  available_kb="$(df -Pk "$probe" | tail -n 1 | awk '{print $4}')"
  [[ "$available_kb" =~ ^[0-9]+$ ]] || die "could not determine free space for $probe"
  required_kb=$((MIN_FREE_GB * 1024 * 1024))
  available_gb=$((available_kb / 1024 / 1024))
  info "disk: ${available_gb} GiB free; minimum ${MIN_FREE_GB} GiB"
  ((available_kb >= required_kb)) || die "insufficient free disk space"
}

check_python() {
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python 3.11 not found: $PYTHON_BIN"
  PYTHON_BIN="$(command -v "$PYTHON_BIN")"
  python_version="$($PYTHON_BIN -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
  [[ "$python_version" == 3.11.* ]] || die "Python 3.11 is required; found $python_version"
  "$PYTHON_BIN" -m venv --help >/dev/null 2>&1 || die "Python venv support is unavailable"
  info "python: $python_version"
}

NVCC_VERSION=""
CUDA_TAG=""

detect_accelerator() {
  local nvcc_bin="" driver_cuda="" gpu_lines="" gpu_count=0

  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_lines="$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true)"
    gpu_count="$(printf '%s\n' "$gpu_lines" | sed '/^[[:space:]]*$/d' | wc -l | tr -d '[:space:]')"
    driver_cuda="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
    info "gpu: ${gpu_count} visible; driver CUDA API ${driver_cuda:-unknown}"
    [[ -z "$gpu_lines" ]] || printf '%s\n' "$gpu_lines" | sed 's/^/[voxcpm2-setup]   /'
  else
    warn "nvidia-smi not found"
  fi

  if command -v nvcc >/dev/null 2>&1; then
    nvcc_bin="$(command -v nvcc)"
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    nvcc_bin=/usr/local/cuda/bin/nvcc
  fi

  if [[ -n "$nvcc_bin" ]]; then
    NVCC_VERSION="$($nvcc_bin --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
    info "nvcc: ${NVCC_VERSION:-unparseable} ($nvcc_bin)"
  else
    warn "NVCC not found"
  fi

  if [[ "$MODE" == "download" ]]; then
    return 0
  fi
  command -v nvidia-smi >/dev/null 2>&1 || die "NVIDIA GPU tooling is required for the runtime"
  ((gpu_count > 0)) || die "no visible NVIDIA GPU"
  [[ -n "$nvcc_bin" && -n "$NVCC_VERSION" ]] || die "a parseable CUDA toolkit/NVCC is required for FlashAttention"
  [[ -n "$driver_cuda" ]] || die "could not parse the NVIDIA driver's CUDA compatibility level"
  [[ "$(printf '%s\n' "$NVCC_VERSION" "$driver_cuda" | sort -V | head -n 1)" == "$NVCC_VERSION" ]] || \
    die "driver CUDA API $driver_cuda is older than NVCC $NVCC_VERSION"

  # The published PyTorch 2.5.1 profiles are cu121 and cu124.  CUDA minor
  # versions are forward-compatible within CUDA 12, so a newer AutoDL toolkit
  # (for example 12.6 or 12.8) must not make an otherwise valid 4090 host fail
  # preflight.  Pick the newest profile not newer than the local toolkit.
  [[ "$NVCC_VERSION" == 12.* ]] || \
    die "only CUDA 12.x toolkits are supported by the pinned cu121/cu124 runtime; found $NVCC_VERSION"
  if [[ "$(printf '%s\n' "12.4" "$NVCC_VERSION" | sort -V | head -n 1)" == "12.4" ]]; then
    CUDA_TAG="cu124"
  elif [[ "$(printf '%s\n' "12.1" "$NVCC_VERSION" | sort -V | head -n 1)" == "12.1" ]]; then
    CUDA_TAG="cu121"
  else
    die "NVCC $NVCC_VERSION is older than the lowest pinned wheel profile (CUDA 12.1)"
  fi
  info "selected PyTorch wheel profile: $CUDA_TAG (local NVCC $NVCC_VERSION)"
}

model_is_complete() {
  local index_file=""
  [[ -s "$MODEL_DIR/config.json" ]] || return 1
  [[ -s "$MODEL_DIR/audiovae.pth" ]] || return 1
  [[ -s "$MODEL_DIR/tokenizer.json" ]] || return 1
  [[ -s "$MODEL_DIR/tokenizer_config.json" ]] || return 1
  index_file="$(find "$MODEL_DIR" -maxdepth 1 -type f -name '*.safetensors.index.json' -print -quit 2>/dev/null)"
  if [[ -n "$index_file" ]]; then
    "$PYTHON_BIN" - "$MODEL_DIR" "$index_file" <<'PY'
import json
from pathlib import Path
import sys

model_dir = Path(sys.argv[1])
index_file = Path(sys.argv[2])
try:
    payload = json.loads(index_file.read_text(encoding="utf-8"))
    filenames = set(payload["weight_map"].values())
except (OSError, KeyError, TypeError, ValueError):
    raise SystemExit(1)
if not filenames:
    raise SystemExit(1)
for filename in filenames:
    candidate = model_dir / filename
    if not candidate.is_file() or candidate.stat().st_size <= 0:
        raise SystemExit(1)
PY
    return $?
  fi
  find "$MODEL_DIR" -maxdepth 1 -type f -name '*.safetensors' -size +0c -print -quit 2>/dev/null | grep -q .
}

venv_is_valid() {
  local target="$1"
  [[ -f "$target/pyvenv.cfg" && -x "$target/bin/python" && -x "$target/bin/pip" ]]
}

host_preflight() {
  info "mode: $MODE"
  info "data root: $DATA_ROOT"
  check_python
  check_disk
  detect_accelerator

  if model_is_complete; then
    info "model: complete; download will be skipped"
  elif [[ -d "$MODEL_DIR" ]]; then
    info "model: partial; ModelScope will continue the same local directory"
  else
    info "model: not downloaded"
  fi

  if venv_is_valid "$ENV_DIR"; then
    info "runtime venv: present"
  elif [[ -e "$ENV_DIR" ]]; then
    warn "runtime path exists but is not a Python venv: $ENV_DIR"
  else
    info "runtime venv: absent"
  fi
}

pip_cmd() {
  local python_exe="$1"
  shift
  env \
    -u PYTHONHOME \
    -u PYTHONPATH \
    -u PIP_INDEX_URL \
    -u PIP_EXTRA_INDEX_URL \
    -u PIP_FIND_LINKS \
    -u PIP_NO_INDEX \
    -u PIP_CONSTRAINT \
    -u PIP_REQUIRE_HASHES \
    -u PIP_TARGET \
    -u PIP_PREFIX \
    -u PIP_USER \
    -u PIP_TRUSTED_HOST \
    PIP_CONFIG_FILE=/dev/null \
    PIP_CACHE_DIR="$PIP_CACHE_DIR" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_PROGRESS_BAR=off \
    "$python_exe" -m pip --retries 10 --timeout 60 "$@"
}

ensure_venv() {
  local target="$1"
  if venv_is_valid "$target"; then
    return 0
  fi
  [[ ! -e "$target" ]] || die "refusing to overwrite non-venv path: $target"
  env -u PYTHONHOME -u PYTHONPATH "$PYTHON_BIN" -m venv "$target"
  pip_cmd "$target/bin/python" install \
    --index-url "$PYPI_INDEX_URL" \
    'pip==24.3.1' 'setuptools==75.6.0' 'wheel==0.45.1'
}

model_download() {
  if model_is_complete; then
    info "model already complete: $MODEL_DIR"
    return 0
  fi

  install -d -m 755 "$DATA_ROOT/venvs" "$DATA_ROOT/models" "$DATA_ROOT/cache"
  install -d -m 755 "$PIP_CACHE_DIR" "$MODELSCOPE_CACHE"
  ensure_venv "$DOWNLOADER_ENV"
  pip_cmd "$DOWNLOADER_ENV/bin/python" install \
    --index-url "$PYPI_INDEX_URL" \
    "modelscope==$MODELSCOPE_VERSION"

  install -d -m 755 "$MODEL_DIR"
  info "downloading public model from ModelScope; existing chunks/cache are reused"
  env \
    -u PYTHONHOME \
    -u PYTHONPATH \
    -u MODELSCOPE_API_TOKEN \
    -u MODELSCOPE_SDK_TOKEN \
    -u HF_TOKEN \
    -u HUGGING_FACE_HUB_TOKEN \
    MODELSCOPE_CACHE="$MODELSCOPE_CACHE" \
    "$DOWNLOADER_ENV/bin/modelscope" download \
      --model "$MODEL_ID" \
      --local_dir "$MODEL_DIR"

  model_is_complete || die "model download ended but required files are incomplete"
  info "model verified: config.json, audiovae.pth and safetensors are present"
}

mirror_has_version() {
  local package="$1" version="$2" python_exe="$3" wheel_links="$4" output=""
  # mirrors.aliyun.com/pytorch-wheels/<cuda>/ is a flat wheel directory, not
  # a PEP 503 project index.  It must be passed as --find-links; using it as
  # --index-url makes pip request a non-existent /torch/ child and return 404.
  if ! output="$(pip_cmd "$python_exe" index versions "$package" \
      --index-url "$PYPI_INDEX_URL" --find-links "$wheel_links" 2>&1)"; then
    die "could not query the configured PyTorch mirror for $package"
  fi
  printf '%s\n' "$output" | grep -Fq "$version" || \
    die "$package $version is absent from the selected mirror; refusing an unpinned fallback"
}

install_runtime() {
  local env_python torch_links torch_build torchcodec_build constraints marker
  [[ -f "$REQUIREMENTS_FILE" ]] || die "missing $REQUIREMENTS_FILE"

  install -d -m 755 "$DATA_ROOT/venvs" "$DATA_ROOT/cache"
  install -d -m 755 "$PIP_CACHE_DIR"
  ensure_venv "$ENV_DIR"
  env_python="$ENV_DIR/bin/python"
  torch_links="$PYTORCH_MIRROR_ROOT/$CUDA_TAG/"
  torch_build="$TORCH_VERSION+$CUDA_TAG"
  torchcodec_build="$TORCHCODEC_VERSION+$CUDA_TAG"
  constraints="$ENV_DIR/.voxcpm2-constraints.txt"
  marker="$ENV_DIR/.voxcpm2-runtime-ready"

  mirror_has_version torch "$torch_build" "$env_python" "$torch_links"
  mirror_has_version torchaudio "$torch_build" "$env_python" "$torch_links"
  mirror_has_version torchcodec "$torchcodec_build" "$env_python" "$torch_links"

  cat > "$constraints" <<EOF
torch==$torch_build
torchaudio==$torch_build
torchcodec==$torchcodec_build
flash-attn==$FLASH_ATTN_VERSION
nano-vllm-voxcpm==$NANO_VERSION
websockets==$WEBSOCKETS_VERSION
EOF

  info "installing pinned PyTorch $torch_build from the Aliyun PyTorch mirror"
  pip_cmd "$env_python" install \
    --index-url "$PYPI_INDEX_URL" \
    --find-links "$torch_links" \
    "torch==$torch_build" "torchaudio==$torch_build" "torchcodec==$torchcodec_build"

  info "installing FlashAttention build prerequisites from the Aliyun PyPI mirror"
  pip_cmd "$env_python" install \
    --index-url "$PYPI_INDEX_URL" \
    'packaging==24.2' 'ninja==1.11.1.3' 'psutil==6.1.1'

  info "installing FlashAttention; this may compile locally but does not touch Pipecat"
  pip_cmd "$env_python" install \
    --index-url "$PYPI_INDEX_URL" \
    --find-links "$torch_links" \
    --constraint "$constraints" \
    --no-build-isolation \
    "flash-attn==$FLASH_ATTN_VERSION"

  info "installing the isolated VoxCPM2 runtime"
  pip_cmd "$env_python" install \
    --index-url "$PYPI_INDEX_URL" \
    --find-links "$torch_links" \
    --constraint "$constraints" \
    --requirement "$REQUIREMENTS_FILE"

  pip_cmd "$env_python" check

  env -u PYTHONHOME -u PYTHONPATH \
    VOXCPM2_EXPECTED_CUDA_TAG="$CUDA_TAG" "$env_python" - <<'PY'
import importlib.metadata as md
import os
import torch

expected_tag = os.environ["VOXCPM2_EXPECTED_CUDA_TAG"]
expected_cuda = {"cu121": "12.1", "cu124": "12.4"}[expected_tag]
assert torch.__version__.startswith(f"2.5.1+{expected_tag}"), torch.__version__
assert torch.version.cuda == expected_cuda, torch.version.cuda
assert torch.cuda.is_available(), "torch cannot see a CUDA device"
for package, expected in {
    "nano-vllm-voxcpm": "2.0.3",
    "torchcodec": f"0.1.1+{expected_tag}",
    "flash-attn": "2.8.3.post1",
    "websockets": "14.1",
}.items():
    actual = md.version(package)
    assert actual == expected, (package, actual, expected)
import nanovllm_voxcpm  # noqa: F401
PY

  cat > "$marker" <<EOF
runtime=nano-vllm-voxcpm-$NANO_VERSION
python=3.11
torch=$torch_build
torchcodec=$torchcodec_build
flash_attn=$FLASH_ATTN_VERSION
websockets=$WEBSOCKETS_VERSION
EOF
  info "runtime verified: $ENV_DIR"
}

host_preflight

if [[ "$MODE" == "check" ]]; then
  info "check complete; no files were written"
  exit 0
fi

if [[ "$MODE" == "download" ]]; then
  model_download
  info "download-only complete"
  exit 0
fi

install_runtime
model_download
info "setup complete"
info "model path: $MODEL_DIR"
info "runtime python: $ENV_DIR/bin/python"
