#!/usr/bin/env bash
# Read-only official VoxCPM2 prerequisite check; does not load a GPU model.
set -euo pipefail

[[ $# -eq 1 && "$1" == "--check-only" ]] || {
  echo 'Usage: setup_single_gpu.sh --check-only (prepared official runtime required; no installer is provided)' >&2
  exit 2
}
[[ "$(uname -s)" == "Linux" ]] || { echo 'Linux is required' >&2; exit 1; }

DATA_ROOT="${FLASHAV2AV_DATA_ROOT:-$HOME/.local/share/flashav2av}"
BRIDGE_ENV="${PIPECAT_TTS_BRIDGE_ENV_FILE:-${VOXCPM2_ENV_FILE:-$DATA_ROOT/config/voxcpm2.env}}"
if [[ -f "$BRIDGE_ENV" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$BRIDGE_ENV"
  set +a
fi
PYTHON_BIN="${VOXCPM2_PYTHON:-$DATA_ROOT/venvs/voxcpm2-nano-2.0.3/bin/python}"
MODEL_PATH="${VOXCPM2_MODEL_PATH:-$DATA_ROOT/models/VoxCPM2}"
SOURCE_PATH="${VOXCPM2_OFFICIAL_SOURCE:-}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "OFFICIAL_VOXCPM2_MISSING executable Python: $PYTHON_BIN" >&2
  echo 'Prepare a separate official VoxCPM environment; configure --single-gpu with --voxcpm-python and --voxcpm-official-source. setup-voxcpm2 alone installs Nano, not the official backend.' >&2
  exit 1
fi

env -u PYTHONHOME -u PYTHONPATH \
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PYTHON_BIN" -B - "$SOURCE_PATH" "$MODEL_PATH" <<'PY'
import importlib
import json
from pathlib import Path
import sys

source, model_path = sys.argv[1:]
if source:
    source_path = Path(source)
    if not (source_path / "voxcpm" / "model" / "voxcpm2.py").is_file():
        raise SystemExit("OFFICIAL_VOXCPM2_MISSING source src/voxcpm/model/voxcpm2.py")
    sys.path.insert(0, str(source_path))
try:
    from voxcpm import VoxCPM
    module = importlib.import_module("voxcpm.model.voxcpm2")
    for dependency in ("torch", "torchaudio", "websockets", "soundfile", "numpy"):
        importlib.import_module(dependency)
except Exception as exc:
    raise SystemExit(f"OFFICIAL_VOXCPM2_IMPORT_FAILED {type(exc).__name__}: {exc}") from exc
methods = ("build_prompt_cache", "merge_prompt_cache", "generate_with_prompt_cache_streaming")
if not callable(getattr(VoxCPM, "from_pretrained", None)) or not any(
    isinstance(value, type) and all(callable(getattr(value, name, None)) for name in methods)
    for value in vars(module).values()
):
    raise SystemExit("OFFICIAL_VOXCPM2_API_MISSING prompt-cache streaming API; select a compatible official revision")

model = Path(model_path).resolve()
required = ["config.json", "audiovae.pth", "tokenizer.json", "tokenizer_config.json"]
indexes = list(model.glob("*.safetensors.index.json"))
if indexes:
    for index in indexes:
        try:
            weights = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            if not isinstance(weights, dict) or not weights:
                raise ValueError("empty weight_map")
            required.extend(set(weights.values()))
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"VOXCPM2_MODEL_INVALID index {index.name}: {exc}") from exc
else:
    weights = list(model.glob("*.safetensors"))
    if not weights:
        raise SystemExit(f"VOXCPM2_MODEL_MISSING safetensors weights: {model}")
    required.extend(path.name for path in weights)
for relative in required:
    path = (model / relative).resolve()
    if not path.is_relative_to(model) or not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"VOXCPM2_MODEL_MISSING_OR_INVALID {relative}")
print(f"OFFICIAL_VOXCPM2_PREREQUISITES_OK model={model}")
print("CHECK_ONLY: imports and required files verified; no model loaded, no GPU memory or real-time performance claim")
PY
