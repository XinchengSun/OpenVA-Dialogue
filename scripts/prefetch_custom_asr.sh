#!/usr/bin/env bash
set -Eeuo pipefail

# Download streaming Paraformer through ModelScope into persistent AutoDL
# storage. This script never installs packages or edits the Pipecat venv.

MODE="download"
DATA_ROOT="${CUSTOM_CASCADE_DATA_ROOT:-${FLASHAV2AV_DATA_ROOT:-$HOME/.local/share/flashav2av}}"
PIPECAT_PYTHON="${PIPECAT_PYTHON:-$DATA_ROOT/venvs/pipecat/bin/python}"
ASR_MODEL="${PIPECAT_ASR_MODEL:-paraformer-zh-streaming}"

usage() {
  cat <<'EOF'
Usage: prefetch_custom_asr.sh [--check-only] [--data-root PATH]

  --check-only      Verify Pipecat/FunASR without constructing AutoModel or
                    downloading weights.
  --data-root PATH  Persistent cache root (default: ~/.local/share/flashav2av).
EOF
}

die() { printf '[custom-asr] ERROR: %s\n' "$*" >&2; exit 1; }
info() { printf '[custom-asr] %s\n' "$*"; }

while (($#)); do
  case "$1" in
    --check-only) MODE="check"; shift ;;
    --data-root)
      (($# >= 2)) || die "--data-root requires a value"
      DATA_ROOT="$2"
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

command -v readlink >/dev/null 2>&1 || die "readlink is required"
DATA_ROOT="$(readlink -m -- "$DATA_ROOT")"
[[ "$DATA_ROOT" == /* && "$DATA_ROOT" != "/" ]] || \
  die "data root must be an absolute non-root path"
[[ -x "$PIPECAT_PYTHON" ]] || die "Pipecat Python is not executable: $PIPECAT_PYTHON"

if [[ "$MODE" == "check" ]]; then
  env -u PYTHONHOME -u PYTHONPATH \
    PYTHONDONTWRITEBYTECODE=1 "$PIPECAT_PYTHON" - <<'PY'
import importlib.metadata as md
import funasr
print("funasr", md.version("funasr"))
PY
  info "check complete; no model was constructed or downloaded"
  exit 0
fi

CACHE_ROOT="$DATA_ROOT/cache/pipecat/modelscope"
[[ "$(readlink -m -- "$CACHE_ROOT")" == "$DATA_ROOT"/* ]] || \
  die "refusing cache path outside data root"
install -d -m 755 "$CACHE_ROOT"

info "prefetching $ASR_MODEL through the ModelScope hub"
env \
  -u PYTHONHOME \
  -u PYTHONPATH \
  -u MODELSCOPE_API_TOKEN \
  -u MODELSCOPE_SDK_TOKEN \
  -u HF_TOKEN \
  -u HUGGING_FACE_HUB_TOKEN \
  MODELSCOPE_CACHE="$CACHE_ROOT" \
  PIPECAT_ASR_MODEL="$ASR_MODEL" \
  PYTHONDONTWRITEBYTECODE=1 \
  "$PIPECAT_PYTHON" - <<'PY'
import os
from funasr import AutoModel

model_name = os.environ["PIPECAT_ASR_MODEL"]
AutoModel(model=model_name, hub="ms", device="cpu", disable_update=True)
print(f"ASR_MODEL_READY={model_name}")
PY

info "download complete; runtime will reuse $CACHE_ROOT"
