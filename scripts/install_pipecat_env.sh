#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${FLASHAV2AV_DATA_ROOT:-$HOME/.local/share/flashav2av}"
BASE_PYTHON="${BASE_PYTHON:-$(command -v python3)}"
VENV_DIR="${PIPECAT_VENV:-$DATA_ROOT/venvs/pipecat}"
ALIYUN_INDEX="${ALIYUN_PYPI_INDEX:-http://mirrors.aliyun.com/pypi/simple}"
NLTK_DIR="${NLTK_DATA:-$DATA_ROOT/cache/pipecat/nltk_data}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DATA_ROOT/cache/pipecat/pip}"

if [[ ! -x "$BASE_PYTHON" ]]; then
  echo "missing base Python: $BASE_PYTHON" >&2
  exit 1
fi
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$VENV_DIR"
fi
PYTHON_BIN="$VENV_DIR/bin/python"
mkdir -p "$PIP_CACHE_DIR"

# AutoDL's mirror currently omits num2words. Install its tiny dependency from
# the fast mirror, then fetch only the missing wheel from official PyPI.
"$PYTHON_BIN" -m pip install \
  --index-url "$ALIYUN_INDEX" --trusted-host mirrors.aliyun.com \
  docopt==0.6.2
"$PYTHON_BIN" -m pip install \
  --index-url https://pypi.org/simple --no-deps \
  num2words==0.5.14
"$PYTHON_BIN" -m pip install \
  --index-url "$ALIYUN_INDEX" --trusted-host mirrors.aliyun.com \
  -r "$ROOT_DIR/requirements-pipecat.txt"
# Pipecat's optional protobuf serializer asks for protobuf 5+, while the
# delivered MediaPipe build requires protobuf 4.x. This WebRTC path does not
# import that serializer, so keep the MediaPipe-compatible runtime and validate
# every Pipecat module we do use below.
"$PYTHON_BIN" -m pip install \
  --index-url "$ALIYUN_INDEX" --trusted-host mirrors.aliyun.com --no-deps --no-warn-conflicts \
  protobuf==4.25.9

# Override inherited binary packages whose Python/Rust or Torch versions can
# otherwise differ from the base environment visible through system-site-packages.
"$PYTHON_BIN" -m pip install \
  --index-url "$ALIYUN_INDEX" --trusted-host mirrors.aliyun.com --ignore-installed --no-warn-conflicts \
  cryptography==49.0.0
"$PYTHON_BIN" -m pip install \
  --index-url "$ALIYUN_INDEX" --trusted-host mirrors.aliyun.com --no-deps --no-warn-conflicts \
  torchaudio==2.8.0

# Pipecat imports the NLTK punkt tables from its string utilities. Download
# them explicitly so service startup never performs an implicit network call.
PUNKT_DIR="$NLTK_DIR/tokenizers/punkt_tab"
PUNKT_ZIP="$NLTK_DIR/tokenizers/punkt_tab.zip"
if [[ ! -d "$PUNKT_DIR" ]]; then
  mkdir -p "$NLTK_DIR/tokenizers"
  curl --fail --location --retry 3 \
    "https://ghfast.top/https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip" \
    --output "$PUNKT_ZIP"
  "$PYTHON_BIN" - "$PUNKT_ZIP" "$NLTK_DIR" <<'PY'
import hashlib
import pathlib
import sys
import zipfile

archive = pathlib.Path(sys.argv[1]).resolve()
destination = pathlib.Path(sys.argv[2]).resolve()
expected_sha256 = "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106"
actual_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
if actual_sha256 != expected_sha256:
    raise RuntimeError(f"punkt_tab SHA-256 mismatch: {actual_sha256}")
with zipfile.ZipFile(archive) as source:
    corrupt = source.testzip()
    if corrupt:
        raise RuntimeError(f"corrupt punkt_tab member: {corrupt}")
    for member in source.infolist():
        target = (destination / member.filename).resolve()
        if destination != target and destination not in target.parents:
            raise RuntimeError(f"unsafe punkt_tab member: {member.filename}")
    source.extractall(destination)
PY
  rm -f "$PUNKT_ZIP"
fi

NLTK_DATA="$NLTK_DIR" CUDA_VISIBLE_DEVICES="" \
  "$PYTHON_BIN" "$ROOT_DIR/scripts/check_pipecat_env.py"
