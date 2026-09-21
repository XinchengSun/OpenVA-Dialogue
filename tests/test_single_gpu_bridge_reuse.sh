#!/usr/bin/env bash
# Live /proc environment checks with a tiny real Python process, never a model.
set -Eeuo pipefail

SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
TMP_PARENT="${TMPDIR:-/tmp}"
TEST_ROOT="$(mktemp -d "${TMP_PARENT%/}/single-gpu-bridge.XXXXXX")"
REAL_PYTHON="$(command -v python3)"
MANAGER="$TEST_ROOT/voice_service/run_bridge.sh"
CONFIG="$TEST_ROOT/bridge.env"

cleanup() {
  set +e
  if [[ -f "$CONFIG" && -f "$MANAGER" ]]; then
    bash "$MANAGER" stop "$CONFIG" >/dev/null 2>&1
  fi
  case "$TEST_ROOT" in
    "${TMP_PARENT%/}"/single-gpu-bridge.*) rm -rf -- "$TEST_ROOT" ;;
    *) printf 'refusing unsafe test cleanup: %s\n' "$TEST_ROOT" >&2 ;;
  esac
}
trap cleanup EXIT

mkdir -p "$TEST_ROOT/voice_service" "$TEST_ROOT/model"
# Allow the tests to run against a Windows checkout mounted on Linux.
sed 's/\r$//' "$SOURCE_ROOT/voice_service/run_bridge.sh" > "$MANAGER"
touch "$TEST_ROOT/voice_service/__init__.py" "$TEST_ROOT/ref.wav"
cat > "$TEST_ROOT/voice_service/voxcpm2_server.py" <<'PY'
import os
from pathlib import Path
import signal

marker = Path(os.environ['FAKE_HEALTH_FILE'])

def stop(*_):
    marker.unlink(missing_ok=True)
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
marker.write_text(str(os.getpid()))
while True:
    signal.pause()
PY

# Only network readiness is simulated. The launched process, ownership checks
# and /proc/PID/environ are real, including values changed by a launcher wrapper.
cat > "$TEST_ROOT/python-wrapper" <<'SH'
#!/usr/bin/env bash
set -Eeuo pipefail
if [[ "${1:-}" == "-" ]]; then
  code="$(cat)"
  if [[ "$code" == *'socket.create_connection'* ]]; then
    exit 1
  fi
  [[ "$code" == *'async def check()'* && -f "$FAKE_HEALTH_FILE" ]]
  exit
fi
if [[ "${1:-}" == '-m' && "${2:-}" == 'voice_service.voxcpm2_server' ]]; then
  if [[ -n "${FAKE_LAUNCH_GPU:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$FAKE_LAUNCH_GPU"
  fi
  exec "$REAL_TEST_PYTHON" "$@"
fi
exit 70
SH
chmod +x "$TEST_ROOT/python-wrapper"

write_config() {
  cat > "$CONFIG" <<EOF
CUDA_VISIBLE_DEVICES=6
CUDA_DEVICE_ORDER=PCI_BUS_ID
VOXCPM2_BACKEND=official_prompt_cache
VOXCPM2_OFFICIAL_DEVICE=cuda:0
DYSTREAM_SINGLE_GPU=6
VOXCPM2_DEVICES=0
VOXCPM2_PYTHON='$TEST_ROOT/python-wrapper'
VOXCPM2_MODEL_PATH='$TEST_ROOT/model'
VOXCPM2_PROMPT_WAV='$TEST_ROOT/ref.wav'
VOXCPM2_START_TIMEOUT_SEC=5
VOXCPM2_STOP_TIMEOUT_SEC=5
FAKE_HEALTH_FILE='$TEST_ROOT/ready'
REAL_TEST_PYTHON='$REAL_PYTHON'
SECRET_SENTINEL=must-not-be-printed
EOF
  if (($#)); then
    printf '%s\n' "$@" >> "$CONFIG"
  fi
}

expect_success() {
  if ! bash "$MANAGER" "$1" "$CONFIG" > "$TEST_ROOT/output" 2>&1; then
    cat "$TEST_ROOT/output" >&2
    echo "expected $1 success" >&2
    exit 1
  fi
  ! grep -q 'must-not-be-printed' "$TEST_ROOT/output"
}

expect_rejection() {
  if bash "$MANAGER" "$1" "$CONFIG" > "$TEST_ROOT/output" 2>&1; then
    echo "expected $1 single-GPU rejection" >&2
    exit 1
  fi
  grep -q 'single-GPU' "$TEST_ROOT/output"
  grep -q 'restart' "$TEST_ROOT/output"
  ! grep -q 'must-not-be-printed' "$TEST_ROOT/output"
}

# Same-card official process is reused, with no duplicate PID.
write_config
expect_success start
first_pid="$(cat "$TEST_ROOT/logs/voxcpm2_bridge.pid")"
expect_success start
expect_success status
[[ "$(cat "$TEST_ROOT/logs/voxcpm2_bridge.pid")" == "$first_pid" ]]
expect_success stop

# A new env file does not change the environment of an already-running bridge.
for mismatch in \
  'CUDA_VISIBLE_DEVICES=7' \
  'VOXCPM2_BACKEND=nano' \
  'CUDA_DEVICE_ORDER=FASTEST_FIRST' \
  'VOXCPM2_OFFICIAL_DEVICE=cuda:1' \
  'unset CUDA_DEVICE_ORDER'; do
  write_config 'DYSTREAM_SINGLE_GPU=' "$mismatch"
  expect_success start
  old_pid="$(cat "$TEST_ROOT/logs/voxcpm2_bridge.pid")"
  write_config
  expect_rejection start
  expect_rejection status
  kill -0 "$old_pid"
  [[ "$(cat "$TEST_ROOT/logs/voxcpm2_bridge.pid")" == "$old_pid" ]]
  # stop is intentionally allowed despite the mismatch.
  expect_success stop
  [[ ! -e "$TEST_ROOT/logs/voxcpm2_bridge.pid" && ! -e "$TEST_ROOT/ready" ]]
done

# restart can recover a legacy bridge and validate the replacement process.
write_config 'DYSTREAM_SINGLE_GPU=' 'CUDA_VISIBLE_DEVICES=7' 'VOXCPM2_BACKEND=nano'
expect_success start
old_pid="$(cat "$TEST_ROOT/logs/voxcpm2_bridge.pid")"
write_config
expect_success restart
[[ "$(cat "$TEST_ROOT/logs/voxcpm2_bridge.pid")" != "$old_pid" ]]
expect_success status
expect_success stop

# Even newly launched processes are checked after exec; bad ones are reclaimed.
write_config 'FAKE_LAUNCH_GPU=7'
expect_rejection start
[[ ! -e "$TEST_ROOT/logs/voxcpm2_bridge.pid" && ! -e "$TEST_ROOT/ready" ]]

echo 'PASS: live single-GPU bridge reuse, mismatch rejection, safe stop/restart and new-process verification'
