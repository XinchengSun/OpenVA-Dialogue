#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
TMP_PARENT="${TMPDIR:-/tmp}"
TMP_ROOT="$(mktemp -d "${TMP_PARENT%/}/voxcpm2-lifecycle.XXXXXX")"
FOREIGN_PID=""

cleanup() {
  set +e
  if [[ -x "${TMP_ROOT}/voice_service/run_bridge.sh" ]]; then
    "${TMP_ROOT}/voice_service/run_bridge.sh" stop "${TMP_ROOT}/bridge.env" \
      >/dev/null 2>&1
  fi
  if [[ -n "${FOREIGN_PID}" ]]; then
    kill "${FOREIGN_PID}" >/dev/null 2>&1 || true
  fi
  case "${TMP_ROOT}" in
    "${TMP_PARENT%/}"/voxcpm2-lifecycle.*) rm -rf -- "${TMP_ROOT}" ;;
    *) printf 'refusing unsafe test cleanup: %s\n' "${TMP_ROOT}" >&2 ;;
  esac
}
trap cleanup EXIT

mkdir -p -- "${TMP_ROOT}/voice_service" "${TMP_ROOT}/model"
cp -- "${SOURCE_ROOT}/voice_service/run_bridge.sh" \
  "${TMP_ROOT}/voice_service/run_bridge.sh"
chmod 700 "${TMP_ROOT}/voice_service/run_bridge.sh"
: >"${TMP_ROOT}/ref.wav"

cat >"${TMP_ROOT}/fake-python" <<'FAKE'
#!/usr/bin/env bash
set -Eeuo pipefail
if [[ "${1:-}" == "-" ]]; then
  code="$(cat)"
  if [[ "${code}" == *"socket.create_connection"* ]]; then
    exit 1
  fi
  [[ -f "${FAKE_HEALTH_FILE}" ]]
  exit
fi
if [[ "${1:-}" == "-m" && "${2:-}" == "voice_service.voxcpm2_server" ]]; then
  [[ "${CUDA_VISIBLE_DEVICES:-}" == "7" ]]
  [[ "${VOXCPM2_DEVICES:-}" == "0" ]]
  trap 'rm -f -- "${FAKE_HEALTH_FILE}"; exit 0' TERM INT
  : >"${FAKE_HEALTH_FILE}"
  while true; do sleep 0.1; done
fi
exit 2
FAKE
chmod 700 "${TMP_ROOT}/fake-python"

# Git Bash on the local development host lacks util-linux setsid. Production
# Linux uses the real binary; this exec-only shim is sufficient for lifecycle
# logic and PID-identity testing.
cat >"${TMP_ROOT}/setsid" <<'FAKE_SET_SID'
#!/usr/bin/env bash
exec "$@"
FAKE_SET_SID
chmod 700 "${TMP_ROOT}/setsid"

cat >"${TMP_ROOT}/bridge.env" <<ENV
PATH=${TMP_ROOT}:/usr/bin:/bin
CUDA_VISIBLE_DEVICES=7
VOXCPM2_PYTHON=${TMP_ROOT}/fake-python
VOXCPM2_MODEL_PATH=${TMP_ROOT}/model
VOXCPM2_PROMPT_WAV=${TMP_ROOT}/ref.wav
VOXCPM2_START_TIMEOUT_SEC=5
VOXCPM2_STOP_TIMEOUT_SEC=5
FAKE_HEALTH_FILE=${TMP_ROOT}/health.ready
SECRET_SENTINEL=must-not-be-printed
printf '%s\n' "\${SECRET_SENTINEL}"
printf '%s\n' "\${SECRET_SENTINEL}" >&2
ENV
chmod 600 "${TMP_ROOT}/bridge.env"

output="$({
  "${TMP_ROOT}/voice_service/run_bridge.sh" start "${TMP_ROOT}/bridge.env"
  "${TMP_ROOT}/voice_service/run_bridge.sh" status "${TMP_ROOT}/bridge.env"
  "${TMP_ROOT}/voice_service/run_bridge.sh" restart "${TMP_ROOT}/bridge.env"
  "${TMP_ROOT}/voice_service/run_bridge.sh" stop "${TMP_ROOT}/bridge.env"
} 2>&1)"
[[ "${output}" == *"health=ok"* ]]
[[ "${output}" != *"must-not-be-printed"* ]]
[[ ! -e "${TMP_ROOT}/logs/voxcpm2_bridge.pid" ]]
[[ ! -e "${TMP_ROOT}/health.ready" ]]

# A reused/foreign PID must never be signalled, even when it is alive.
(
  cd -- "${TMP_ROOT}"
  sleep 30
) &
FOREIGN_PID=$!
printf '%s\n' "${FOREIGN_PID}" >"${TMP_ROOT}/logs/voxcpm2_bridge.pid"
if "${TMP_ROOT}/voice_service/run_bridge.sh" stop "${TMP_ROOT}/bridge.env" \
  >/dev/null 2>&1; then
  printf 'foreign PID stop unexpectedly succeeded\n' >&2
  exit 1
fi
kill -0 "${FOREIGN_PID}"

printf 'fake VoxCPM2 bridge lifecycle: OK\n'
