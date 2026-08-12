#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if ! command -v setsid >/dev/null 2>&1 || [[ ! -d /proc/$$ ]]; then
  printf 'Fish S2 Pro lifecycle: SKIP (Linux setsid/proc required)\n'
  exit 0
fi

TMP_PARENT="${TMPDIR:-/tmp}"
TMP_ROOT="$(mktemp -d "${TMP_PARENT%/}/fish-s2pro-lifecycle.XXXXXX")"
FOREIGN_PID=""

cleanup() {
  set +e
  if [[ -x "${TMP_ROOT}/scripts/manage_fish_s2pro.sh" ]]; then
    "${TMP_ROOT}/scripts/manage_fish_s2pro.sh" stop "${TMP_ROOT}/fish.env" \
      >/dev/null 2>&1
  fi
  if [[ -n "${FOREIGN_PID}" ]]; then
    kill -TERM "${FOREIGN_PID}" >/dev/null 2>&1 || true
  fi
  case "${TMP_ROOT}" in
    "${TMP_PARENT%/}"/fish-s2pro-lifecycle.*) rm -rf -- "${TMP_ROOT}" ;;
    *) printf 'refusing unsafe test cleanup: %s\n' "${TMP_ROOT}" >&2 ;;
  esac
}
trap cleanup EXIT

mkdir -p -- "${TMP_ROOT}/scripts" "${TMP_ROOT}/bin" "${TMP_ROOT}/logs"
cp -- "${SOURCE_ROOT}/scripts/manage_fish_s2pro.sh" \
  "${TMP_ROOT}/scripts/manage_fish_s2pro.sh"
chmod 700 "${TMP_ROOT}/scripts/manage_fish_s2pro.sh"

cat >"${TMP_ROOT}/bin/curl" <<'FAKE_CURL'
#!/usr/bin/env bash
set -Eeuo pipefail
last="${!#}"
if [[ "${last}" == */health ]]; then
  [[ "${FAKE_FORCE_UNHEALTHY:-0}" != "1" ]] || exit 1
  [[ -f "${FAKE_HEALTH_FILE}" ]]
  exit
fi
exit 1
FAKE_CURL
chmod 700 "${TMP_ROOT}/bin/curl"

cat >"${TMP_ROOT}/bin/mv" <<'FAKE_MV'
#!/usr/bin/env bash
set -Eeuo pipefail
destination="${!#}"
if [[ -n "${FAKE_MV_FAIL_DEST:-}" && "${destination}" == *"${FAKE_MV_FAIL_DEST}" ]]; then
  exit 1
fi
exec /bin/mv "$@"
FAKE_MV
chmod 700 "${TMP_ROOT}/bin/mv"

cat >"${TMP_ROOT}/scripts/run_fish_s2pro_dual_gpu.sh" <<'FAKE_FISH'
#!/usr/bin/env bash
set -Eeuo pipefail
[[ "${1:-}" == "low_ttfa_gapless" ]]
[[ "${FISH_CUDA_VISIBLE_DEVICES:-}" == "2,3" ]]
child=""
cleanup() {
  rm -f -- "${FAKE_HEALTH_FILE}"
  [[ -z "${child}" ]] || wait "${child}" 2>/dev/null || true
  exit 0
}
trap cleanup TERM INT
(trap 'exit 0' TERM INT; while true; do sleep 0.1; done) &
child=$!
printf '%s\n' "${child}" >"${FAKE_CHILD_PID_FILE}"
: >"${FAKE_HEALTH_FILE}"
while true; do sleep 0.1; done
FAKE_FISH
chmod 700 "${TMP_ROOT}/scripts/run_fish_s2pro_dual_gpu.sh"

cat >"${TMP_ROOT}/fish.env" <<ENV
PATH=${TMP_ROOT}/bin:/usr/bin:/bin
FISH_CUDA_VISIBLE_DEVICES=2,3
FISH_HTTP_HOST=127.0.0.1
FISH_HTTP_PORT=8001
FISH_START_TIMEOUT_SEC=5
FISH_STOP_TIMEOUT_SEC=5
FAKE_HEALTH_FILE=${TMP_ROOT}/health.ready
FAKE_CHILD_PID_FILE=${TMP_ROOT}/child.pid
SECRET_SENTINEL=must-not-be-printed
printf '%s\n' "\${SECRET_SENTINEL}"
printf '%s\n' "\${SECRET_SENTINEL}" >&2
ENV
chmod 600 "${TMP_ROOT}/fish.env"

output="$({
  "${TMP_ROOT}/scripts/manage_fish_s2pro.sh" start "${TMP_ROOT}/fish.env"
  "${TMP_ROOT}/scripts/manage_fish_s2pro.sh" status "${TMP_ROOT}/fish.env"
  "${TMP_ROOT}/scripts/manage_fish_s2pro.sh" start "${TMP_ROOT}/fish.env"
  "${TMP_ROOT}/scripts/manage_fish_s2pro.sh" restart "${TMP_ROOT}/fish.env"
  "${TMP_ROOT}/scripts/manage_fish_s2pro.sh" stop "${TMP_ROOT}/fish.env"
} 2>&1)"
[[ "${output}" == *"status=ready health=ok"* ]]
[[ "${output}" == *"already ready"* ]]
[[ "${output}" != *"must-not-be-printed"* ]]
[[ ! -e "${TMP_ROOT}/logs/fish_s2pro.pid" ]]
[[ ! -e "${TMP_ROOT}/health.ready" ]]

# If the PID half of state disappears, the verified identity must be retained
# and remain sufficient for a safe stop instead of being silently deleted.
"${TMP_ROOT}/scripts/manage_fish_s2pro.sh" start "${TMP_ROOT}/fish.env" >/dev/null
rm -f -- "${TMP_ROOT}/logs/fish_s2pro.pid"
"${TMP_ROOT}/scripts/manage_fish_s2pro.sh" stop "${TMP_ROOT}/fish.env" >/dev/null
[[ ! -e "${TMP_ROOT}/logs/fish_s2pro.identity" ]]
[[ ! -e "${TMP_ROOT}/health.ready" ]]

# A live process whose identity becomes unverifiable during startup keeps its
# state and receives no signal. Restore the known-good identity only for test
# cleanup, then stop it through the normal verified path.
cp -- "${TMP_ROOT}/fish.env" "${TMP_ROOT}/unhealthy.env"
printf 'FAKE_FORCE_UNHEALTHY=1\nFISH_START_TIMEOUT_SEC=4\n' \
  >>"${TMP_ROOT}/unhealthy.env"
"${TMP_ROOT}/scripts/manage_fish_s2pro.sh" start "${TMP_ROOT}/unhealthy.env" \
  >"${TMP_ROOT}/unhealthy.out" 2>&1 &
MANAGER_PID=$!
for _ in {1..50}; do
  [[ -f "${TMP_ROOT}/logs/fish_s2pro.identity" ]] && break
  sleep 0.1
done
[[ -f "${TMP_ROOT}/logs/fish_s2pro.identity" ]]
cp -- "${TMP_ROOT}/logs/fish_s2pro.identity" "${TMP_ROOT}/good.identity"
sed -i '$s/.*/corrupted-token/' "${TMP_ROOT}/logs/fish_s2pro.identity"
if wait "${MANAGER_PID}"; then
  printf 'unverifiable Fish startup unexpectedly succeeded\n' >&2
  exit 1
fi
[[ -f "${TMP_ROOT}/logs/fish_s2pro.pid" ]]
[[ -f "${TMP_ROOT}/logs/fish_s2pro.identity" ]]
cp -- "${TMP_ROOT}/good.identity" "${TMP_ROOT}/logs/fish_s2pro.identity"
"${TMP_ROOT}/scripts/manage_fish_s2pro.sh" stop "${TMP_ROOT}/unhealthy.env" >/dev/null

# A verified pending process is actionable: status reports recovery-required,
# and stop safely terminates it and clears every state fragment.
"${TMP_ROOT}/scripts/manage_fish_s2pro.sh" start "${TMP_ROOT}/fish.env" >/dev/null
mapfile -t active_identity <"${TMP_ROOT}/logs/fish_s2pro.identity"
printf 'v1\n%s\n%s\n%s\n%s\n%s\ninjected-recovery\n' \
  "${active_identity[1]}" "${active_identity[2]}" "${active_identity[3]}" \
  "${active_identity[4]}" "${active_identity[5]}" \
  >"${TMP_ROOT}/logs/fish_s2pro.pending"
rm -f -- "${TMP_ROOT}/logs/fish_s2pro.pid" \
  "${TMP_ROOT}/logs/fish_s2pro.identity"
if "${TMP_ROOT}/scripts/manage_fish_s2pro.sh" status "${TMP_ROOT}/fish.env" \
  >/dev/null 2>&1; then
  printf 'pending Fish status unexpectedly reported ready\n' >&2
  exit 1
else
  [[ "$?" -eq 4 ]]
fi
"${TMP_ROOT}/scripts/manage_fish_s2pro.sh" stop "${TMP_ROOT}/fish.env" >/dev/null
[[ ! -e "${TMP_ROOT}/logs/fish_s2pro.pending" ]]

# A partial state commit must not be mistaken for success. The newly launched
# process group is stopped and the already-committed identity fragment removed.
cp -- "${TMP_ROOT}/fish.env" "${TMP_ROOT}/state-failure.env"
printf 'FAKE_MV_FAIL_DEST=fish_s2pro.pid\n' >>"${TMP_ROOT}/state-failure.env"
if "${TMP_ROOT}/scripts/manage_fish_s2pro.sh" start \
  "${TMP_ROOT}/state-failure.env" >/dev/null 2>&1; then
  printf 'injected Fish state-write failure unexpectedly succeeded\n' >&2
  exit 1
fi
[[ ! -e "${TMP_ROOT}/health.ready" ]]
[[ ! -e "${TMP_ROOT}/logs/fish_s2pro.pid" ]]
[[ ! -e "${TMP_ROOT}/logs/fish_s2pro.identity" ]]
[[ ! -e "${TMP_ROOT}/logs/fish_s2pro.pending" ]]
[[ ! -e "${TMP_ROOT}/health.ready" ]]

# A stale pending record clears itself without signalling a reused PID.
printf 'v1\n99999999\n99999999\n99999999\n1\nstale-token\nstale-test\n' \
  >"${TMP_ROOT}/logs/fish_s2pro.pending"
if "${TMP_ROOT}/scripts/manage_fish_s2pro.sh" status "${TMP_ROOT}/fish.env" \
  >/dev/null 2>&1; then
  printf 'stale pending Fish status unexpectedly reported ready\n' >&2
  exit 1
else
  [[ "$?" -eq 3 ]]
fi
[[ ! -e "${TMP_ROOT}/logs/fish_s2pro.pending" ]]

# Runtime GPU overlap must be rejected even if a private env was edited after
# configuration generation.
cp -- "${TMP_ROOT}/fish.env" "${TMP_ROOT}/overlap.env"
printf 'CUDA_VISIBLE_DEVICES=0,2\n' >>"${TMP_ROOT}/overlap.env"
if "${TMP_ROOT}/scripts/manage_fish_s2pro.sh" start "${TMP_ROOT}/overlap.env" \
  >/dev/null 2>&1; then
  printf 'overlapping Fish/DyStream GPU map unexpectedly succeeded\n' >&2
  exit 1
fi
[[ ! -e "${TMP_ROOT}/logs/fish_s2pro.pid" ]]

# A live but unproven PID must never receive a signal.
(
  cd -- "${TMP_ROOT}"
  exec setsid sleep 30
) &
FOREIGN_PID=$!
sleep 0.1
raw="$(<"/proc/${FOREIGN_PID}/stat")"
rest="${raw##*) }"
read -r -a fields <<<"${rest}"
printf '%s\n' "${FOREIGN_PID}" >"${TMP_ROOT}/logs/fish_s2pro.pid"
printf 'v1\n%s\n%s\n%s\n%s\nforeign-token\n' \
  "${FOREIGN_PID}" "${fields[2]}" "${fields[3]}" "${fields[19]}" \
  >"${TMP_ROOT}/logs/fish_s2pro.identity"
if "${TMP_ROOT}/scripts/manage_fish_s2pro.sh" stop "${TMP_ROOT}/fish.env" \
  >/dev/null 2>&1; then
  printf 'foreign PID stop unexpectedly succeeded\n' >&2
  exit 1
fi
kill -0 "${FOREIGN_PID}"

printf 'Fish S2 Pro lifecycle: OK\n'
