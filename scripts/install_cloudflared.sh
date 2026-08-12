#!/usr/bin/env bash
set -euo pipefail

VERSION="${CLOUDFLARED_VERSION:-2026.7.3}"
EXPECTED_SHA256="${CLOUDFLARED_SHA256:-049777d30f9bf93da6df8bbe31383460eb2aa51a832c6551824d56f9fcc55974}"
RUNTIME_ROOT="${CUSTOM_CASCADE_RUNTIME:-${FLASHAV2AV_DATA_ROOT:-$HOME/.local/share/flashav2av}/runtime}"
TARGET="${CLOUDFLARED_BIN:-$RUNTIME_ROOT/tools/cloudflared}"
URL="https://pkg.cloudflare.com/cloudflared/pool/main/c/cloudflared/cloudflared_${VERSION}_amd64.deb"

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "ERROR: this pinned installer supports x86_64 only" >&2
  exit 1
fi

if [[ -x "$TARGET" ]]; then
  actual="$($TARGET version 2>/dev/null || true)"
  if [[ "$actual" == *"$VERSION"* ]]; then
    echo "cloudflared already installed: $TARGET ($actual)"
    exit 0
  fi
fi

temporary_dir="$(mktemp -d)"
cleanup() {
  rm -rf -- "$temporary_dir"
}
trap cleanup EXIT

command -v dpkg-deb >/dev/null 2>&1 \
  || { echo "ERROR: dpkg-deb is required to unpack the official package" >&2; exit 1; }

download="$temporary_dir/cloudflared.deb"
curl --fail --location --retry 3 --connect-timeout 10 \
  --output "$download" "$URL"
printf '%s  %s\n' "$EXPECTED_SHA256" "$download" | sha256sum --check --status

extract_dir="$temporary_dir/extracted"
dpkg-deb --extract "$download" "$extract_dir"
source_binary="$extract_dir/usr/bin/cloudflared"
[[ -x "$source_binary" ]] \
  || { echo "ERROR: official package did not contain usr/bin/cloudflared" >&2; exit 1; }

mkdir -p "$(dirname "$TARGET")"
install -m 0755 "$source_binary" "$TARGET"
"$TARGET" version
echo "installed cloudflared: $TARGET"
