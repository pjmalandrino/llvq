#!/usr/bin/env bash
# Type-check the macOS half of the workspace (llvq-metal, and the `metal` feature of llvq-llm and llvq-runtime, NEON
# arm of the CPU backend included) on a Linux machine, as CI does:
#
#   ops/check-macos-types.sh clippy --locked -p llvq-metal --all-targets -- -D warnings
#   ops/check-macos-types.sh clippy --locked -p llvq-llm --features metal --all-targets -- -D warnings
#   ops/check-macos-types.sh clippy --locked -p llvq-runtime --features metal --all-targets -- -D warnings
#
# It proves types only and links nothing. The target is aarch64-apple-darwin, whose standard library rustup installs
# on any host (`rustup target add aarch64-apple-darwin`); `cargo check` and `cargo clippy` run the build scripts but
# link no binary, so the build scripts only need to succeed. The ones that compile C (onig, zstd-sys) want a compiler
# and an archiver for the target: this script hands them stand-ins in a directory of its own, a compiler that writes
# an empty object and an archiver that writes an empty archive. No Apple SDK is needed, because nothing is linked
# against a framework. Nothing it builds can run, no shader meets the Metal compiler, and no Metal test is run; the
# Rust types are real. The CUDA counterpart is ops/check-cuda-types.sh; from a Mac, ops/check-linux-cuda.sh checks
# the Linux half the same way.
set -euo pipefail
case "$(uname -s)" in
  Linux) ;;
  *) echo "refused: this script is for Linux; on a Mac, cargo clippy --features metal is the real thing" >&2; exit 1 ;;
esac
TRIPLE=aarch64-apple-darwin
rustup target list --installed | grep -qx "$TRIPLE" \
  || { echo "refused: the $TRIPLE standard library is not installed (rustup target add $TRIPLE)" >&2; exit 1; }
REPO=$(cd "$(dirname "$0")/.." && pwd)
W=${LLVQ_MACOS_CHECK_DIR:-${TMPDIR:-/tmp}/llvq-macos-check}
mkdir -p "$W/bin"
cat > "$W/bin/fakecc" <<'FAKE'
#!/bin/bash
out=""; prev=""
for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; case "$a" in -o?*) out="${a#-o}";; esac; prev="$a"; done
[ -n "$out" ] && : > "$out"
exit 0
FAKE
cat > "$W/bin/fakear" <<'FAKE'
#!/bin/bash
for a in "$@"; do case "$a" in *.a) : > "$a";; esac; done
exit 0
FAKE
chmod +x "$W/bin/fakecc" "$W/bin/fakear"
export CC_aarch64_apple_darwin="$W/bin/fakecc" CXX_aarch64_apple_darwin="$W/bin/fakecc" \
  AR_aarch64_apple_darwin="$W/bin/fakear"
# The target goes through the environment and not through `--target`, which would land after a caller's `--`.
export CARGO_BUILD_TARGET="$TRIPLE"
cd "$REPO"
exec cargo "$@"
