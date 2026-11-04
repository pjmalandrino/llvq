#!/usr/bin/env bash
# Type-check the Linux half of llvq-llm (`--features cuda`: fused_cuda.rs, fusedrun) on a Mac, with no Docker.
#
#   ops/check-linux-on-mac.sh            # clippy -p llvq-llm --features cuda --all-targets
#   ops/check-linux-on-mac.sh check      # cargo check instead
#
# Nothing is linked and no kernel is compiled. Three build scripts need help on a Mac:
# - onig_sys (tokenizers) compiles C: clang with a Linux target and the macOS SDK headers;
# - openssl-sys reads headers only: Homebrew's openssl@3;
# - candle-kernels runs nvcc: a stub on PATH lists GPU codes and writes EMPTY outputs. A binary built
#   this way is useless; it exists only so rustc reaches our code.
# ops/check-cuda.sh (Docker, real CUDA headers) stays the reference when Docker can pull its image.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
CMD=${1:-clippy}

STUB=$(mktemp -d)
trap 'rm -rf "$STUB"' EXIT
mkdir -p "$STUB/bin" "$STUB/include"
: > "$STUB/include/cuda.h"
cat > "$STUB/bin/nvcc" <<'EOF'
#!/bin/bash
for a in "$@"; do [ "$a" = "--list-gpu-code" ] && { printf 'sm_80\nsm_86\nsm_89\nsm_90\n'; exit 0; }; done
prev=""; od=""; cus=()
for a in "$@"; do
  [ "$prev" = "-o" ] && { mkdir -p "$(dirname "$a")"; : > "$a"; }
  [ "$prev" = "--output-directory" ] && od="$a"
  case "$a" in *.cu) cus+=("$a");; esac
  prev="$a"
done
if [ -n "$od" ]; then for c in "${cus[@]}"; do : > "$od/$(basename "$c" .cu).ptx"; done; fi
exit 0
EOF
chmod +x "$STUB/bin/nvcc"

T=x86_64-unknown-linux-gnu
# candle-kernels' build script may be cached from a run without the stub: give its ptx.rs the files it names.
for d in target/$T/debug/build/candle-kernels-*/out; do
  [ -f "$d/ptx.rs" ] || continue
  grep -oE '[a-z_]+\.ptx' "$d/ptx.rs" | while read -r f; do [ -e "$d/$f" ] || : > "$d/$f"; done
done

SDK=$(xcrun --show-sdk-path)
PATH="$STUB/bin:$PATH" CUDA_ROOT="$STUB" CUDA_COMPUTE_CAP=89 \
  X86_64_UNKNOWN_LINUX_GNU_OPENSSL_DIR="$(brew --prefix openssl@3)" \
  CC_x86_64_unknown_linux_gnu=clang AR_x86_64_unknown_linux_gnu=ar \
  CFLAGS_x86_64_unknown_linux_gnu="--target=$T -isysroot $SDK -I$SDK/usr/include" \
  CUDARC_CUDA_VERSION=12040 \
  cargo "$CMD" --target "$T" -p llvq-llm --features cuda --all-targets
