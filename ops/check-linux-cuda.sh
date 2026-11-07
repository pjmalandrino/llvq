#!/usr/bin/env bash
# Type-check the Linux CUDA half of llvq-llm (fused_cuda.rs, persist4e_cuda.rs, the cuda binaries) on a Mac.
#
#   ops/check-linux-cuda.sh check  -p llvq-llm --features cuda --all-targets
#   ops/check-linux-cuda.sh clippy -p llvq-llm --features cuda --all-targets
#   ops/check-linux-cuda.sh clippy --workspace --exclude llvq-cuda --exclude llvq-metal --all-targets   # CI's Linux build
#
# `cargo check` links nothing, so the build scripts only need to succeed: candle-kernels wants nvcc and a CUDA root,
# ring, zstd-sys and onig want a Linux C compiler, openssl-sys wants headers. This script hands them stand-ins in a
# directory of its own: an nvcc that writes empty PTX, a C compiler that writes empty objects (and runs the host clang
# to preprocess), Homebrew's OpenSSL headers. Nothing it builds can run; the Rust types are real. Found on
# 2026-11-07: before it, docs/METHODE.md's "every edit under cfg(linux) is type-checked from the Mac" covered llvq-cuda
# only, and the cuda half of llvq-llm met its first compiler on the image build.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
W=${LLVQ_LINUX_CHECK_DIR:-${TMPDIR:-/tmp}/llvq-linux-check}
OPENSSL=${OPENSSL_DIR_FOR_LINUX:-/opt/homebrew/opt/openssl@3}
test -d "$OPENSSL/include/openssl" || { echo "refused: no OpenSSL headers at $OPENSSL (brew install openssl@3)" >&2; exit 1; }
mkdir -p "$W/bin" "$W/cuda/include"
: > "$W/cuda/include/cuda.h"
cat > "$W/bin/fakecc" <<'EOF'
#!/bin/bash
comp=0; for a in "$@"; do [ "$a" = "-c" ] && comp=1; done
if [ $comp = 0 ]; then exec /usr/bin/clang "$@"; fi
out=""; prev=""
for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; case "$a" in -o?*) out="${a#-o}";; esac; prev="$a"; done
[ -n "$out" ] && : > "$out"
exit 0
EOF
cat > "$W/bin/fakear" <<'EOF'
#!/bin/bash
for a in "$@"; do case "$a" in *.a) : > "$a";; esac; done
exit 0
EOF
cat > "$W/bin/nvcc" <<'EOF'
#!/bin/bash
for a in "$@"; do [ "$a" = "--list-gpu-code" ] && { printf 'sm_80\nsm_86\nsm_89\nsm_90\n'; exit 0; }; done
for a in "$@"; do [ "$a" = "--version" ] && { echo "Cuda compilation tools, release 12.4, V12.4.131"; exit 0; }; done
out=""; prev=""
for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; [ "$prev" = "--output-file" ] && out="$a"; prev="$a"; done
[ -n "$out" ] && : > "$out"
exit 0
EOF
chmod +x "$W/bin/"*
export PATH="$W/bin:$PATH" CC_x86_64_unknown_linux_gnu="$W/bin/fakecc" CXX_x86_64_unknown_linux_gnu="$W/bin/fakecc" \
  AR_x86_64_unknown_linux_gnu="$W/bin/fakear" X86_64_UNKNOWN_LINUX_GNU_OPENSSL_DIR="$OPENSSL" CUDA_COMPUTE_CAP=89 \
  CUDARC_CUDA_VERSION=12040 CUDA_PATH="$W/cuda"
cd "$REPO"
# bindgen_cuda include_str!s one PTX per kernel file, which the stand-in nvcc does not always write: an empty one
# for each name it lists, in every candle-kernels output directory, before the build that reads them.
run() { nice -n 10 cargo "$@" --target x86_64-unknown-linux-gnu --target-dir "$W/target"; }
if ! run "$@" 2>"$W/first.err"; then
  if grep -q "candle-kernels.*\.ptx\`: No such file" "$W/first.err" || grep -q "couldn't read .*candle-kernels" "$W/first.err"; then
    for d in "$W"/target/x86_64-unknown-linux-gnu/*/build/candle-kernels-*/out; do
      for n in affine binary cast conv fill indexing quantized reduce sort ternary unary; do [ -f "$d/$n.ptx" ] || : > "$d/$n.ptx"; done
    done
    run "$@"
  else
    cat "$W/first.err" >&2; exit 1
  fi
else
  cat "$W/first.err" >&2
fi
