#!/usr/bin/env bash
# Type-check the CUDA half of the workspace (llvq-cuda, and the `cuda` feature of llvq-llm and llvq-runtime) on a
# Linux machine with no GPU and no CUDA toolkit, as CI does:
#
#   ops/check-cuda-types.sh clippy --locked -p llvq-cuda --all-targets -- -D warnings
#   ops/check-cuda-types.sh clippy --locked -p llvq-llm --features cuda --all-targets -- -D warnings
#   ops/check-cuda-types.sh clippy --locked -p llvq-runtime --features cuda --all-targets -- -D warnings
#
# It proves types only and links nothing. `cargo check` and `cargo clippy` run the build scripts but link no binary,
# so the build scripts only need to succeed: cudarc reads the CUDA version from CUDARC_CUDA_VERSION instead of asking
# nvcc, and candle-kernels wants an nvcc and a CUDA root. This script hands them stand-ins in a directory of its own:
# an nvcc that answers --version and --list-gpu-code and writes empty output files, and a CUDA root whose include/
# holds an empty cuda.h. Linux has the C compiler and the OpenSSL headers the other build scripts want, so nothing
# else is faked. Nothing it builds can run, no kernel text meets NVRTC, and no CUDA test is run; the Rust types are
# real. The Mac counterpart is ops/check-linux-cuda.sh, which also fakes the C side.
set -euo pipefail
case "$(uname -s)" in
  Linux) ;;
  *) echo "refused: this script is for Linux; on a Mac use ops/check-linux-cuda.sh" >&2; exit 1 ;;
esac
REPO=$(cd "$(dirname "$0")/.." && pwd)
W=${LLVQ_CUDA_CHECK_DIR:-${TMPDIR:-/tmp}/llvq-cuda-check}
mkdir -p "$W/bin" "$W/cuda/include"
: > "$W/cuda/include/cuda.h"
cat > "$W/bin/nvcc" <<'EOF'
#!/bin/bash
for a in "$@"; do [ "$a" = "--list-gpu-code" ] && { printf 'sm_80\nsm_86\nsm_89\nsm_90\n'; exit 0; }; done
for a in "$@"; do [ "$a" = "--version" ] && { echo "Cuda compilation tools, release 12.4, V12.4.131"; exit 0; }; done
out=""; prev=""
for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; [ "$prev" = "--output-file" ] && out="$a"; prev="$a"; done
[ -n "$out" ] && : > "$out"
exit 0
EOF
chmod +x "$W/bin/nvcc"
export PATH="$W/bin:$PATH" CUDA_COMPUTE_CAP=89 CUDARC_CUDA_VERSION=12040 CUDA_PATH="$W/cuda"
cd "$REPO"
TARGET=${CARGO_TARGET_DIR:-$REPO/target}
# bindgen_cuda include_str!s one PTX per kernel file, which the stand-in nvcc does not always write: an empty one
# for each name it lists, in every candle-kernels output directory, before the build that reads them. The first try
# runs without colour so that its error text can be matched.
if ! CARGO_TERM_COLOR=never cargo "$@" 2> "$W/first.err"; then
  if grep -q "candle-kernels.*\.ptx\`: No such file" "$W/first.err" || grep -q "couldn't read .*candle-kernels" "$W/first.err"; then
    for d in "$TARGET"/*/build/candle-kernels-*/out; do
      for n in affine binary cast conv fill indexing quantized reduce sort ternary unary; do [ -f "$d/$n.ptx" ] || : > "$d/$n.ptx"; done
    done
    cargo "$@"
  else
    cat "$W/first.err" >&2; exit 1
  fi
else
  cat "$W/first.err" >&2
fi
