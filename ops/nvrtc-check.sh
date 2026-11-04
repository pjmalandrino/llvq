#!/usr/bin/env bash
# Compile the persistent units with NVRTC 12.4, as gpu.rs would, with no card,
# then read ptxas 12.4's register, spill and shared counts for sm_89.
#
#   ops/nvrtc-check.sh            # the units below, compute_89
#   ARCH=compute_80 ops/nvrtc-check.sh
#
# Runs in nvidia/cuda:12.4.1-devel-ubuntu22.04 (arm64 on the Mac, nothing
# emulated), the base of both ops/Dockerfile.cuda and ops/Dockerfile.check;
# it needs no Rust. What a green line means: NVRTC accepts the unit, and
# the devel ptxas reports what it would allocate. What it does not: the card
# JIT-compiles the PTX with the DRIVER's ptxas (580.x on the L40S), whose
# allocation may differ; the bench reads `local_bytes` from the loaded module
# on the card and refuses a spilling kernel there. Indicative, then.
set -euo pipefail
cd "$(dirname "$0")/.."
ARCH=${ARCH:-compute_89}
SM=sm_${ARCH#compute_}
OUT=${OUT:-target/nvrtc-check}
mkdir -p "$OUT"
IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
K=llvq-cuda/kernels
BASE="$K/llvq_slot.cuh $K/matvec.cu $K/llvq_f1rank.cuh $K/llvq_f1rank_v3.cuh $K/llvq_tetra48.cuh $K/llvq_rot.cuh $K/llvq_pers.cuh"
docker run --rm --platform linux/arm64 -v "$PWD":/src -w /src $IMAGE bash -c "
set -euo pipefail
gcc -O2 -o /tmp/nvrtc_compile ops/nvrtc_compile.c -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lnvrtc
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
for unit in pers:$K/tetra48_pers.cu segment:$K/pers_segment.cu; do
  name=\${unit%%:*}; file=\${unit#*:}
  /tmp/nvrtc_compile $ARCH $OUT/\$name.ptx '#define TILE_BLOCKS 64u' $BASE \$file
  ptxas -arch=$SM -v $OUT/\$name.ptx -o $OUT/\$name.cubin 2> $OUT/\$name.ptxas.txt
  echo \"== \$name ($ARCH, devel ptxas: indicative, the card's driver ptxas decides)\"
  awk '/Compiling entry function/ { match(\$0, /function .[A-Za-z0-9_]+./); f = substr(\$0, RSTART + 10, RLENGTH - 11) }
       /bytes spill stores/ { sp = \$5 + \$9 }
       /Used [0-9]+ registers/ { match(\$0, /Used [0-9]+ registers/); printf \"  %-24s %s, spill %d B\\n\", f, substr(\$0, RSTART + 5, RLENGTH - 5), sp }' $OUT/\$name.ptxas.txt
done
"
