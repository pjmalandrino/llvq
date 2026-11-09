#!/usr/bin/env bash
# Count the inner-loop SASS of the served Tetra block and of its variants, at $0, on the Mac.
#
#   ops/tetra-sass-probe.sh
#
# Compiles ops/tetra-sass-probe/probe.cu after the served headers (the order of the F1 floor unit) and
# the F1 floor unit itself, with the devel nvcc 12.4 for sm_89, in nvidia/cuda:12.4.1-devel-ubuntu22.04.
# The counts are indicative: the card's driver ptxas compiles the served PTX (243 instructions a block
# in pers-sass-2026-11-05, against 237 here). Journal: docs/mesures/sass-tetra-int8-2026-11-08.txt.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-target/tetra-sass-probe}
mkdir -p "$OUT"
K=llvq-cuda/kernels
{ echo '#define TILE_BLOCKS 64u'; cat $K/llvq_slot.cuh $K/matvec.cu $K/llvq_f1rank.cuh $K/llvq_f1rank_v3.cuh \
  $K/llvq_tetra48.cuh ops/tetra-sass-probe/probe.cu; } > "$OUT/probe_unit.cu"
{ echo '#define TILE_BLOCKS 128u'; cat $K/llvq_slot.cuh $K/matvec.cu $K/llvq_f1rank.cuh $K/f1rank.cu \
  $K/llvq_f1rank_v1.cuh $K/f1rank_v1.cu $K/llvq_f1rank_v2.cuh $K/f1rank_v2.cu $K/llvq_f1rank_v3.cuh $K/f1rank_v3.cu \
  $K/nullk.cu; } > "$OUT/floor_unit.cu"
docker run --rm --platform linux/arm64 -v "$PWD":/src -w /src nvidia/cuda:12.4.1-devel-ubuntu22.04 bash -c "
set -euo pipefail
for u in probe floor; do nvcc -arch=sm_89 -cubin -O3 -o $OUT/\$u.cubin $OUT/\${u}_unit.cu; done
for f in row_f32 row_f32_trio row_ex_sp row_ex_soa row_ex_all row_dec row_i8 row_i8_sp row_i8_best row_i16_best row_lean_f32 row_lean_i8 row_lean_i16; do
  cuobjdump -sass -fun \$f $OUT/probe.cubin > $OUT/\$f.sass; done
for f in tv_f1r_word tv_f1r tv_f1r_v1 tv_f1r_v2 tv_f1r_v3; do cuobjdump -sass -fun \$f $OUT/floor.cubin > $OUT/\$f.sass; done
" > /dev/null
for f in row_f32 row_f32_trio row_ex_sp row_ex_soa row_ex_all row_dec row_i8 row_i8_sp row_i8_best row_i16_best row_lean_f32 row_lean_i8 row_lean_i16 \
         tv_f1r_word tv_f1r tv_f1r_v1 tv_f1r_v2 tv_f1r_v3; do
  # The block loop: the longest backward branch that loads a word (LDG) and the activation (LDS), with no barrier;
  # the floor kernels nest it in a tile loop. row_dec reads no activation.
  python3 -I ops/sass_loops.py "$OUT/$f.sass" | awk -v f="$f" '
    /^loop/ { n = $3; getline; ops = $0; if (ops ~ /LDG/ && (ops ~ /LDS/ || f == "row_dec") && (best == "" || n + 0 > bn + 0) && ops !~ /BAR/) { bn = n; best = ops } }
    END { printf "%-14s %4s  %s\n", f, bn, best }'
done
