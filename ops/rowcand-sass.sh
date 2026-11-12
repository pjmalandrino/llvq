#!/usr/bin/env bash
# SASS instructions a block of 24 weights for every registered row arm (llvq-cuda/kernels/rowcand.cu, ROWC_ARMS), at
# $0, on the Mac: the column the decode-speed research phase (R2) reports beside each arm's stage-1 verdict.
#
#   cargo test -p llvq-cuda --lib -- --ignored dump_rowcand_module   # writes target/rowcand/rowcand_unit.cu (tile 64)
#   ops/rowcand-sass.sh                                               # devel nvcc 12.4, sm_89: indicative
#   CUBIN=<dir>/rowbench.cubin ops/rowcand-sass.sh                    # the card's own driver cubin (rowbench --dump)
#
# Needs Docker (nvidia/cuda:12.4.1-devel-ubuntu22.04, arm64, as ops/tetra-sass-probe.sh and ops/tree-sass-probe.sh).
# For each arm, the kernel tv_rowc_<name> is disassembled (cuobjdump -sass -fun) and its loops counted
# (ops/sass_loops.py). The block loop is the SHORTEST backward-branch loop that loads from global memory (LDG or a
# generic LD) and reads the activation from shared memory (LDS), with no barrier, no binary16 conversion (HADD2: the
# tail loops, 23 and 77 instructions, are shorter than the block loop) and no store to shared memory (STS: staging): the kernel nests a site loop (with
# barriers) and a row loop (block loop plus epilogue) around it. A diagnostic (ROWC_NONE) needs no LDS. Instructions a
# block = the loop's count / BPI, BPI the arm's registered blocks an iteration. The card's driver ptxas compiles
# NVRTC's PTX, so a devel count differs by a few instructions (243 on the card against 237 here, pers-sass and
# sass-tetra-int8): compare arms within one source, never across the two.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-target/rowcand}
mkdir -p "$OUT"
CUBIN=${CUBIN:-}
ARMS=$(python3 -I -c '
import re, sys
t = open("llvq-cuda/kernels/rowcand.cu").read()
blk = t[t.index("#define ROWC_ARMS(X)"):]
for line in blk.splitlines()[1:]:
    line = line.strip().rstrip("\\").strip()
    if not line:
        break
    f = [x.strip() for x in line[2:-1].split(",")]
    print(f"{f[0]}:{f[2]}:{f[5]}")
')
if [ -z "$CUBIN" ]; then
  [ -f "$OUT/rowcand_unit.cu" ] || { echo "no $OUT/rowcand_unit.cu: run dump_rowcand_module first" >&2; exit 1; }
  CUBIN=$OUT/rowcand_unit.cubin
  docker run --rm --platform linux/arm64 -v "$PWD":/src -w /src nvidia/cuda:12.4.1-devel-ubuntu22.04 \
    nvcc -arch=sm_89 -cubin -O3 -Xptxas -v -o "$CUBIN" "$OUT/rowcand_unit.cu" 2> "$OUT/rowcand_unit.ptxas.txt"
  SRC="devel nvcc 12.4, sm_89"
else
  SRC="card cubin $CUBIN"
fi
SYMS=""
for a in $ARMS; do SYMS="$SYMS tv_rowc_${a%%:*}"; done
docker run --rm --platform linux/arm64 -v "$PWD":/src -w /src nvidia/cuda:12.4.1-devel-ubuntu22.04 bash -c "
set -euo pipefail
for s in $SYMS; do cuobjdump -sass -fun \$s $CUBIN > $OUT/\$s.sass; done
"
printf 'SASS\tarm\towes\tbpi\tloop_instr\tinstr_a_block\tloop_ops\t(%s)\n' "$SRC"
for a in $ARMS; do
  name=${a%%:*}; rest=${a#*:}; owes=${rest%%:*}; bpi=${rest##*:}
  python3 -I ops/sass_loops.py "$OUT/tv_rowc_$name.sass" | awk -v n="$name" -v o="$owes" -v b="$bpi" '
    /^loop/ { c = $3; getline; ops = $0;
              ld = (ops ~ /LDG/ || ops ~ /LD\.E/ || ops ~ /(^|, )LD /); lds = (ops ~ /LDS/);
              if (ld && (lds || o == "ROWC_NONE") && ops !~ /BAR/ && ops !~ /HADD2/ && ops !~ /STS/ && (best == "" || c + 0 < bn + 0)) { bn = c; best = ops } }
    END { if (best == "") printf "SASS\t%s\t%s\t%s\tnone\t\tno block loop found\n", n, o, b;
          else printf "SASS\t%s\t%s\t%s\t%d\t%.1f\t%s\n", n, o, b, bn, bn / b, best }'
done
