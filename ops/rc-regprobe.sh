#!/usr/bin/env bash
# Register pressure of 4E's timed token symbol, module by module, at $0 on the Mac (step R6 of the decode-speed
# research phase): the devel ptxas 12.4 on the PTX of dumped module texts, the timed entry alone, at its own cap
# (__launch_bounds__(512, 1): 128 registers) and at tighter caps (--maxrregcount), so modules that all read 128
# registers and 0 B at the real cap can be ranked by what they spill when squeezed.
#
#   cargo test -p llvq-cuda --lib -- --ignored dump_tree_modules    # writes target/tree-modules/<name>.cu
#   ops/rc-regprobe.sh <name> ...                                    # e.g. 4e-v9e1 4e-v9ep 4e-v9ep2
#
# The PTX comes from the devel nvcc 12.4 (NVVM, as NVRTC 12.4 on the card); the card's driver ptxas (580.178.04) is
# a later allocator and decides. These counts are a proxy to rank modules, never a forecast of the card's.
# Writes target/tree-modules/<name>.ptx and <name>.regprobe.txt; prints one line a module and a cap.
# ENTRY overrides the symbol (default pers_4e_tokens_f1_t512); CAPS the caps (default "0 120 112 104 96", 0 = none).
set -euo pipefail
cd "$(dirname "$0")/.."
D=target/tree-modules
ENTRY=${ENTRY:-pers_4e_tokens_f1_t512}
CAPS=${CAPS:-0 120 112 104 96}
for name in "$@"; do
  [ -f "$D/$name.cu" ] || { echo "no $D/$name.cu: run dump_tree_modules first" >&2; exit 1; }
done
docker run --rm --platform linux/arm64 -v "$PWD":/src -w /src nvidia/cuda:12.4.1-devel-ubuntu22.04 bash -c "
set -euo pipefail
for name in $*; do
  [ -f $D/\$name.ptx ] && [ $D/\$name.ptx -nt $D/\$name.cu ] || nvcc -arch=sm_89 -ptx -O3 -o $D/\$name.ptx $D/\$name.cu
  : > $D/\$name.regprobe.txt
  for cap in $CAPS; do
    flag=''; [ \$cap = 0 ] || flag=\"--maxrregcount \$cap\"
    ptxas -arch=sm_89 -O3 -v \$flag -e $ENTRY -o /dev/null $D/\$name.ptx 2>&1 \
      | grep -A2 \"Function properties for $ENTRY\" | tr -s ' ' | tr '\n' ' ' \
      | sed \"s/^/\$name cap \$cap: /\" >> $D/\$name.regprobe.txt
    echo >> $D/\$name.regprobe.txt
  done
done
" >/dev/null 2>&1 || true
for name in "$@"; do cat "$D/$name.regprobe.txt"; done
