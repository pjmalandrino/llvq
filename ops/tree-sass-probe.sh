#!/usr/bin/env bash
# Compile dumped tree and 4E module texts for sm_89 and write each symbol's SASS and ptxas lines, at $0, on the Mac.
#
#   cargo test -p llvq-cuda --lib -- --ignored dump_tree_modules   # writes target/tree-modules/<name>.cu
#   ops/tree-sass-probe.sh <name>:<symbol>[,<symbol>...] ...
#
# Each <name>.cu compiled once with the devel nvcc 12.4 (-O3 -Xptxas -v) in ops/tetra-sass-probe.sh's image,
# then cuobjdump -sass -fun <symbol>, written as target/tree-modules/<name>.<symbol>.sass, the ptxas lines as
# <name>.ptxas.txt. Loops: python3 -I ops/sass_loops.py <file.sass>. The counts are indicative: the card's
# driver ptxas compiles NVRTC's PTX. Journal: docs/mesures/large-attn-tuile-pleine-mac-2026-11-11.txt.
set -euo pipefail
cd "$(dirname "$0")/.."
D=target/tree-modules
for spec in "$@"; do
  name=${spec%%:*}; syms=${spec#*:}
  [ -f "$D/$name.cu" ] || { echo "no $D/$name.cu: run dump_tree_modules first" >&2; exit 1; }
  docker run --rm --platform linux/arm64 -v "$PWD":/src -w /src nvidia/cuda:12.4.1-devel-ubuntu22.04 bash -c "
set -euo pipefail
nvcc -arch=sm_89 -cubin -O3 -Xptxas -v -o $D/$name.cubin $D/$name.cu 2> $D/$name.ptxas.txt
for s in ${syms//,/ }; do cuobjdump -sass -fun \$s $D/$name.cubin > $D/$name.\$s.sass; done
"
  for s in ${syms//,/ }; do
    echo "== $name $s"
    grep -A2 "Function properties for $s\$" "$D/$name.ptxas.txt" | tail -2
  done
done
