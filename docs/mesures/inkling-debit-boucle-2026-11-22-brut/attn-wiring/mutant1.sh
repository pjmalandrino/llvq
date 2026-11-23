#!/bin/bash
# One mutant on one file of a worktree, against one test binary: apply (exactly one site), run, restore, check sha.
# usage: mutant1.sh <worktree> <file> <name> <old> <new> <test> [features]
set -u
W=$1; F=$W/$2; name=$3; old=$4; new=$5; t=$6; feat=${7:-metal}
O=$HOME/llvq-inkling-mini/lot4-coord/night/mutants; mkdir -p $O
H0=$(shasum -a 256 "$F" | cut -d' ' -f1)
edit() { python3 -I - "$F" "$1" "$2" <<'PY'
import sys
p, a, b = sys.argv[1:4]
s = open(p).read(); n = s.count(a)
print(f"sites {n}")
if n != 1: sys.exit(3)
open(p, 'w').write(s.replace(a, b))
PY
}
echo "== $name"
edit "$old" "$new" || { echo "mutant $name NOT applied"; exit 3; }
(cd $W && nice -n 10 cargo test --release -p llvq-llm --features $feat --test $t -j 4 -- --test-threads=1 > $O/$name.log 2>&1); echo "exit $?"
grep -E "test result:|^test .* (ok|FAILED)$|^error" $O/$name.log | cut -c1-160
edit "$new" "$old" || { echo "restore of $name FAILED"; exit 4; }
[ "$(shasum -a 256 "$F" | cut -d' ' -f1)" = "$H0" ] && echo "restored, sha $H0" || { echo "file NOT restored"; exit 5; }
