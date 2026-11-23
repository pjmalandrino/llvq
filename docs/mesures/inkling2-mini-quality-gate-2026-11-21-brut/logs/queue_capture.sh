#!/bin/bash
# Stage A of the night of 2026-11-21: the memory-safe G2 mutant, then the full capture.
# Every heavy step goes through the guard's lock. Fails fast.
set -euo pipefail
W=/Users/pjmalandrino/Documents/Pro/workspace/poc/llvq/.claude/worktrees/inkling-mini
D=$HOME/llvq-inkling-mini
H=$HOME/.llvq-guard/heavy.sh
MUT=/private/tmp/claude-501/-Users-pjmalandrino--claude-spares-spare-VAIlTv/03c8a1bc-8db6-4a13-872d-5270e2db3f35/scratchpad/g2_mutant2.py
export INKLING_THREADS=10
cd "$W/ops/inkling"

echo "== $(date '+%F %T') G2 mutant (pass 2 dropped, 64 tokens)"
"$H" "inkling: G2 mutant, memory-safe" uv run "$MUT" > "$D/eval/g2_mutant.out" 2> "$D/logs/g2_mutant.err" || { echo "mutant run failed"; tail -5 "$D/logs/g2_mutant.err"; exit 3; }
tail -1 "$D/eval/g2_mutant.out"
python3 -I -c "
import json,sys
d=json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])
ok = d['max_abs_diff'] > 1e-3
print('mutant differs from the library forward:', ok, d)
sys.exit(0 if ok else 4)
" "$D/eval/g2_mutant.out"

echo "== $(date '+%F %T') capture, 64 x 2048, mps"
"$H" "inkling: capture 64x2048 mps" uv run capture.py --device mps --group 32 > "$D/logs/capture.log" 2>&1 || { echo "capture failed"; tail -15 "$D/logs/capture.log"; exit 5; }
tail -6 "$D/logs/capture.log"
echo "== $(date '+%F %T') capture done"
ls "$D/calib" | wc -l
du -sh "$D/calib"
df -h /Users/pjmalandrino | tail -1
