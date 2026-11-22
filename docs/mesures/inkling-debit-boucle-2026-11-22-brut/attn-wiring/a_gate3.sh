#!/bin/bash
# Gate 3 of the attention wiring: one WikiText-2 window, 2,047 positions teacher-forced through the one-row path,
# --attn fused against --attn candle, both with --glue fused. Then clippy and the lib tests on the fixed worktree.
set -u
A=$HOME/llvq-inkling-mini/lot4-a; B=$A/bin/w1/inkrun; R=$A/runs; L=$HOME/llvq-inkling-mini/lot4-coord/night
H=$HOME/.llvq-guard/heavy.sh
for a in candle fused; do
  $H "coord: ppl attn $a from 1" $A/run1.sh $B metal 2048 ppl_${a}_from1 --require-layers 24 --glue fused --attn $a --score-from 1 --prompt-len 1025 --nll-out $R/ppl_${a}_from1.nll.json; echo "$(date '+%T') ppl_${a}_from1 rc=$?"
  $H "coord: ppl attn $a from 1024" $A/run1.sh $B metal 2048 ppl_${a}_from1024 --require-layers 24 --glue fused --attn $a --score-from 1024 --nll-out $R/ppl_${a}_from1024.nll.json; echo "$(date '+%T') ppl_${a}_from1024 rc=$?"
done
python3 -I $L/pplpair_a.py ppl_candle_from1 ppl_candle_from1024 -- ppl_fused_from1 ppl_fused_from1024
W=$WORKTREE
cd $W || exit 2
$H "coord: clippy attention wiring" nice -n 10 cargo clippy -p llvq-llm --features metal --all-targets -j 4 > $L/a_clippy2.log 2>&1; echo "clippy exit $? warnings $(grep -c '^warning' $L/a_clippy2.log) errors $(grep -c '^error' $L/a_clippy2.log)"
$H "coord: lib tests attention wiring" nice -n 10 cargo test --release -p llvq-llm --features metal -j 4 --lib inkling -- --test-threads=1 > $L/a_lib2.log 2>&1; echo "lib exit $?"
grep -E "test result:|^test .* FAILED|^error" $L/a_lib2.log | cut -c1-150
echo "end $(date '+%T')"
