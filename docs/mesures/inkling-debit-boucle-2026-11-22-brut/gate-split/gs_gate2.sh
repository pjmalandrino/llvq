#!/bin/bash
# The gate and router split on the real model: the new binary against the pull-request head's (bin/a1), bit for bit.
set -u
W=$WORKTREE
A=$HOME/llvq-inkling-mini/lot4-a; R=$A/runs; H=$HOME/.llvq-guard/heavy.sh; L=$HOME/llvq-inkling-mini/lot4-coord/night
cd $W || exit 2
$H "coord: build gate split" nice -n 10 cargo build --release -p llvq-llm --features metal -j 4 --bin inkrun > $L/gs_build.log 2>&1; echo "build exit $?"
mkdir -p $A/bin/g1 && cp target/release/inkrun $A/bin/g1/inkrun
for p in 32 600; do
  for b in a1 g1; do
    $H "coord: $b all $p" $A/run1.sh $A/bin/$b/inkrun metal $p gs_${b}_all_$p --n 17 --require-layers 24 --glue fused --attn fused+scores; echo "gs_${b}_all_$p rc=$?"
  done
  cmp -s $R/gs_a1_all_$p.f32 $R/gs_g1_all_$p.f32 && echo "every switch, prompt $p: the split against the head before it BIT FOR BIT, $(stat -f %z $R/gs_g1_all_$p.f32) bytes" || echo "every switch, prompt $p: LOGITS DIFFER"
  python3 -I -c "
import json
a=json.load(open('$R/gs_a1_all_$p.json')); b=json.load(open('$R/gs_g1_all_$p.json'))
print('  tokens equal', a['tokens']==b['tokens'], '| step before %.4f, split %.4f' % (a['step_seconds_median'], b['step_seconds_median']), '| GB end %.2f %.2f' % (a['memory']['end']['used']/1e9, b['memory']['end']['used']/1e9))"
done
$H "coord: g1 default 32" $A/run1.sh $A/bin/g1/inkrun metal 32 gs_g1_def_32 --n 17 --require-layers 24; echo "gs_g1_def_32 rc=$?"
cmp -s $R/before_candle_32.f32 $R/gs_g1_def_32.f32 && echo "default path, prompt 32: BIT FOR BIT the logits of a76a44b" || echo "default path: LOGITS DIFFER"
T=$HOME/llvq-inkling-mini/lot1-b/tables/unembed.q4g64.llvqraw
B=$HOME/llvq-inkling-mini/lot4-coord/bench_decode.sh
$B gatesplit-all-int4head --glue fused --attn fused+scores --head int4 --head-table $T 2>&1 | tail -1
sleep 30
$B gatesplit-all --glue fused --attn fused+scores 2>&1 | tail -1
echo "end $(date '+%T')"
