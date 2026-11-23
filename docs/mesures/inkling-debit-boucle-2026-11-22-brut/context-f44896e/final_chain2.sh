#!/bin/bash
# Last readings at the pull-request head after the gate split: decode speed, 600 and 2,000 tokens of context.
set -u
W=$WORKTREE
L=$HOME/llvq-inkling-mini/lot4-coord/night; T=$HOME/llvq-inkling-mini/lot1-b/tables/unembed.q4g64.llvqraw
D=$HOME/llvq-inkling-mini; H=$HOME/.llvq-guard/heavy.sh
cd $W || exit 2
h=$(git rev-parse --short HEAD)
$H "coord: build inkrun $h" nice -n 10 cargo build --release -p llvq-llm --features metal -j 4 --bin inkrun > $L/build_$h.log 2>&1; echo "build exit $?"
B=$D/lot4-coord/bench_decode.sh
$B $h-all-int4head --glue fused --attn fused+scores --head int4 --head-table $T 2>&1 | tail -1
sleep 30
$B $h-all --glue fused --attn fused+scores 2>&1 | tail -1
sleep 30
$B $h-default 2>&1 | tail -1
mkdir -p $L/ctx2
for p in 600 2000; do
  for r in 1 2 3; do
    $H "coord: context $p r$r" target/release/inkrun --config $D/textonly/config.json --checkpoint $D/textonly/model.safetensors --layers $D/layers --tokens $D/lot2-d/wikitext_w0_2048.json --prompt-key ids --prompt-len $p --device metal --n 32 --require-layers 24 --glue fused --attn fused+scores --head int4 --head-table $T --out $L/ctx2/context_${p}_r$r.json > $L/ctx2/context_${p}_r$r.log 2>&1
    python3 -I -c "
import json; j=json.load(open('$L/ctx2/context_${p}_r$r.json')); m=j['memory']
print('context $p r$r: step %.4f s = %.2f tok/s, prompt read in %.2f s, GB after load %.2f end %.2f' % (j['step_seconds_median'], 1/j['step_seconds_median'], j['seconds_prefill'], m['after_resident']['used']/1e9, m['end']['used']/1e9))"
    sleep 20
  done
done
echo "end $(date '+%T')"
