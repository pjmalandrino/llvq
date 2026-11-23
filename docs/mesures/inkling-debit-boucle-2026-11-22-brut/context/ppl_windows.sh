#!/bin/bash
# Two more WikiText-2 windows for the whole stack: every switch (binary a1) against no switch, 2,047 positions each.
set -u
A=$HOME/llvq-inkling-mini/lot4-a; B=$A/bin/a1/inkrun; R=$A/runs; H=$HOME/.llvq-guard/heavy.sh
for w in w1 w2; do
  for arm in all ref; do
    if [ $arm = all ]; then F="--glue fused --attn fused+scores"; else F=""; fi
    $H "coord: ppl $w $arm from 1" $A/run1.sh $B metal $w ppl_${w}_${arm}_from1 --require-layers 24 $F --score-from 1 --prompt-len 1025 --nll-out $R/ppl_${w}_${arm}_from1.nll.json; echo "$(date '+%T') ppl_${w}_${arm}_from1 rc=$?"
    $H "coord: ppl $w $arm from 1024" $A/run1.sh $B metal $w ppl_${w}_${arm}_from1024 --require-layers 24 $F --score-from 1024 --nll-out $R/ppl_${w}_${arm}_from1024.nll.json; echo "$(date '+%T') ppl_${w}_${arm}_from1024 rc=$?"
  done
done
echo "end $(date '+%T')"
