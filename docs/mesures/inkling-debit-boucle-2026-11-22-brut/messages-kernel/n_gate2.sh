#!/bin/bash
# Gate 2 of the message kernel: the real model with every switch on (binary n1) against the candle reference.
set -u
A=$HOME/llvq-inkling-mini/lot4-a; B=$A/bin/n1/inkrun; R=$A/runs; L=$HOME/llvq-inkling-mini/lot4-coord/night
H=$HOME/.llvq-guard/heavy.sh
for p in 32 600; do
  $H "coord: n1 all $p" $A/run1.sh $B metal $p n1_all_$p --n 17 --require-layers 24 --glue fused --attn fused+scores --peak-probe on --routing-steps-out $R/n1_all_$p.routing.json; echo "n1_all_$p rc=$?"
done
$H "coord: ppl n1 from 1" $A/run1.sh $B metal 2048 ppl_n1_from1 --require-layers 24 --glue fused --attn fused+scores --score-from 1 --prompt-len 1025 --nll-out $R/ppl_n1_from1.nll.json; echo "$(date '+%T') ppl_n1_from1 rc=$?"
$H "coord: ppl n1 from 1024" $A/run1.sh $B metal 2048 ppl_n1_from1024 --require-layers 24 --glue fused --attn fused+scores --score-from 1024 --nll-out $R/ppl_n1_from1024.nll.json; echo "$(date '+%T') ppl_n1_from1024 rc=$?"
$H "coord: ppl ref from 1" $A/run1.sh $B metal 2048 ppl_ref_from1 --require-layers 24 --score-from 1 --prompt-len 1025 --nll-out $R/ppl_ref_from1.nll.json; echo "$(date '+%T') ppl_ref_from1 rc=$?"
$H "coord: ppl ref from 1024" $A/run1.sh $B metal 2048 ppl_ref_from1024 --require-layers 24 --score-from 1024 --nll-out $R/ppl_ref_from1024.nll.json; echo "$(date '+%T') ppl_ref_from1024 rc=$?"
echo "end $(date '+%T')"
