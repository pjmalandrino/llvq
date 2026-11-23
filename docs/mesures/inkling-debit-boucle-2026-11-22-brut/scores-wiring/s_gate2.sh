#!/bin/bash
# Gate 2 of the score-kernel wiring: --attn fused+scores on the real model, tokens, logits, routing, perplexity.
set -u
A=$HOME/llvq-inkling-mini/lot4-a; B=$A/bin/s1/inkrun; R=$A/runs; L=$HOME/llvq-inkling-mini/lot4-coord/night
H=$HOME/.llvq-guard/heavy.sh
$H "coord: s1 candle check" $A/run1.sh $B metal 32 s1_candle_fused_32 --n 17 --require-layers 24 --glue fused --attn candle --peak-probe on --routing-steps-out $R/s1_candle_fused_32.routing.json; echo "s1_candle_fused_32 rc=$?"
for g in candle fused; do for p in 32 600; do
  $H "coord: s1 scores $g $p" $A/run1.sh $B metal $p s1_scores_${g}_$p --n 17 --require-layers 24 --glue $g --attn fused+scores --peak-probe on --routing-steps-out $R/s1_scores_${g}_$p.routing.json; echo "s1_scores_${g}_$p rc=$?"
done; done
$H "coord: ppl scores from 1" $A/run1.sh $B metal 2048 ppl_scores_from1 --require-layers 24 --glue fused --attn fused+scores --score-from 1 --prompt-len 1025 --nll-out $R/ppl_scores_from1.nll.json; echo "$(date '+%T') ppl_scores_from1 rc=$?"
$H "coord: ppl scores from 1024" $A/run1.sh $B metal 2048 ppl_scores_from1024 --require-layers 24 --glue fused --attn fused+scores --score-from 1024 --nll-out $R/ppl_scores_from1024.nll.json; echo "$(date '+%T') ppl_scores_from1024 rc=$?"
echo "end $(date '+%T')"
