#!/bin/bash
# Gate 2 of the attention wiring: 32- and 600-token prompts, 17 rows, both glues; --attn candle (bit for bit the
# untouched head's) and --attn fused (class B), with logits and routing dumps.
set -u
A=$HOME/llvq-inkling-mini/lot4-a; B=$A/bin/w1/inkrun; R=$A/runs
for g in candle fused; do for p in 32 600; do for a in candle fused; do
  $A/run1.sh $B metal $p w1_${a}_${g}_$p --n 17 --require-layers 24 --glue $g --attn $a --peak-probe on --routing-steps-out $R/w1_${a}_${g}_$p.routing.json
  echo "w1_${a}_${g}_$p rc=$?"
done; done; done
