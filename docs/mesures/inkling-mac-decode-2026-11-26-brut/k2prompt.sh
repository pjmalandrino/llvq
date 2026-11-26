#!/bin/bash
# Does K2 move the read of a 9,000-token prompt? Arm B of lot 2, alternated: c50 = CANDLE_METAL_COMPUTE_PER_BUFFER=50
# (candle's default, the cadence before this lot), c30 = the new default. From the worktree root.
set -u
O=docs/mesures/inkling-mac-decode-2026-11-26-brut; R=$O/run_quiet.sh
D=$HOME/llvq-inkling-mini
T=$D/lot1-b/tables/unembed.q4g64.llvqraw
STACK="--glue fused --attn fused+scores --head int4 --head-table $T"
COMMON="--n 32 --tokens docs/mesures/inkling-mac-serve-2026-11-26-brut/long.json --prompt-key ids --prompt-len 9000 $STACK --prefill-chunk 512 --attn-chunk 128 --kv-window bounded=1024"
arm() { local l=$1; shift; while :; do env "$@" $R $O $l $COMMON; rc=$?; [ $rc = 4 ] || break; done; [ $rc = 0 ] || { echo "STOP $l"; exit 1; }; }
for r in 1 2; do
  arm kp-c50-$r CANDLE_METAL_COMPUTE_PER_BUFFER=50
  arm kp-c30-$r K2_ARM=c30
done
echo "k2prompt: all done"
