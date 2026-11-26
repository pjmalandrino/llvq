#!/bin/bash
# Step 0 of lot 3: arm B of lot 2 (A) against arm B with experts batched and the bias on the device (B), 9,000 tokens,
# three rounds A, B, A, B, A, B, no peak probe, 32 tokens decoded as in lot 2. From the worktree root.
set -u
O=docs/mesures/inkling-mac-decode-2026-11-26-brut; R=$O/run_quiet.sh
D=$HOME/llvq-inkling-mini
T=$D/lot1-b/tables/unembed.q4g64.llvqraw
STACK="--glue fused --attn fused+scores --head int4 --head-table $T"
LONG=docs/mesures/inkling-mac-serve-2026-11-26-brut/long.json
ARMB="--prefill-chunk 512 --attn-chunk 128 --kv-window bounded=1024"
COMMON="--n 32 --tokens $LONG --prompt-key ids --prompt-len 9000 $STACK $ARMB"
# an arm whose start is out of bounds (exit 4) is redone; any other failure stops the script
arm() { local l=$1; shift; while :; do $R $O $l $COMMON "$@"; rc=$?; [ $rc = 4 ] || break; echo "redo $l"; done
        [ $rc = 0 ] || { echo "STOP: $l rc=$rc"; exit $rc; }; }
for r in 1 2 3; do
  arm s0-a$r
  arm s0-b$r --moe-rows batch --attn-bias device
done
echo "step0: all done"
