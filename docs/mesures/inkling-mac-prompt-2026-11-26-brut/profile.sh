#!/bin/bash
# Step 1 of prereg inkling-mac-prompt-2026-11-26: the read of arm B by part, 2,000 then 9,000 tokens. From the worktree root.
set -u
B=docs/mesures/inkling-mac-prompt-2026-11-26-brut; R=$B/run_arm.sh
D=$HOME/llvq-inkling-mini
T=$D/lot1-b/tables/unembed.q4g64.llvqraw
STACK="--glue fused --attn fused+scores --head int4 --head-table $T"
W0=$D/lot2-d/wikitext_w0_2048.json
LONG=docs/mesures/inkling-mac-serve-2026-11-26-brut/long.json
ARMB="--prefill-chunk 512 --attn-chunk 128 --kv-window bounded=1024"
$R prof2000 --tokens $W0 --prompt-key ids --prompt-len 2000 $STACK $ARMB --prefill-phases on; sleep 45
$R prof9000 --tokens $LONG --prompt-key ids --prompt-len 9000 $STACK $ARMB --prefill-phases on
echo "profile: all done"
