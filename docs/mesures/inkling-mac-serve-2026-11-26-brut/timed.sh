#!/bin/bash
# The timed readings of prereg inkling-mac-serve-2026-11-26, in order, 45 s apart. From the worktree root.
set -u
B=docs/mesures/inkling-mac-serve-2026-11-26-brut; R=$B/run_arm.sh
D=$HOME/llvq-inkling-mini
T=$D/lot1-b/tables/unembed.q4g64.llvqraw
STACK="--glue fused --attn fused+scores --head int4 --head-table $T"
W0=$D/lot2-d/wikitext_w0_2048.json
LONG=$B/long.json
ARMB="--prefill-chunk 512 --attn-chunk 128 --kv-window bounded=1024"
for r in 1 2 3; do $R decode32-r$r --tokens $D/lot4-coord/tokens.json $STACK; sleep 45; done
$R a2000 --tokens $W0 --prompt-key ids --prompt-len 2000 $STACK; sleep 45
$R b2000 --tokens $W0 --prompt-key ids --prompt-len 2000 $STACK $ARMB; sleep 45
$R a2000-peak --tokens $W0 --prompt-key ids --prompt-len 2000 $STACK --peak-probe on; sleep 45
$R b2000-peak --tokens $W0 --prompt-key ids --prompt-len 2000 $STACK $ARMB --peak-probe on; sleep 45
# Stop rule: arm B at 2,000 tokens above 20 GB, peak or after the prompt.
python3 -I - $B/b2000-peak.json $B/b2000.json <<'PY' || { echo "STOP: arm B at 2,000 tokens above 20 GB"; exit 3; }
import json, sys
pk = (json.load(open(sys.argv[1])).get("peak_probe") or {}).get("whole_run", 0)
ap = json.load(open(sys.argv[2]))["memory"]["after_prefill"]["used"]
print(f"arm B 2,000: peak {pk / 1e9:.2f} GB, after the prompt {ap / 1e9:.2f} GB")
sys.exit(0 if max(pk, ap) <= 20e9 else 1)
PY
$R b9000 --tokens $LONG --prompt-key ids --prompt-len 9000 $STACK $ARMB; sleep 45
$R b9000-peak --tokens $LONG --prompt-key ids --prompt-len 9000 $STACK $ARMB --peak-probe on
echo "timed: all done"
