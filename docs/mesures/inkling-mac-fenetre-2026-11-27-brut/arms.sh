#!/bin/bash
# Arms of prereg inkling-fenetre-2026-11-27, 45 s apart, in the order given. From the worktree root.
# Copied from ../inkling-mac-prompt-2026-11-26-brut/arms.sh: the same tokens, stack and arm B.
# usage: arms.sh <len> <label>=<extra flags> ...   (each label one run of arm B plus its flags)
set -u
B=docs/mesures/inkling-mac-fenetre-2026-11-27-brut; R=$B/run_arm.sh
D=$HOME/llvq-inkling-mini
T=$D/lot1-b/tables/unembed.q4g64.llvqraw
STACK="--glue fused --attn fused+scores --head int4 --head-table $T"
LONG=docs/mesures/inkling-mac-serve-2026-11-26-brut/long.json
ARMB="--prefill-chunk 512 --attn-chunk 128 --kv-window bounded=1024"
len=$1; shift
if [ "$len" = 2000 ]; then TOK="--tokens $D/lot2-d/wikitext_w0_2048.json"; else TOK="--tokens $LONG"; fi
first=1
for spec in "$@"; do
  label=${spec%%=*}; extra=${spec#*=}
  if [ "$(date +%H%M)" -ge 0645 ] && [ "$(date +%H%M)" -lt 1200 ]; then echo "arms: 06:45 passed, $label not started"; exit 3; fi
  [ $first = 1 ] || sleep 45; first=0
  $R "$label" $TOK --prompt-key ids --prompt-len "$len" $STACK $ARMB $extra
  shasum -a 256 "$B/$label.logits.f32"
done
echo "arms: all done"
