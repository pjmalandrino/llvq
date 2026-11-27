#!/bin/bash
# The runs of prereg inkling-fenetre-ppl-2026-11-27, in one job under the guard's lock, from the worktree root:
#   ~/.llvq-guard/heavy.sh "lotD: ..." docs/mesures/inkling-mac-fenetre-ppl-2026-11-27-brut/all.sh < /dev/null
# Part D's binary (target/lotd/inkrun), Metal, part D's reference flags (arm B of inkling-mac-prompt-2026-11-26), and
# --score-prompt on --nll-out. Two arms that differ by --attn-window alone, full then narrow, text by text: the six
# single windows, then the two joined texts. 45 s between runs, a hard limit per run, /usr/bin/time -l around each.
set -u
R=docs/mesures/inkling-mac-fenetre-ppl-2026-11-27-brut
D=$HOME/llvq-inkling-mini
BIN=target/lotd/inkrun
LIMIT=${LIMIT:-900}
T=$D/lot1-b/tables/unembed.q4g64.llvqraw
STACK=(--glue fused --attn fused+scores --head int4 --head-table "$T")
ARMB=(--prefill-chunk 512 --attn-chunk 128 --kv-window bounded=1024)
state() {
  printf 'load1 %s gpu %s%% memfree %s\n' "$(sysctl -n vm.loadavg | awk '{print $2}')" \
    "$(ioreg -r -d 1 -w 0 -c IOAccelerator | sed -n 's/.*"Device Utilization %"=\([0-9]*\).*/\1/p' | head -1)" \
    "$(memory_pressure -Q | awk -F': ' '/percentage/{print $2}')"
}
echo "all: start $(date '+%F %T') at $(git rev-parse HEAD); binary $(shasum -a 256 $BIN | cut -c1-64)"
echo "all: tree $(git status --short llvq-llm llvq-runtime configs | tr '\n' ' ')"
first=1
one() { # <text label> <tokens file> <arm>
  local text=$1 file=$2 arm=$3 label=$1-$3
  if [ "$(date +%H%M)" -ge 0615 ] && [ "$(date +%H%M)" -lt 1200 ]; then echo "all: 06:15 passed, $label not started"; exit 3; fi
  [ $first = 1 ] || sleep 45; first=0
  echo "$(date '+%F %T') $(state)" | tee "$R/$label.machine"
  nice -n 15 /usr/bin/time -l perl -e 'alarm shift; exec @ARGV or die' "$LIMIT" "$BIN" \
    --config "$D/textonly/config.json" --checkpoint "$D/textonly/model.safetensors" --layers "$D/layers" \
    --device metal --require-layers 24 --tokens "$file" --prompt-key ids "${STACK[@]}" "${ARMB[@]}" \
    --attn-window "$arm" --score-prompt on --nll-out "$R/$label.nll.json" --out "$R/$label.json" \
    < /dev/null > "$R/$label.log" 2>&1
  echo "all: $label rc=$? $(grep 'inkrun: scored' "$R/$label.log")"
}
W="$D/lot2-d/wikitext_w0_2048.json $D/lot4-x/wikitext_w1_2048.json $D/lot4-x/wikitext_w2_2048.json $R/wikitext_w3_2048.json $R/wikitext_w4_2048.json $R/wikitext_w5_2048.json"
w=0
for f in $W; do
  one w$w "$f" full; one w$w "$f" narrow; w=$((w + 1))
done
for j in w012 w345; do
  one j$j "$R/joined_${j}_6144.json" full; one j$j "$R/joined_${j}_6144.json" narrow
done
echo "all: done $(date '+%F %T')"
