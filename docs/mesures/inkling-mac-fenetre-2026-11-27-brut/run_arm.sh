#!/bin/bash
# One inkrun arm of prereg inkling-fenetre-2026-11-27, under the guard. Copied from
# ../inkling-mac-prompt-2026-11-26-brut/run_arm.sh; changes: the binary of this lot's own target directory
# (target/lotd/inkrun, a copy of the release build), a hard limit of LIMIT seconds (perl alarm), stdin closed, /usr/bin/time -l around inkrun.
# Writes <label>.json, .log (inkrun's stderr, then time -l), .logits.f32 and .machine (state before the run).
# A quiet machine is load1 <= 4 and GPU <= 60 %: waits up to 10 minutes for it, then runs and says "machine busy".
# usage: run_arm.sh <label> <inkrun flags...>      (from the worktree root)
set -euo pipefail
label=${1:?label}; shift
O=$(cd "$(dirname "$0")" && pwd)
D=$HOME/llvq-inkling-mini
BIN=${BIN:-target/lotd/inkrun}
LIMIT=${LIMIT:-600}
state() {
  printf 'load1 %s gpu %s%% memfree %s\n' "$(sysctl -n vm.loadavg | awk '{print $2}')" \
    "$(ioreg -r -d 1 -w 0 -c IOAccelerator | sed -n 's/.*"Device Utilization %"=\([0-9]*\).*/\1/p' | head -1)" \
    "$(memory_pressure -Q | awk -F': ' '/percentage/{print $2}')"
}
busy=""
for i in $(seq 1 21); do
  s=$(state); l=$(echo "$s" | awk '{print $2}'); g=$(echo "$s" | awk '{print $4}' | tr -d %)
  if awk -v l="$l" -v g="$g" 'BEGIN{exit !(l <= 4 && g <= 60)}'; then break; fi
  [ "$i" = 21 ] && { busy="machine busy"; break; }
  sleep 30
done
echo "$(date '+%F %T') $s ${busy}" | tee "$O/$label.machine"
# Inside a job that already holds the guard's lock (all.sh), INLOCK=1 runs inkrun directly.
G=("$HOME/.llvq-guard/heavy.sh" "lotD: inkrun $label"); [ "${INLOCK:-0}" = 1 ] && G=(nice -n 15)
"${G[@]}" /usr/bin/time -l perl -e 'alarm shift; exec @ARGV or die' "$LIMIT" "$BIN" \
  --config "$D/textonly/config.json" --checkpoint "$D/textonly/model.safetensors" --layers "$D/layers" \
  --device metal --require-layers 24 --n 32 --out "$O/$label.json" --logits-out "$O/$label.logits.f32" "$@" \
  < /dev/null > "$O/$label.log" 2>&1
python3 -I - "$O/$label.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); m = r["memory"]; g = lambda k: (m.get(k) or {}).get("used", 0) / 1e9
print(f"prompt {r['prompt_tokens']} read in {r['seconds_prefill']:.2f} s; step median {r['step_seconds_median']:.4f} s = "
      f"{1 / r['step_seconds_median']:.2f} tok/s; GB after load {g('after_resident'):.2f}, after the prompt "
      f"{g('after_prefill'):.2f}, end {g('end'):.2f}; attn_window {r['provenance'].get('attn_window')}")
PY
