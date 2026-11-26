#!/bin/bash
# One inkrun arm under the guard, started only on a quiet machine: load1 < 6, GPU < 10 %, guard lock free.
# Waits for it up to 30 minutes; if it never comes, writes "machine busy" in <label>.machine and exits 4 without
# running (an arm started out of bounds is redone, not kept). Writes <label>.json, .log, .logits.f32, .machine.
# usage: run_quiet.sh <out dir> <label> <inkrun flags...>      (from the worktree root; --n is the caller's)
set -euo pipefail
O=${1:?out dir}; label=${2:?label}; shift 2
D=$HOME/llvq-inkling-mini
state() {
  printf 'load1 %s gpu %s%% memfree %s\n' "$(sysctl -n vm.loadavg | awk '{print $2}')" \
    "$(ioreg -r -d 1 -w 0 -c IOAccelerator | sed -n 's/.*"Device Utilization %"=\([0-9]*\).*/\1/p' | head -1)" \
    "$(memory_pressure -Q | awk -F': ' '/percentage/{print $2}')"
}
ok=0
for i in $(seq 1 120); do
  s=$(state); l=$(echo "$s" | awk '{print $2}'); g=$(echo "$s" | awk '{print $4}' | tr -d %)
  if [ ! -d "$HOME/.llvq-guard/lock" ] && awk -v l="$l" -v g="$g" 'BEGIN{exit !(l < 6 && g < 10)}'; then ok=1; break; fi
  sleep 15
done
if [ $ok = 0 ]; then echo "$(date '+%F %T') $s machine busy" | tee "$O/$label.machine"; exit 4; fi
echo "$(date '+%F %T') $s" | tee "$O/$label.machine"
"$HOME/.llvq-guard/heavy.sh" "mac-inkling: inkrun $label" target/release/inkrun \
  --config "$D/textonly/config.json" --checkpoint "$D/textonly/model.safetensors" --layers "$D/layers" \
  --device metal --require-layers 24 --out "$O/$label.json" --logits-out "$O/$label.logits.f32" "$@" \
  > "$O/$label.log" 2>&1
python3 -I - "$O/$label.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"prompt {r['prompt_tokens']} read in {r['seconds_prefill']:.2f} s; step median {r['step_seconds_median']:.4f} s = "
      f"{1 / r['step_seconds_median']:.2f} tok/s; all_equal {r.get('all_equal')}")
PY
shasum -a 256 "$O/$label.logits.f32"
