#!/bin/bash
# One inkrun arm of prereg inkling-mac-serve-2026-11-26, under the guard. Writes <label>.json, <label>.log,
# <label>.logits.f32 and one line of machine state (<label>.machine) before the run: load, GPU, memory pressure.
# A quiet machine is load1 <= 4 and GPU <= 60 %: waits up to 10 minutes for it, then runs and says "machine busy".
# usage: run_arm.sh <label> <inkrun flags...>      (from the worktree whose target/release/inkrun to run)
set -euo pipefail
label=${1:?label}; shift
O=$(cd "$(dirname "$0")" && pwd)
D=$HOME/llvq-inkling-mini
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
"$HOME/.llvq-guard/heavy.sh" "mac-inkling: inkrun $label" target/release/inkrun \
  --config "$D/textonly/config.json" --checkpoint "$D/textonly/model.safetensors" --layers "$D/layers" \
  --device metal --require-layers 24 --n 32 --out "$O/$label.json" --logits-out "$O/$label.logits.f32" "$@" \
  > "$O/$label.log" 2>&1
python3 -I - "$O/$label.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); m = r["memory"]; g = lambda k: (m.get(k) or {}).get("used", 0) / 1e9
pk = (r.get("peak_probe") or {})
print(f"prompt {r['prompt_tokens']} read in {r['seconds_prefill']:.2f} s; step median {r['step_seconds_median']:.4f} s = "
      f"{1 / r['step_seconds_median']:.2f} tok/s; GB after load {g('after_resident'):.2f}, after the prompt "
      f"{g('after_prefill'):.2f}, end {g('end'):.2f}; peak {pk}; all_equal {r.get('all_equal')}")
PY
