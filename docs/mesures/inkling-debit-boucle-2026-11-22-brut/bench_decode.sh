#!/bin/bash
# The loop's standard reading of decode speed: 3 rounds in separate processes, 24 resident layers, the 32-token
# prompt, 32 greedy tokens, Metal. Appends one line a round to bench_decode.tsv and prints the median of medians.
# usage: bench_decode.sh <label> [extra inkrun flags...]     (run from the worktree whose target/release/inkrun to read)
set -euo pipefail
label=${1:?label}; shift || true
D=$HOME/llvq-inkling-mini; O=$D/lot4-coord; T=$O/bench_decode.tsv
head=$(git rev-parse --short HEAD)
[ -f "$T" ] || printf 'time\tlabel\thead\tround\tmedian_step_s\tmin_s\tmax_s\ttok_s\tprefill_s\tGB_after_load\tGB_end\tsame_tokens\tload_1min\n' > "$T"
for r in 1 2 3; do
  out=$O/bench_${label}_r$r.json
  "$HOME/.llvq-guard/heavy.sh" "coord: bench $label r$r" target/release/inkrun --config "$D/textonly/config.json" \
    --checkpoint "$D/textonly/model.safetensors" --layers "$D/layers" --tokens "$O/tokens.json" --device metal \
    --n 32 --require-layers 24 --out "$out" "$@" 2>&1 | grep -v '^\[heavy\].*waiting' | tail -1
  python3 -I - "$out" "$label" "$head" "$r" "$T" <<'PY'
import json, sys, time, os
r = json.load(open(sys.argv[1])); s = sorted(r["step_seconds"]); m = r["memory"]
load = os.getloadavg()[0]
row = [time.strftime("%F %T"), sys.argv[2], sys.argv[3], sys.argv[4], f"{r['step_seconds_median']:.4f}", f"{s[0]:.4f}", f"{s[-1]:.4f}",
       f"{r['tokens_per_second_median_step']:.2f}", f"{r['seconds_prefill']:.2f}", f"{m['after_resident']['used']/1e9:.2f}",
       f"{m['end']['used']/1e9:.2f}", str(r.get("all_equal")), f"{load:.1f}"]
open(sys.argv[5], "a").write("\t".join(row) + "\n"); print("  ".join(row[1:]))
PY
  [ $r -lt 3 ] && sleep 45
done
python3 -I - "$T" "$label" <<'PY'
import sys, statistics
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])][1:]
v = [float(x[4]) for x in rows if x[1] == sys.argv[2]][-3:]
print(f"{sys.argv[2]}: median of 3 rounds {statistics.median(v):.4f} s a token [{min(v):.4f}, {max(v):.4f}] = {1/statistics.median(v):.2f} tok/s")
PY
