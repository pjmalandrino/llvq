#!/bin/bash
A=$HOME/llvq-inkling-mini/lot4-a; R=$A/runs; L=$HOME/llvq-inkling-mini/lot4-coord/night
for g in candle fused; do for p in 32 600; do
  echo "=== glue $g, prompt $p"
  cmp -s $R/before_${g}_$p.f32 $R/w1_candle_${g}_$p.f32 && echo "attn candle: logits bit for bit the untouched head's" || echo "attn candle: LOGITS DIFFER from the untouched head"
  python3 -I $L/cmp3.py before_${g}_$p w1_fused_${g}_$p $R | cut -c1-400
done; done
python3 -I - $R <<'PY'
import json, sys, glob, os
R = sys.argv[1]
for n in sorted(glob.glob(R + '/w1_*.json') + glob.glob(R + '/before_*.json')):
    if n.endswith('.routing.json'): continue
    j = json.load(open(n)); m = j.get('memory', {})
    g = lambda k: m.get(k, {}).get('used', 0) / 1e9
    pk = j.get('peak_probe') or {}
    print(f"{os.path.basename(n)[:-5]:24s} step {j.get('step_seconds_median', 0):.4f} s  prefill {j.get('seconds_prefill', 0):6.2f} s  GB dense {g('after_dense'):.2f} experts {g('after_resident'):.2f} end {g('end'):.2f} peak prefill {pk.get('prefill', 0)/1e9:.2f} run {pk.get('whole_run', 0)/1e9:.2f}")
PY
