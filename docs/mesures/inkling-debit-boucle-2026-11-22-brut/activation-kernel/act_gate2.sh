#!/bin/bash
# Gate 2 of the activation kernel: the real model, bit for bit against candle's operations and against the untouched head.
set -u
A=$HOME/llvq-inkling-mini/lot4-a; B=$A/bin/a1/inkrun; R=$A/runs; H=$HOME/.llvq-guard/heavy.sh
for p in 32 600; do
  $H "coord: a1 default $p" $A/run1.sh $B metal $p a1_def_$p --n 17 --require-layers 24 --peak-probe on; echo "a1_def_$p rc=$?"
  cmp -s $R/before_candle_$p.f32 $R/a1_def_$p.f32 && echo "default path, prompt $p: logits BIT FOR BIT those of the head before the night's work (a76a44b)" || echo "default path, prompt $p: LOGITS DIFFER from a76a44b"
  for m in fused candle; do
    $H "coord: a1 all $m $p" $A/run1.sh $B metal $p a1_all_${m}_$p --n 17 --require-layers 24 --glue fused --attn fused+scores --moe-act $m --peak-probe on; echo "a1_all_${m}_$p rc=$?"
  done
  cmp -s $R/a1_all_fused_$p.f32 $R/a1_all_candle_$p.f32 && echo "every switch, prompt $p: --moe-act fused against candle BIT FOR BIT" || echo "every switch, prompt $p: LOGITS DIFFER"
  python3 -I -c "
import json
a=json.load(open('$R/a1_all_candle_$p.json')); b=json.load(open('$R/a1_all_fused_$p.json')); d=json.load(open('$R/a1_def_$p.json'))
print('  step: act candle %.4f, act fused %.4f, default path %.4f | moe_act' % (a['step_seconds_median'], b['step_seconds_median'], d['step_seconds_median']), a['provenance'].get('moe_act'), b['provenance'].get('moe_act'), '| GB end %.2f %.2f' % (a['memory']['end']['used']/1e9, b['memory']['end']['used']/1e9))"
done
echo "end $(date '+%T')"
