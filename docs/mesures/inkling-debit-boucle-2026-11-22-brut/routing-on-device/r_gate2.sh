#!/bin/bash
# Gate 2 of the routing kept on the device: the real model, --routing device against --routing host, bit for bit.
set -u
A=$HOME/llvq-inkling-mini/lot4-a; B=$A/bin/r1/inkrun; R=$A/runs; H=$HOME/.llvq-guard/heavy.sh
for spec in "s32 32 fused+scores" "s600 600 fused+scores" "c32 32 candle" "f600 600 fused"; do
  set -- $spec; tag=$1; p=$2; attn=$3
  for r in host device; do
    $H "coord: r1 $tag $r" $A/run1.sh $B metal $p r1_${r}_$tag --n 17 --require-layers 24 --glue fused --attn $attn --routing $r --peak-probe on; echo "r1_${r}_$tag rc=$?"
  done
  cmp -s $R/r1_host_$tag.f32 $R/r1_device_$tag.f32 && echo "$tag: logits BIT FOR BIT, $(stat -f %z $R/r1_host_$tag.f32) bytes" || echo "$tag: LOGITS DIFFER"
  python3 -I -c "
import json
a=json.load(open('$R/r1_host_$tag.json')); b=json.load(open('$R/r1_device_$tag.json'))
print('  tokens equal', a['tokens']==b['tokens'], '| step host %.4f device %.4f' % (a['step_seconds_median'], b['step_seconds_median']), '| routing_on_device', a['provenance'].get('routing_on_device'), b['provenance'].get('routing_on_device'), '| GB end %.2f %.2f' % (a['memory']['end']['used']/1e9, b['memory']['end']['used']/1e9))"
done
echo "end $(date '+%T')"
