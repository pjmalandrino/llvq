#!/bin/bash
# Step 1 screening: one decode run per value of CANDLE_METAL_COMPUTE_PER_BUFFER, the default (unset) first and last.
set -u
O=docs/mesures/inkling-mac-decode-2026-11-26-brut
$O/decode_arm.sh k2s-default || exit 1
for v in 5 10 20 30 40 75 100 200 500 1000; do
  $O/decode_arm.sh k2s-$v ENV=CANDLE_METAL_COMPUTE_PER_BUFFER=$v || exit 1
done
$O/decode_arm.sh k2s-default2 || exit 1
echo "k2 screen: all done"
