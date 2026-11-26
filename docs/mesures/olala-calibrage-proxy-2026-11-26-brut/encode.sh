#!/bin/bash
cd "$(dirname "$0")/.."
for L in 17 6 28; do
  nice -n 5 ./probe3/target/release/olalaprobe3 cal/L$L.g0.f32 256 768 384 $L 0 cal/L$L.g0
  nice -n 5 ./probe3/target/release/olalaprobe3 cal/L$L.g1.f32 256 384 768 $L 1 cal/L$L.g1
done
echo done > cal/encode.done
