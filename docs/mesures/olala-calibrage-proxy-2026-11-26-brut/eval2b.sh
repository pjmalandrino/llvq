#!/bin/bash
# Replaces eval2.sh, stopped before its first read: same arms, on the thread count of the base
# (3), plus two null controls that change no weight.
cd "$(dirname "$0")/.."
run(){ venv/bin/python cal/fwd3.py 2>&1 | grep -v -i "warn\|mamba3:\|moe:\|attention impl" | tail -1; }
# Null controls first, they need no encode: the bf16 model against its own saved base.
OFF=0 LAYERS=99 ARM=noise R=0 TAG=null3 THREADS=3 run
OFF=0 LAYERS=99 ARM=noise R=0 TAG=null4 THREADS=4 run
while [ ! -f cal/encode.done ] || [ ! -f cal/eval1.done ]; do sleep 20; done
export THREADS=3
for OFF in 0 2048 4096 6144; do export OFF
  ARM=t1 TAG=t1 run
  ARM=t2 TAG=t2 run
done
venv/bin/python cal/collect.py 2>&1 | grep -v -i warn > cal/collect.out
echo done > cal/all.done
