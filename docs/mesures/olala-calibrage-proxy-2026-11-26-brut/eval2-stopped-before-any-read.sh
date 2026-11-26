#!/bin/bash
cd "$(dirname "$0")/.."
while [ ! -f cal/encode.done ] || [ ! -f cal/eval1.done ]; do sleep 20; done
run(){ venv/bin/python cal/fwd3.py 2>&1 | grep -v -i "warn\|mamba3:\|moe:\|attention impl" | tail -1; }
export THREADS=4
for OFF in 0 2048 4096 6144; do export OFF
  ARM=t1 TAG=t1 run
  ARM=t2 TAG=t2 run
done
venv/bin/python cal/collect.py 2>&1 | grep -v -i warn > cal/collect.out
echo done > cal/all.done
