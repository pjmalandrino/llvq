#!/bin/bash
cd "$(dirname "$0")/.."
run(){ nice -n 2 venv/bin/python cal/fwd3.py 2>&1 | grep -v -i "warn\|mamba3:\|moe:\|attention impl" | tail -1; }
export THREADS=3
for OFF in 0 2048 4096 6144; do export OFF; ARM=base run; done
for OFF in 0 2048 4096 6144; do export OFF
  ARM=q4 TAG=q4 run
  ARM=noise R=0.085 TAG=n1 run
  ARM=noise R=0.0074 TAG=n2 run
  ARM=noise R=0.0100 TAG=n4 run
done
echo done > cal/eval1.done
