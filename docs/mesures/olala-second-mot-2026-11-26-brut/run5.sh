#!/bin/bash
cd "$(dirname "$0")"
run(){ venv/bin/python fwd_hot.py "$@" > log_$3.txt 2>&1; echo "$3 $(grep 'hot cells' log_$3.txt) $(tail -1 log_$3.txt)"; }
for off in 0 2048; do
  HOT_FRAC=0.2 run 4 512 hot20_o$off 0.085 0 1 $off
  HOT_FRAC=0.5 run 4 512 hot50_o$off 0.085 0 1 $off
  run 4 512 two_o$off 0.0074 0 1 $off
done
echo done > run5.done
