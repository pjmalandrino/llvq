#!/bin/bash
cd "$(dirname "$0")"
run(){ venv/bin/python fwd.py "$@" > log_$3.txt 2>&1; tail -1 log_$3.txt; }
run 4 512 base 0 0 1
run 4 512 exp085 0.085 0 1
run 4 512 exp085_dense010 0.085 0.010 1
run 4 512 all085 0.085 0.085 1
run 4 512 exp010 0.010 0 1
run 4 512 exp040 0.040 0 1
run 24 512 route 0 0 0 2048
