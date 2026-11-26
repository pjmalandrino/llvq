#!/bin/bash
cd "$(dirname "$0")"
while pgrep -f run2.sh >/dev/null; do sleep 10; done
venv/bin/python fwd2.py 24 512 route2 0 0 0 2048 > log_route2.txt 2>&1
venv/bin/python fwd2.py 24 512 route3 0 0 0 14336 > log_route3.txt 2>&1
echo done > run3.done
