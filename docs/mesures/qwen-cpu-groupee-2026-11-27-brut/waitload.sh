#!/bin/bash
# usage: waitload.sh <max_seconds>: waits while the 1-minute load is above 6, then prints load and memory.
end=$(( $(date +%s) + ${1:-1200} ))
while :; do
  l=$(sysctl -n vm.loadavg | awk '{print $2}')
  if awk "BEGIN{exit !($l <= 6)}" || [ "$(date +%s)" -ge "$end" ]; then break; fi
  echo "[waitload] $(date '+%T') load $l > 6, waiting"; sleep 30
done
echo "[waitload] $(date '+%F %T') load $(sysctl -n vm.loadavg) | $(memory_pressure -Q | tail -1)"
