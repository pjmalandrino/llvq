#!/bin/bash
# Night copy of ~/.llvq-guard/sentinel.sh for the Inkling gate of 2026-11-21. Same thresholds and same action, with
# one change: the swap criterion is the GROWTH over the swap in use when this script starts, because macOS keeps
# several GB of stale swap after a pressure event and the original's absolute 2 GB threshold then fires at once.
# Critical (2 samples in a row): pauses the heavy job's process group with SIGSTOP (reversible: kill -CONT -- -pgid)
#   and exits. Warning (sustained): exits without touching anything. Either exit wakes the orchestrator.
# It reads the guard's lock and writes only its own log.
G="$HOME/.llvq-guard"; LOCK="$G/lock"; LOG="$HOME/llvq-inkling-mini/logs/sentinel-night.log"
MAX_SECONDS=${MAX_SECONDS:-5400}
LOAD_WARN=13; MEMFREE_WARN=25; GPU_WARN=${GPU_WARN:-101}   # GPU warning off by default: the capture uses the GPU on purpose
LOAD_CRIT=15; MEMFREE_CRIT=12; SWAP_GROWTH_CRIT_MB=2048
swap_now() { sysctl -n vm.swapusage | sed 's/.*used = \([0-9.]*\)M.*/\1/' | awk '{print int($1)}'; }
base=$(swap_now)
cl=0; cm=0; cg=0; kl=0; km=0; ks=0; t0=$(date +%s)
pgid_of() { sed -n 's/.*pgid=\([0-9]*\).*/\1/p' "$LOCK/owner" 2>/dev/null; }
report() {
  echo "SENTINEL $1 at $(date '+%F %T'): $2"
  echo "holder: $(cat "$LOCK/owner" 2>/dev/null || echo none)"
  ps -Ao pid,pgid,pcpu,rss,comm -r | head -6
  echo "$(date '+%F %T') EXIT $1: $2" >> "$LOG"
}
echo "$(date '+%F %T') START base_swap=${base}M" >> "$LOG"
while :; do
  load=$(sysctl -n vm.loadavg | awk '{print $2}')
  mem=$(memory_pressure -Q 2>/dev/null | awk -F': ' '/percentage/{gsub("%","",$2); print $2+0}')
  swap=$(swap_now)
  # the baseline follows the swap down as it drains, never up
  [ "$swap" -lt "$base" ] && base=$swap
  grow=$((swap - base))
  gpu=$(ioreg -r -d 1 -w 0 -c IOAccelerator 2>/dev/null | sed -n 's/.*"Device Utilization %"=\([0-9]*\).*/\1/p' | head -1)
  gpu=${gpu:-0}
  holder=$(cut -d'|' -f1 "$LOCK/owner" 2>/dev/null)
  echo "$(date '+%T') load=$load memfree=$mem% swap=${swap}M (+${grow}M) gpu=$gpu% holder=${holder:-none}" >> "$LOG"
  awk -v a="$load" -v b="$LOAD_WARN" 'BEGIN{exit !(a>b)}' && cl=$((cl+1)) || cl=0
  awk -v a="$load" -v b="$LOAD_CRIT" 'BEGIN{exit !(a>b)}' && kl=$((kl+1)) || kl=0
  [ "$mem" -lt "$MEMFREE_WARN" ] && cm=$((cm+1)) || cm=0
  [ "$mem" -lt "$MEMFREE_CRIT" ] && km=$((km+1)) || km=0
  [ "$grow" -gt "$SWAP_GROWTH_CRIT_MB" ] && ks=$((ks+1)) || ks=0
  [ "$gpu" -ge "$GPU_WARN" ] && cg=$((cg+1)) || cg=0
  if [ $kl -ge 2 ] || [ $km -ge 2 ] || [ $ks -ge 2 ]; then
    p=$(pgid_of); [ -n "$p" ] && kill -STOP -- -"$p" 2>/dev/null && echo "$(date '+%F %T') PAUSED pgid $p" >> "$LOG"
    report CRITICAL "load=$load memfree=$mem% swap=${swap}M (+${grow}M over base) gpu=$gpu%; paused pgid ${p:-none}"; exit 0
  fi
  if [ $cl -ge 4 ] || [ $cm -ge 4 ] || [ $cg -ge 8 ]; then
    report WARNING "load=$load memfree=$mem% swap=${swap}M gpu=$gpu% (load ${cl}x, mem ${cm}x, gpu ${cg}x)"; exit 0
  fi
  [ $(( $(date +%s) - t0 )) -ge "$MAX_SECONDS" ] && { report HEARTBEAT "no alert in $MAX_SECONDS s"; tail -3 "$G/jobs.log" 2>/dev/null; exit 0; }
  sleep 15
done
