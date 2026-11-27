#!/bin/bash
# Lot G, job 1: release build of cpurun, then G-i (lockstep) and G-ii (tokens).
# Run under ~/.llvq-guard/heavy.sh, which sets LLVQ_THREADS=6 and RAYON_NUM_THREADS=6.
set -euo pipefail
W=/Users/pjmalandrino/Documents/Pro/workspace/poc/llvq/.claude/worktrees/qwen-cpu
R=$W/docs/mesures/qwen-cpu-groupee-2026-11-27-brut
M=${MODEL:-$HOME/q4b-sealed-2026-09-23/qwen3-4b-sealed.bin}
TAG=${TAG:-4b}
EXP=${EXP:-configs/qwen3-4b-tetra-e4.json}
GRP=${GRP:-configs/qwen3-4b-tetra-e4-grouped.json}
NEW_SHORT=${NEW_SHORT:-256}
cd "$W"
export LLVQ_THREADS=6 RAYON_NUM_THREADS=6 HF_HUB_OFFLINE=1
T=$R/tlim.sh
$R/waitload.sh 1200
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  echo "== build $(date '+%F %T')"
  $T 3600 cargo build --release -p llvq-llm --bin cpurun 2>&1 | tail -5
fi
shasum -a 256 target/release/cpurun | tee "$R/binary-$TAG.sha256"
$R/waitload.sh 300
echo "== G-i lockstep $(date '+%F %T') load $(sysctl -n vm.loadavg)"
LLVQ_CONFIG=$EXP $T 3000 /usr/bin/time -l target/release/cpurun "$M" lockstep "$GRP" 2048 0 > "$R/gi-lockstep-$TAG.log" 2>&1
tail -3 "$R/gi-lockstep-$TAG.log"
for arm in exp grp; do
  cfg=$EXP; [ $arm = grp ] && cfg=$GRP
  for p in short 2048; do
    n=$NEW_SHORT; [ $p = 2048 ] && n=64
    $R/waitload.sh 300
    echo "== G-ii $arm $p $n $(date '+%F %T') load $(sysctl -n vm.loadavg)"
    LLVQ_CONFIG=$cfg LLVQ_RUN_DUMP=$R/gii-$TAG-$arm-$p.json $T 1200 /usr/bin/time -l target/release/cpurun "$M" timed $p $n > "$R/gii-$TAG-$arm-$p.log" 2>&1
    grep -E "loaded form|timed: prompt" "$R/gii-$TAG-$arm-$p.log"
  done
done
echo "== done $(date '+%F %T')"
