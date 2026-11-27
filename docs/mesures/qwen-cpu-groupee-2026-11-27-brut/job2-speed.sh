#!/bin/bash
# Lot G, job 2: G-iii, arms alternated (expanded, grouped), round 0 discarded,
# rounds 1-3 timed. One process per arm and prompt, each under /usr/bin/time -l.
# Run under ~/.llvq-guard/heavy.sh, which sets LLVQ_THREADS=6 and RAYON_NUM_THREADS=6.
set -euo pipefail
W=/Users/pjmalandrino/Documents/Pro/workspace/poc/llvq/.claude/worktrees/qwen-cpu
R=$W/docs/mesures/qwen-cpu-groupee-2026-11-27-brut
M=${MODEL:-$HOME/q4b-sealed-2026-09-23/qwen3-4b-sealed.bin}
TAG=${TAG:-4b}
EXP=${EXP:-configs/qwen3-4b-tetra-e4.json}
GRP=${GRP:-configs/qwen3-4b-tetra-e4-grouped.json}
ROUNDS=${ROUNDS:-3}
NEW_SHORT=${NEW_SHORT:-256}
cd "$W"
export LLVQ_THREADS=6 RAYON_NUM_THREADS=6 HF_HUB_OFFLINE=1
T=$R/tlim.sh
$R/waitload.sh 600
shasum -a 256 target/release/cpurun
for r in $(seq 0 "$ROUNDS"); do
  for p in short 2048; do
    n=$NEW_SHORT; [ $p = 2048 ] && n=64
    for arm in exp grp; do
      cfg=$EXP; [ $arm = grp ] && cfg=$GRP
      f=$R/giii-$TAG-r$r-$arm-$p
      echo "== G-iii round $r $arm $p $n $(date '+%F %T') load $(sysctl -n vm.loadavg) | $(memory_pressure -Q | tail -1)" | tee "$f.log"
      LLVQ_CONFIG=$cfg LLVQ_RUN_DUMP=$f.json $T 1200 /usr/bin/time -l target/release/cpurun "$M" timed $p $n >> "$f.log" 2>&1
      grep -E "timed: prompt|maximum resident|peak memory footprint" "$f.log"
    done
  done
done
echo "== done $(date '+%F %T')"
