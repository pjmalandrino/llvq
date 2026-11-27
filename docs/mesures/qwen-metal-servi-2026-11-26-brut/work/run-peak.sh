#!/bin/bash
# Peak memory footprint at the long prompt: one process a config, one generation (ECARTS A-E3).
set -u
cd /Users/pjmalandrino/Documents/Pro/workspace/poc/llvq/.claude/worktrees/mac-inkling
B=docs/mesures/qwen-metal-servi-2026-11-26-brut; M=$B/work/metalrun-1e35fa1b; D=~/q8b-14b-sealed-2026-09-23
for t in 4b 8b 14b; do
  case $t in 4b) S=~/q4b-sealed-2026-09-23/qwen3-4b-sealed.bin; p=4096;; 8b) S=$D/qwen3-8b-sealed-B.bin; p=2048;; 14b) S=$D/qwen3-14b-sealed.bin; p=2048;; esac
  for c in qwen3-$t-tetra-e4.json qwen3-$t-tetra-e4-metal.json; do
    { echo "start $(date '+%F %T') loadavg $(sysctl -n vm.loadavg) | $(memory_pressure -Q | tail -1)"
      ~/.llvq-guard/heavy.sh "lotA: peak $c" env -u LLVQ_THREADS /usr/bin/time -l $M time $S configs/$c cfg:configs/$c 64 1 $p 0 2>&1
      echo "exit $?"; } > $B/peak-$c.txt
  done
done
echo done
