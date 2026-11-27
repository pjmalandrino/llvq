#!/bin/bash
# A-iii and A-iv: the served expanded config against the new Metal config, one process a size and a length.
set -u
cd /Users/pjmalandrino/Documents/Pro/workspace/poc/llvq/.claude/worktrees/mac-inkling
B=docs/mesures/qwen-metal-servi-2026-11-26-brut; M=$B/work/metalrun-1e35fa1b; D=~/q8b-14b-sealed-2026-09-23
run() { # tag sealed size n_new timed prompt
  local tag=$1 S=$2 t=$3 n=$4 timed=$5 p=$6
  while :; do l=$(sysctl -n vm.loadavg | awk '{print $2}'); awk -v l="$l" 'BEGIN{exit !(l<6)}' && break; echo "load $l, waiting" >&2; sleep 60; done
  { echo "start $(date '+%F %T') loadavg $(sysctl -n vm.loadavg) | $(memory_pressure -Q | tail -1)"
    ~/.llvq-guard/heavy.sh "lotA: A-iv $tag" env -u LLVQ_THREADS $M time $S configs/qwen3-$t-tetra-e4.json \
      cfg:configs/qwen3-$t-tetra-e4.json,cfg:configs/qwen3-$t-tetra-e4-metal.json $n $timed $p 1 2>&1
    echo "exit $? end $(date '+%F %T') loadavg $(sysctl -n vm.loadavg)"; } > $B/a-iv-$tag.txt
}
run 4b-short ~/q4b-sealed-2026-09-23/qwen3-4b-sealed.bin 4b 256 5 0
run 4b-long ~/q4b-sealed-2026-09-23/qwen3-4b-sealed.bin 4b 64 5 4096
run 8b-short $D/qwen3-8b-sealed-B.bin 8b 256 3 0
run 8b-long $D/qwen3-8b-sealed-B.bin 8b 64 3 2048
run 14b-short $D/qwen3-14b-sealed.bin 14b 256 3 0
run 14b-long $D/qwen3-14b-sealed.bin 14b 64 3 2048
echo all done
