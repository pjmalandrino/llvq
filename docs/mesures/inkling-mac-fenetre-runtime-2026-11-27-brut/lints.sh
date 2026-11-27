#!/bin/bash
# Lints of lot H, from the worktree root, under the guard. Metal build, CPU build, the Linux CUDA half type-checked.
set -u
B=docs/mesures/inkling-mac-fenetre-runtime-2026-11-27-brut
T="perl -e alarm(shift);exec(@ARGV)||die 1200"
echo "lints: $(date '+%F %T') at $(git rev-parse HEAD); tree: $(git status --short llvq-llm llvq-runtime configs | tr '\n' ' ')"
perl -e 'alarm shift; exec @ARGV or die' 1200 cargo clippy -p llvq-runtime --features metal --all-targets > $B/clippy-metal.log 2>&1; echo "clippy metal rc=$?"
perl -e 'alarm shift; exec @ARGV or die' 1200 cargo clippy -p llvq-runtime --all-targets > $B/clippy-cpu.log 2>&1; echo "clippy cpu rc=$?"
perl -e 'alarm shift; exec @ARGV or die' 1500 ops/check-linux-cuda.sh clippy -p llvq-runtime --features cuda --all-targets > $B/linux-check.log 2>&1; echo "linux check rc=$?"
for f in clippy-metal clippy-cpu linux-check; do echo "$f: $(grep -c '^warning' $B/$f.log) warning lines, $(grep -c '^error' $B/$f.log) error lines"; done
