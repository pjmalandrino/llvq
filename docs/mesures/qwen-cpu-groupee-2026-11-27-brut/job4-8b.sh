#!/bin/bash
# Lot G, job 4: G-iv, the same three gates on the 8B, 1 + 2 rounds. Reuses job1 (no build) and job2.
set -euo pipefail
R=/Users/pjmalandrino/Documents/Pro/workspace/poc/llvq/.claude/worktrees/qwen-cpu/docs/mesures/qwen-cpu-groupee-2026-11-27-brut
export MODEL=$HOME/q8b-14b-sealed-2026-09-23/qwen3-8b-sealed-B.bin TAG=8b \
       EXP=configs/qwen3-8b-tetra-e4.json GRP=configs/qwen3-8b-tetra-e4-grouped.json SKIP_BUILD=1 ROUNDS=2
shasum -a 256 "$MODEL"
"$R/job1-identity.sh"
"$R/job2-speed.sh"
