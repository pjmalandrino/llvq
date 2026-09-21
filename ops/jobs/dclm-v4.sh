#!/usr/bin/env bash
# The calibration volume at four times, preregistered:
#   proofs/preregistration-dclm-v4-2026-09-18.md
#   sha256 1c8a94a3ea452b25ec8007d524f0a2015f9d66e47d4e9a87546c7408b6d0baaf
#
# Corpus held at DCLM-edu, volume 131,072 -> 524,288 tokens, everything else
# identical: tetra1, rotation seed 0x110feed, nogs, h_shrink 1, gain_scale 1,
# LLVQ_INT4_TYPES=v_proj, 2.2044 kernel b/weight.
#
# Read against dclm x1, which scored 57.95 on this plan. The mechanism is
# named in the prereg: volume can only help down_proj, the one activation
# short of samples at 13.5 per dimension against 51 for q, k, gate and up.
#
# On rtx-pro-6000 and not l40sx1: the L40S pool has been saturated all day,
# and the guard exists for SPEED ratios, which an MMLU count is not.
#
# Cost announced before the go: about 18 min on rtx-pro-6000, $0.83, timeout 1 h.
set -euo pipefail
O=/out/dclm-v4-2026-09-18
F=$O/qwen3-4b-dclm-v4.bin
uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor rtx-pro-6000 --any-flavor --timeout 1h \
  --bucket Pier-Jean/jobs-artifacts --out-mount /out \
  --name dclm-v4 \
  'echo "== oracle (hard rule 10) ==" ; oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tail -2' \
  "echo '== dclm at four times the volume, FULL split ==' ; date ; \
   LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP=$O/mmlu-4b-dclm-v4-FULL.csv \
   mmlu $F cuda 2>&1 | tee $O/out-dclm-v4.txt | tail -10 ; date" \
  "echo '== dump ==' ; wc -l $O/mmlu-4b-dclm-v4-FULL.csv"
