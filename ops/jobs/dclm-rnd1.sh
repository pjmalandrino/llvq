#!/usr/bin/env bash
# A1 of docs/archive/plan-qualite-gratuite-2026-09-19.md: random window draws at the
# served volume, corpus and everything else held at the base.
#
# The base is dclm at x1 with a CONTIGUOUS PREFIX, 57.95. This arm changes the
# sampler and nothing else: LLVQ_CALIB_SEED=1 draws its 64 windows over the
# shard. It exists because four times the volume with a prefix COST 1.20 pp,
# and redundancy rather than coverage is the reading that would explain it.
#
# Predicted 58.3 [55.4, 61.2]. Perplexity read 16.3550 against the base's
# 16.2415, which is 0.70 percent worse and predicts nothing about the exam.
#
# On l40sx1: the rtx-pro-6000 bills 55 min where this card bills 23.
# Cost announced before the go: about 23 min, $0.72, timeout 1 h.
set -euo pipefail
O=/out/dclm-rnd1-2026-09-19
uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout 1h \
  --bucket Pier-Jean/jobs-artifacts --out-mount /out \
  --name dclm-rnd1 \
  'echo "== oracle (hard rule 10) ==" ; oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tail -2' \
  "echo '== random windows, x1 volume, FULL split ==' ; date ; \
   LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP=$O/mmlu-4b-dclm-rnd1-FULL.csv \
   mmlu $O/qwen3-4b-dclm-rnd1.bin cuda 2>&1 | tee $O/out-rnd1.txt | tail -10 ; date" \
  "echo '== dump ==' ; wc -l $O/mmlu-4b-dclm-rnd1-FULL.csv"
