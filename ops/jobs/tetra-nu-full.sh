#!/usr/bin/env bash
# Bare Tetra on the full split, preregistered:
#   proofs/preregistration-tetra-nu-full-2026-09-18.md
#   sha256 b7f16ad923e0e5d3f656aebf5b57f1d538ff30fb44e87349b73db64b4113749d
#
# Bare Tetra at the 4B has never been scored on the 14,042 questions. Its only
# figure is 53.49 on the 2,280 sample, and it is the baseline of the claim that
# matters most: the paper's quality at the paper's rate, 2.1498 kernel b/weight,
# with int4 on nothing.
#
# The sample-to-full shift has no stable sign: +0.85 on the served object,
# -0.18 on f16, +2.24 on the 8B.
#
# Cost announced before the go: about 24 min on l40sx1, $0.72, timeout 1 h.
set -euo pipefail
O=/out/tetra-nu-full-2026-09-18
F=/out/tetra-4b-2026-09-06/qwen3-4b-tetra.bin
uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor rtx-pro-6000 --any-flavor --timeout 1h \
  --bucket Pier-Jean/jobs-artifacts --out-mount /out \
  --name tetra-nu-full \
  "mkdir -p $O" \
  'echo "== oracle (hard rule 10) ==" ; oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tail -2' \
  "echo '== bare Tetra 4B, FULL split ==' ; date ; \
   LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP=$O/mmlu-4b-tetra-FULL.csv \
   mmlu $F cuda 2>&1 | tee $O/out-tetra.txt | tail -10 ; date" \
  "echo '== dump ==' ; wc -l $O/mmlu-4b-tetra-FULL.csv"
