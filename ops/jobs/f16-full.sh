#!/usr/bin/env bash
# The f16 reference on the FULL MMLU split, preregistered:
#   proofs/preregistration-f16-full-2026-09-18.md
#   sha256 2719b5b52f449818addb5521f54138e7d6b1f568c105fd7923389587b1e0e804
#
# No f16 has ever been scored on the 14,042 questions. Every "N points below
# f16" in this repository therefore divides a full-split number by a sampled
# one, and the census of 2026-09-11 showed such an estimate can sit 0.85 pp
# from the full value.
#
# One arm: nothing is compared inside this job. The dump carries the plan
# fingerprint a74a6d6213602979, which makes it pairable afterwards against every
# committed FULL dump.
#
# Cost announced before the go: about 24 min on l40sx1, $0.72, timeout 1 h.
set -euo pipefail
O=/out/f16-full-2026-09-18
uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout 1h \
  --bucket Pier-Jean/jobs-artifacts --out-mount /out \
  --name f16-full \
  "mkdir -p $O" \
  'echo "== oracle (hard rule 10) ==" ; oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tail -2' \
  "echo '== Qwen3-4B in f16, FULL split ==' ; date ; \
   LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP=$O/mmlu-4b-f16-FULL.csv \
   mmlu Qwen/Qwen3-4B cuda 2>&1 | tee $O/out-f16.txt | tail -10 ; date" \
  "echo '== dump ==' ; wc -l $O/mmlu-4b-f16-FULL.csv ; tail -1 $O/mmlu-4b-f16-FULL.csv"
