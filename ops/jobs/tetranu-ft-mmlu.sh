#!/usr/bin/env bash
# Score the row-scale-trained bare Tetra on the full split. Preregistered:
#   proofs/preregistration-tetranu-rowscales-2026-09-19.md, sha256 1f703eb0342538cfd4ff4fa9...
#   deviations E6-E10 in the -ECARTS.md beside it, written BEFORE this arm.
#
# Zero bits: the file is byte-for-byte the size of the one it came from, and only
# row_scales moved. Signed prediction, revised before launch: +0.4 pp [-0.8, +1.6]
# over bare Tetra's 54.64.
#
# Cost: about 23 min on l40sx1, $0.72, timeout 1h.
set -euo pipefail
O=/out/tetranu-ft-mmlu-2026-09-19
F=/out/tetranu-ft-2026-09-19/qwen3-4b-tetra-ft.bin
uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout 1h \
  --bucket Pier-Jean/jobs-artifacts --out-mount /out \
  --name tetranu-ft-mmlu \
  "mkdir -p $O" \
  'echo "== oracle (hard rule 10) ==" ; oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tail -2' \
  "echo '== bare Tetra with trained row scales, FULL split ==' ; date ; \
   LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP=$O/mmlu-4b-tetranu-ft-FULL.csv \
   mmlu $F cuda 2>&1 | tee $O/out.txt | tail -10 ; date" \
  "echo '== dump ==' ; wc -l $O/mmlu-4b-tetranu-ft-FULL.csv"
