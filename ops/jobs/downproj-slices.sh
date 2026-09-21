#!/usr/bin/env bash
# Is down_proj's gain concentrated by depth, preregistered:
#   proofs/preregistration-downproj-slices-2026-09-18.md
#   sha256 863fed682d3c5a6e4cdc76ba1f5efbfe5984360c9dcaf8dd1656bb6373c7c520
#
# Three slices of twelve layers, 298,844,160 weights and +0.17273 kernel
# b/weight each. No witness arm: the primary is the comparison BETWEEN the
# slices, and the shipped dump has reproduced byte for byte across three jobs.
#
# Requires the layer-window grammar landed 2026-09-18, so the image must be
# republished before this runs: an older image refuses `down_proj@0-11` by name.
#
# Cost announced before the go: about 72 min on l40sx1, $2.16, timeout 2 h.
set -euo pipefail
O=/out/downproj-slices-2026-09-18
F=/out/tetra-q5-2026-09-09/qwen3-4b-tetra-q5.bin
uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout 2h \
  --bucket Pier-Jean/jobs-artifacts --out-mount /out \
  --name downproj-slices \
  "mkdir -p $O" \
  'echo "== oracle (hard rule 10) ==" ; oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tail -2' \
  "for W in 0-11 12-23 24-35 ; do \
     echo \"== down_proj@\$W, FULL split ==\" ; date ; \
     LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_RESTORE_Q4=down_proj@\$W \
     LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP=$O/mmlu-down-\$W.csv \
     mmlu $F cuda 2>&1 | tee $O/out-\$W.txt | tail -8 ; date ; \
   done" \
  "echo '== three dumps ==' ; wc -l $O/mmlu-down-*.csv"
