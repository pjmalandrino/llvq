#!/usr/bin/env bash
# Does the int4 allocation transpose to the 8B, preregistered:
#   proofs/preregistration-vod-8b-2026-09-18.md
#   sha256 25d9990e78a83f0b080e4cba15e0915e7258ea739b404a707754a457ff44de82
#
# No re-encoding: LLVQ_RESTORE_Q4 reads the checkpoint and replaces at load, on
# any sealed file, and the Tetra 8B is in the bucket. Three types and not two,
# because the 8B has no Q5 mix: v_proj is inside the arm, not the baseline.
#
# Budget v+o+down at the 8B: 2.9260 kernel b/weight, margin 0.0740 under b_max,
# against the 4B's 2.9408 and 0.0592 (computed, 6,945,767,424 projection weights).
#
# No 8B has ever been scored on the full split, so both arms are firsts.
# Cost announced before the go: about 95 min on l40sx1, $2.85, timeout 3 h.
set -euo pipefail
O=/out/vod-8b-2026-09-18
F=/out/tetra-8b-2026-09-06/qwen3-8b-tetra.bin
uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout 3h \
  --bucket Pier-Jean/jobs-artifacts --out-mount /out \
  --name vod-8b \
  "mkdir -p $O" \
  'echo "== oracle (hard rule 10) ==" ; oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tail -2' \
  "echo '== arm A: bare Tetra 8B, FULL split ==' ; date ; \
   LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP=$O/mmlu-8b-tetra-FULL.csv \
   mmlu $F cuda 2>&1 | tee $O/out-tetra.txt | tail -8 ; date" \
  "echo '== arm B: v_proj, o_proj and down_proj at int4 g128, FULL split ==' ; date ; \
   LLVQ_MODEL=Qwen/Qwen3-8B LLVQ_RESTORE_Q4=v_proj,o_proj,down_proj \
   LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP=$O/mmlu-8b-vod-FULL.csv \
   mmlu $F cuda 2>&1 | tee $O/out-vod.txt | tail -8 ; date" \
  "echo '== both dumps ==' ; wc -l $O/mmlu-8b-*.csv"
