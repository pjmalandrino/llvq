#!/usr/bin/env bash
# Confirmation of o_proj at int4 g128, preregistered:
#   proofs/preregistration-oproj-int4-full-2026-09-16.md
#
# Two arms of the SAME served Q5 file on the FULL 14,042-question split. The
# primary result is NOT this score: it is the paired gain on the 11,762
# questions that took no part in selecting this arm, computed from the dumps
# afterwards. The full figure contains the 2,280 selection questions and is
# therefore not independent.
#
# The exploration that selected o_proj carries individual 95 % intervals over
# six comparisons with no multiplicity correction, so it selected and
# established nothing. This run is what can establish something.
#
# Budget: o_proj at int4 is +0.1954 b/param, taking the model from 2.7645 to
# 2.9599 against a b_max of 3.00.
set -euo pipefail
O=/out/oproj-full-2026-09-16
F=/out/tetra-q5-2026-09-09/qwen3-4b-tetra-q5.bin
uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout 2h \
  --bucket Pier-Jean/jobs-artifacts --out-mount /out \
  --name oproj-int4-full \
  "mkdir -p $O" \
  'echo "== oracle (hard rule 10) ==" ; oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tail -2' \
  "echo '== arm A: shipped Q5, FULL split ==' ; date ; \
   LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP=$O/mmlu-shipped-FULL.csv \
   mmlu $F cuda 2>&1 | tee $O/out-shipped.txt | tail -8 ; date" \
  "echo '== arm B: o_proj at int4 g128, same file, FULL split ==' ; date ; \
   LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_RESTORE_Q4=o_proj \
   LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP=$O/mmlu-oproj-FULL.csv \
   mmlu $F cuda 2>&1 | tee $O/out-oproj.txt | tail -8 ; date" \
  "echo '== both dumps ==' ; wc -l $O/mmlu-shipped-FULL.csv $O/mmlu-oproj-FULL.csv"
