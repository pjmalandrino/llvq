#!/usr/bin/env bash
# Confirmation of down_proj at int4 g128, preregistered:
#   proofs/preregistration-downproj-int4-full-2026-09-18.md
#   sha256 155fcda3d1d8f47e54a23e564ad65a45552ec3e9a370211dd614551e9cc1326a
#
# Two arms of the SAME served Q5 file on the FULL 14,042-question split. The
# primary result is NOT this score: it is the paired gain on the 11,762
# questions that took no part in selecting this arm, computed from the dumps
# afterwards.
#
# Why this arm only became legal on 2026-09-18. The exploration read down_proj
# as over budget by setting whole-model b/param against a b_max defined in
# kernel b/weight. Redone in the right unit it is 2.2044 -> 2.7226 kernel
# b/weight, under 3.00, on the same condition o_proj fits on: a native int4
# kernel for the shape, which has run on v_proj's 1024 x 2560 and no other.
#
# Cost announced before the go: about 48 min on l40sx1, $1.44, timeout 2 h.
set -euo pipefail
O=/out/downproj-full-2026-09-18
F=/out/tetra-q5-2026-09-09/qwen3-4b-tetra-q5.bin
uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout 2h \
  --bucket Pier-Jean/jobs-artifacts --out-mount /out \
  --name downproj-int4-full \
  "mkdir -p $O" \
  'echo "== oracle (hard rule 10) ==" ; oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tail -2' \
  "echo '== arm A: shipped Q5, FULL split ==' ; date ; \
   LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP=$O/mmlu-shipped-FULL.csv \
   mmlu $F cuda 2>&1 | tee $O/out-shipped.txt | tail -8 ; date" \
  "echo '== arm B: down_proj at int4 g128, same file, FULL split ==' ; date ; \
   LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_RESTORE_Q4=down_proj \
   LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP=$O/mmlu-downproj-FULL.csv \
   mmlu $F cuda 2>&1 | tee $O/out-downproj.txt | tail -8 ; date" \
  "echo '== both dumps ==' ; wc -l $O/mmlu-shipped-FULL.csv $O/mmlu-downproj-FULL.csv"
