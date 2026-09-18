#!/usr/bin/env bash
# v + o + down at int4 g128, preregistered:
#   proofs/preregistration-vod-int4-full-2026-09-18.md
#   sha256 48c319fec8b463bc44741ec5a50f070d09ebffbe92d2f19788724a28a2c48320
#
# v_proj is already int4 in the served file, so "the three" needs only two
# restorations. Both components are confirmed on held-out sets of their own,
# +1.55 and +4.05 pp, so the primary here is the FULL split: nothing was
# selected by this arm.
#
# Budget: 2.9408 kernel b/weight against b_max 3.00, 0.0592 of margin, on the
# condition every confirmation carries: a native int4 kernel per shape.
#
# Cost announced before the go: about 47 min on l40sx1, $1.42, timeout 2 h.
set -euo pipefail
O=/out/vod-full-2026-09-18
F=/out/tetra-q5-2026-09-09/qwen3-4b-tetra-q5.bin
uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout 2h \
  --bucket Pier-Jean/jobs-artifacts --out-mount /out \
  --name vod-int4-full \
  "mkdir -p $O" \
  'echo "== oracle (hard rule 10) ==" ; oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tail -2' \
  "echo '== arm A: shipped Q5, FULL split ==' ; date ; \
   LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP=$O/mmlu-shipped-FULL.csv \
   mmlu $F cuda 2>&1 | tee $O/out-shipped.txt | tail -8 ; date" \
  "echo '== arm B: o_proj and down_proj at int4 g128, FULL split ==' ; date ; \
   LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_RESTORE_Q4=o_proj,down_proj \
   LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP=$O/mmlu-vod-FULL.csv \
   mmlu $F cuda 2>&1 | tee $O/out-vod.txt | tail -8 ; date" \
  "echo '== both dumps ==' ; wc -l $O/mmlu-shipped-FULL.csv $O/mmlu-vod-FULL.csv"
