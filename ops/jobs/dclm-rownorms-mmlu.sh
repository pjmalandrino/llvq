#!/usr/bin/env bash
# Score the row_norms-trained DCLM base on the full split. Preregistered:
#   proofs/preregistration-dclm-rownorms-2026-09-22.md, sha256 <FILL AFTER STAMPING>
#
# Zero bits: the file keeps the byte count of the DCLM base, 1,794,564,765, and
# only row_scales and the f16 norms moved. Paired afterwards, on the dumps,
# against mmlu-4b-dclm-ft-FULL.csv (61.11, the row_scales control).
#
# Cost: about 24 min on l40sx1, $0.72, timeout 1h.
set -euo pipefail
O=/out/dclm-rownorms-mmlu-2026-09-22
F=/out/dclm-rownorms-2026-09-22/qwen3-4b-dclm-rownorms.bin
uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout 1h \
  --bucket Pier-Jean/jobs-artifacts --out-mount /out \
  --name dclm-rownorms-mmlu \
  "mkdir -p $O" \
  'echo "== oracle (hard rule 10) ==" ; oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tail -2' \
  "echo '== DCLM base with trained row scales and norms, FULL split ==' ; date ; \
   LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP=$O/mmlu-4b-dclm-rownorms-FULL.csv \
   mmlu $F cuda 2>&1 | tee $O/out.txt | tail -10 ; date" \
  "echo '== dump ==' ; wc -l $O/mmlu-4b-dclm-rownorms-FULL.csv"
