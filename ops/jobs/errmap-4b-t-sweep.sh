#!/usr/bin/env bash
# Does a trust half-width exist where the correction gains perplexity without
# losing accuracy?
#
# T = 0.06 was chosen as the PERPLEXITY optimum and cost 3.06 pp of MMLU
# (docs/mesures/errmap-mmlu-4b-2026-09-16.txt). Nothing says the two curves
# cross at the same place. Four arms of the same file — T = 0 (the witness),
# 0.01, 0.02, 0.04 — on the same 2,280 questions, so they pair with each other.
#
# ⚠️ limit=40, so these four dumps pair among themselves and NOT with the two
# census dumps of 2026-09-16. `bin/mmlupair` refuses two sampling plans.
set -euo pipefail
O=/out/errmap-4b-tsweep-2026-09-16
S=/out/errmap-4b-2026-09-16
uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout 1h \
  --bucket Pier-Jean/jobs-artifacts --out-mount /out \
  --name errmap-4b-t-sweep \
  "mkdir -p $O" \
  'echo "== oracle (hard rule 10) ==" ; oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tail -2' \
  "for T in temoin t01 t02 t04 ; do \
     echo \"== arm \$T ==\" ; date ; \
     F=$S/sealed-\$T.bin ; [ \$T = temoin ] && F=$S/sealed-temoin.bin ; \
     LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP=$O/mmlu-\$T.csv \
     mmlu \$F cuda 40 2>&1 | tee $O/out-\$T.txt | tail -6 ; date ; \
   done" \
  "echo '== recap ==' ; grep -h 'MMLU (micro' $O/out-*.txt ; wc -l $O/mmlu-*.csv"
