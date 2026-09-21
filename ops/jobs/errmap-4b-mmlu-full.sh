#!/usr/bin/env bash
# MMLU on the FULL 14,042-question split, two arms of the SAME 4B file:
# Tetra as it is written, and Tetra with the gain centroids of 180 matrices
# multiplied by the sensitivity map's own factors at T = 0.06.
#
# The two files differ by 360 f64 numbers and nothing else — same codes, same
# row scales, same header, same 2.1696 b/weight — so the comparison is paired
# at the strongest level this repository can produce, and McNemar on the
# discordant questions is the test, not the sampling error of an arm.
#
# Why full and not 40 a subject: the effect predicted from perplexity is about
# 0.95 pp, which the 2,280-question sample cannot resolve. The full split was
# measured at 24 min and 0.70 $ an arm on 2026-09-11 (job 6aa442b4).
#
# ⚠️ Both dumps are on the census plan, so they pair with each other and with
# NO dump at limit=40 — `bin/mmlupair` refuses two plans.
set -euo pipefail
O=/out/errmap-4b-2026-09-16
A=$O/sealed-temoin.bin
B=$O/sealed-corrige.bin
uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout 2h \
  --bucket Pier-Jean/jobs-artifacts --out-mount /out \
  --name errmap-4b-mmlu-full \
  'echo "== oracle (hard rule 10) ==" ; oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tail -2' \
  "echo '== arm A: Tetra as written ==' ; date ; \
   LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP=$O/mmlu-temoin-FULL.csv \
   mmlu $A cuda 2>&1 | tee $O/mmlu-temoin.txt | tail -30 ; date" \
  "echo '== arm B: the same file, centroids corrected at T=0.06 ==' ; date ; \
   LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP=$O/mmlu-corrige-FULL.csv \
   mmlu $B cuda 2>&1 | tee $O/mmlu-corrige.txt | tail -30 ; date" \
  "echo '== both dumps, headers and line counts ==' ; wc -l $O/mmlu-temoin-FULL.csv $O/mmlu-corrige-FULL.csv ; head -3 $O/mmlu-temoin-FULL.csv ; head -3 $O/mmlu-corrige-FULL.csv"
