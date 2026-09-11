#!/usr/bin/env bash
# MMLU on the FULL 14,042-question split, dense reconstruction of the served
# .llvq. No third positional argument: `bin/mmlu` defaults its limit to
# usize::MAX, which is the whole split.
#
# Why the dense arm and not the kernel: the two were paired on 2,280 questions
# on 2026-09-11 at -0.14 pp with 3 discordant of 2,280 (job 6aa414dd), so this
# number is the served object's MMLU to a tenth of a point, for 0.81 $ instead
# of 19.24 $. The kernel arm at full waits on the LLVQ_PREFILL sweep.
#
# ⚠️ This dump is on a DIFFERENT sampling plan from every other dump in
# docs/data/mmlu-dumps/ (limit=census against limit=40), and `bin/mmlupair`
# refuses to pair two plans. It is an absolute score, comparable to the
# paper's Table 6 and to nothing in this repository until the other arms are
# replayed at full.
set -euo pipefail
F=/out/tetra-q5-2026-09-09/qwen3-4b-tetra-q5.bin
O=/out/f1e-full-2026-09-11
uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout 1h \
  --bucket Pier-Jean/jobs-artifacts --out-mount /out \
  --name f1e-full-dense \
  "mkdir -p $O" \
  'echo "== oracle (hard rule 10) ==" ; oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tail -2' \
  "echo '== MMLU, FULL SPLIT, dense reconstruction of the served file ==' ; date ; \
   LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP=$O/mmlu-4b-tetra-q5-file-dense-FULL.csv \
   mmlu $F cuda 2>&1 | tee $O/mmlu-full-dense.txt | tail -34 ; date"
