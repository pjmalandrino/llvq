#!/usr/bin/env bash
# 60 under 3 b/param, or the substitution rate. Preregistered:
#   proofs/preregistration-dclm-down1223-2026-09-19.md, sha256 3a98fbee2e921bb50e914bf7...
# Constant file, 0.43 pp of noise. Cost: about 23 min on l40sx1, $0.72.
set -euo pipefail
O=/out/dclm-down1223-2026-09-19
F=/out/dclm-4b-2026-09-18/qwen3-4b-dclm.bin
uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout 1h \
  --bucket Pier-Jean/jobs-artifacts --out-mount /out \
  --name dclm-down1223 \
  "mkdir -p $O" \
  'echo "== oracle (hard rule 10) ==" ; oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tail -2' \
  "echo '== DCLM base + down_proj@12-23 at int4, FULL split ==' ; date ; \
   LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_RESTORE_Q4=down_proj@12-23 \
   LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP=$O/mmlu-4b-dclm-down1223-FULL.csv \
   mmlu $F cuda 2>&1 | tee $O/out.txt | tail -10 ; date" \
  "echo '== dump ==' ; wc -l $O/mmlu-4b-dclm-down1223-FULL.csv"
