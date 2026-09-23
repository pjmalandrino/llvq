#!/usr/bin/env bash
# The embedding pays for int4 tables. Preregistered:
#   proofs/preregistration-embed-q4-swap-2026-09-23.md — stamp it (ots stamp) BEFORE this.
# Constant codes, 0.43 pp of noise, paired with the 61.11 census dump.
# Signed prediction: +1.2 pp [+0.2, +2.2]. Cost: about 24 min on l40sx1, $0.72, timeout 1h.
# Needs the image republished with `embedq` (ops/Dockerfile.cuda, 2026-09-23).
set -euo pipefail
O=/out/embed-q4-swap-2026-09-23
F=/out/dclm-ft-2026-09-19/qwen3-4b-dclm-ft.bin
E=$O/qwen3-4b-dclm-ft-e4.bin
uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout 1h \
  --bucket Pier-Jean/jobs-artifacts --out-mount /out \
  --name embed-q4-swap \
  "mkdir -p $O" \
  'echo "== oracle (hard rule 10) ==" ; oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tail -2' \
  "echo '== embedq q4 on the 61.11 object ==' ; date ; \
   embedq $F $E q4 2>&1 | tee $O/embedq.txt ; \
   sha256sum $F $E | tee $O/sha256.txt ; ls -l $F $E | tee -a $O/sha256.txt" \
  "echo '== q4 embedding + o_proj + down_proj@12-23 at int4, FULL split ==' ; date ; \
   LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_RESTORE_Q4=o_proj,down_proj@12-23 \
   LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP=$O/mmlu-4b-embed-q4-swap-FULL.csv \
   mmlu $E cuda 2>&1 | tee $O/out.txt | tail -10 ; date" \
  "echo '== dump ==' ; wc -l $O/mmlu-4b-embed-q4-swap-FULL.csv"
