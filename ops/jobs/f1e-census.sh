#!/usr/bin/env bash
# The F1e MMLU census through the served kernel: two arms on the SAME mixed
# file, one job, the same 2,280 questions. Prereg
# proofs/preregistration-f1e-census-2026-09-11.md (stamped at de89530); the
# runbook is configs/README.md. ~2.1 h and ~3.75 $ on l40sx1 (*computed* on
# the measured prefill slope), 5.40 $ at the 3 h ceiling.
#
# Not launched on 2026-09-11: the operator paused it with everything ready.
# Run from the repository root; the pair (`mmlupair`) runs on the Mac
# afterwards, it is not in the image.
set -euo pipefail
F=/out/tetra-q5-2026-09-09/qwen3-4b-tetra-q5.bin
C=/usr/local/share/llvq/configs/qwen3-4b-tetra-q5.json
O=/out/f1e-census-2026-09-11
uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout 3h \
  --bucket Pier-Jean/jobs-artifacts --out-mount /out \
  --name f1e-census \
  "mkdir -p $O" \
  "nvidia-smi > $O/gpu.txt 2>&1 || true" \
  "test -f $C" \
  'echo "== oracle (hard rule 10) ==" ; oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tail -2' \
  "echo '== arm A: the dense reconstruction of the MIXED file, 40 a subject ==' ; date ; LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP=$O/mmlu-4b-tetra-q5-file-dense.csv mmlu $F cuda 40 2>&1 | tee $O/mmlu-dense.txt | tail -28 ; date" \
  "echo '== arm B: the served KERNEL on the same file, same 2,280 questions ==' ; date ; LLVQ_CONFIG=$C LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP=$O/mmlu-4b-tetra-q5-file-kernel.csv mmlu $F cuda 40 2>&1 | tee $O/mmlu-kernel.txt | tail -28 ; date" \
  "echo '== both dumps, headers ==' ; head -12 $O/mmlu-4b-tetra-q5-file-dense.csv ; echo ; head -12 $O/mmlu-4b-tetra-q5-file-kernel.csv ; echo '(mmlupair runs on the Mac after hf buckets cp: it is not in the image)'"
