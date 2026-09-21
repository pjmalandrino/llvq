#!/usr/bin/env bash
# UD-IQ2_M against the same 2,280 questions, preregistered:
#   proofs/preregistration-iq2sm-2026-09-18.md, sha256 23e6e052047de1d2...
#
# IQ2_S is dropped, not skipped: llama-quantize refuses it without an imatrix,
# so these quants are calibrated exactly as ours are, and building one would
# measure OUR imatrix rather than what a user downloads. IQ2_M is published
# pre-built by unsloth for this exact model.
#
# Read from the file: 1,532,931,872 bytes = 3.0487 b/param whole model, which
# sits between our Q5 at 2.8126 (56.37) and our +down at 3.2807 (60.44).
#
# On l40sx1 and not rtx-pro-6000: the rtx took 55 min where the l40s takes 23
# for the same MMLU, at 1.5x the hourly rate. Measured 2026-09-18, twice.
#
# Cost announced before the go: about 12 min on l40sx1, $0.36, timeout 40m.
set -euo pipefail
M=/out/m4-iq2-cuda-1b57b7d3
G=/out/iq2m-2026-09-19
uv run ops/run.py bench \
  --image ghcr.io/ggml-org/llama.cpp:full-cuda \
  --flavor l40sx1 --timeout 40m \
  --bucket Pier-Jean/jobs-artifacts --out-mount /out \
  --name iq2m \
  "ls -l $G/Qwen3-4B-UD-IQ2_M.gguf $M/mmlu-prompts.jsonl ; sha256sum $G/Qwen3-4B-UD-IQ2_M.gguf" \
  "S=\$(command -v llama-server || echo /app/llama-server) ; \
   echo \"== serve UD-IQ2_M ==\" ; date ; \
   \$S -m $G/Qwen3-4B-UD-IQ2_M.gguf --port 8080 -ngl 99 -c 4096 --host 127.0.0.1 > $G/server.log 2>&1 & \
   SRV=\$! ; \
   for i in \$(seq 1 150) ; do curl -sf http://127.0.0.1:8080/health > /dev/null && break ; sleep 2 ; done ; \
   echo 'server up' ; date ; \
   python3 $M/gguf_mmlu_thin.py $M/mmlu-prompts.jsonl $G/mmlu-4b-gguf-iq2m.csv 2>&1 | tail -14 ; \
   kill \$SRV ; wait \$SRV 2>/dev/null || true ; date" \
  "echo '== dump ==' ; wc -l $G/mmlu-4b-gguf-iq2m.csv ; tail -2 $G/server.log"
