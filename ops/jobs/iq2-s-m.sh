#!/usr/bin/env bash
# IQ2_S and IQ2_M, preregistered:
#   proofs/preregistration-iq2sm-2026-09-18.md
#   sha256 23e6e052047de1d29c774bc9...
#
# The record's IQ2 number is XXS at 38.87, the bottom rung at ~2.06 bpw, against
# our bare Tetra at 2.1498 and 53.49. That comparison flatters us. S (~2.50) and
# M (~2.70) are the rungs a reader would ask about.
#
# Nothing is rebuilt: the prompts, the source GGUF and the scorer all come from
# the bucket unchanged, so these arms are byte-comparable with the XXS arm.
# The rate is read from the FILE SIZE, not from llama.cpp's advertised bpw.
#
# Cost announced before the go: about 20 min on l40sx1, $0.60, timeout 1 h.
set -euo pipefail
M=/out/m4-iq2-cuda-1b57b7d3
O=/out/iq2-s-m-2026-09-18
uv run ops/run.py bench \
  --image ghcr.io/ggml-org/llama.cpp:full-cuda \
  --flavor rtx-pro-6000 --any-flavor --timeout 1h \
  --bucket Pier-Jean/jobs-artifacts --out-mount /out \
  --name iq2-s-m \
  "mkdir -p $O ; ls -l $M/qwen3-4b-f16.gguf $M/mmlu-prompts.jsonl" \
  "Q=\$(command -v llama-quantize || echo /app/llama-quantize) ; \
   S=\$(command -v llama-server   || echo /app/llama-server) ; \
   echo \"quantize=\$Q server=\$S\" ; \$Q --help 2>&1 | head -2 || true" \
  "for T in IQ2_S IQ2_M ; do \
     echo \"== quantize \$T ==\" ; date ; \
     \$(command -v llama-quantize || echo /app/llama-quantize) \
       $M/qwen3-4b-f16.gguf $O/qwen3-4b-\${T,,}.gguf \$T 2>&1 | tail -3 ; date ; \
   done ; ls -l $O ; sha256sum $O/*.gguf | tee $O/gguf.sha256" \
  "for T in iq2_s iq2_m ; do \
     echo \"== serve and score \$T ==\" ; date ; \
     \$(command -v llama-server || echo /app/llama-server) \
       -m $O/qwen3-4b-\$T.gguf --port 8080 -ngl 99 -c 4096 --host 127.0.0.1 > $O/server-\$T.log 2>&1 & \
     SRV=\$! ; \
     for i in \$(seq 1 120) ; do \
       curl -sf http://127.0.0.1:8080/health > /dev/null && break ; sleep 2 ; \
     done ; \
     python3 $M/gguf_mmlu_thin.py $M/mmlu-prompts.jsonl $O/mmlu-4b-gguf-\$T.csv 2>&1 | tail -12 ; \
     kill \$SRV ; wait \$SRV 2>/dev/null || true ; date ; \
   done" \
  "echo '== dumps et tailles ==' ; wc -l $O/*.csv ; ls -l $O/*.gguf"
