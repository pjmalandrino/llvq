#!/usr/bin/env bash
# Train the row scales on the DCLM base. Preregistered:
#   proofs/preregistration-dclm-rowscales-2026-09-19.md, sha256 f2ae5244e2322261da4afbd0...
# Signed prediction: +3.5 pp [+1.5, +4.5] over 57.95, at an unchanged 2.8126 b/param.
#
# The step count is decided ON the card by a six-step probe. The probe's first
# KL is also the free check on the new int4 export path: bare Tetra read about
# 0.44, and the DCLM base is a better object, so materially above that means the
# export is wrong and the run should be stopped rather than billed for two hours.
#
# Cost: BUDGET of training plus loading, about 2 h 15 billed on l40sx1, ~$4.00.
set -euo pipefail
O=/out/dclm-rowscales-2026-09-19
E=/out/dclm-export-2026-09-19
S=/out/llvqtune-2026-09-19.tgz
uv run ops/run.py bench \
  --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime \
  --flavor l40sx1 --timeout 4h \
  --bucket Pier-Jean/jobs-artifacts --out-mount /out \
  --name dclm-rowscales \
  'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader' \
  'pip install --quiet --no-input transformers safetensors pyarrow huggingface_hub' \
  "mkdir -p /tmp/src && tar xzf $S -C /tmp/src && ls /tmp/src" \
  "python -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available())'" \
  "cd /tmp/src && PYTHONPATH=/tmp/src EXPORT=$E OUT=$O BUDGET=7200 bash train.sh" \
  "echo '== what landed ==' ; ls -la $O ; tail -2 $O/journal.jsonl"
