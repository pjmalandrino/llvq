#!/usr/bin/env bash
# Train the row scales of bare Tetra on a card. Preregistered:
#   proofs/preregistration-tetranu-rowscales-2026-09-19.md, sha256 1f703eb0342538cfd4ff4fa9...
#   deviations in the -ECARTS.md beside it.
#
# The step count is decided ON the card by a six-step probe, not here. The
# input is a wall budget; the token count is what comes out. That inversion is
# the lesson of 2026-09-19, where a step was priced from a cross-entropy
# measurement and the run came out three times slower than planned.
#
# Cost: BUDGET of training plus loading and the probe. At 7200 s of budget,
# about 2 h 30 billed on l40sx1, so about $4.50. Timeout caps it at 4 h.
set -euo pipefail
O=/out/tetranu-rowscales-2026-09-19
E=/out/tetranu-export-2026-09-19
S=/out/llvqtune-2026-09-19.tgz
uv run ops/run.py bench \
  --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime \
  --flavor l40sx1 --timeout 4h \
  --bucket Pier-Jean/jobs-artifacts --out-mount /out \
  --name tetranu-rowscales \
  'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader' \
  'pip install --quiet --no-input transformers safetensors pyarrow huggingface_hub' \
  "mkdir -p /tmp/src && tar xzf $S -C /tmp/src && ls /tmp/src" \
  "python -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available())'" \
  "cd /tmp/src && PYTHONPATH=/tmp/src EXPORT=$E OUT=$O BUDGET=7200 bash train.sh" \
  "echo '== what landed ==' ; ls -la $O ; tail -2 $O/journal.jsonl"
