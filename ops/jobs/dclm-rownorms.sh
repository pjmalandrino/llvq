#!/usr/bin/env bash
# Train the row scales AND the RMSNorm weights on the DCLM base. Preregistered:
#   proofs/preregistration-dclm-rownorms-2026-09-22.md, sha256 <FILL AFTER STAMPING>
# The control is the 61.11 arm (ops/jobs/dclm-rowscales.sh): same export, same
# seed, same objective, same budget, same card. Only --mode differs.
#
# The probe's first KL (`first_loss` in probe.jsonl) must read the control's
# 0.35267 within 1 %: tau and sigma start at 1, so step one is the same model
# on the same batch, and only GPU nondeterminism separates the two. Anything
# else means the routing is wrong; stop the job before it bills 2 h.
#
# Cost: BUDGET of training plus loading, about 2 h 15 billed on l40sx1, ~$4.00.
set -euo pipefail
O=/out/dclm-rownorms-2026-09-22
E=/out/dclm-export-2026-09-19
S=/out/llvqtune-2026-09-22.tgz
uv run ops/run.py bench \
  --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime \
  --flavor l40sx1 --timeout 4h \
  --bucket Pier-Jean/jobs-artifacts --out-mount /out \
  --name dclm-rownorms \
  'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader' \
  'pip install --quiet --no-input transformers safetensors pyarrow huggingface_hub' \
  "mkdir -p /tmp/src && tar xzf $S -C /tmp/src && ls /tmp/src" \
  "python -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available())'" \
  "cd /tmp/src && PYTHONPATH=/tmp/src MODE=row_norms EXPORT=$E OUT=$O BUDGET=7200 bash train.sh" \
  "echo '== what landed ==' ; ls -la $O ; tail -2 $O/journal.jsonl"
