#!/usr/bin/env bash
# DRAFT (integrator), not launched. Supersedes ft8b.sh for launch; same job body.
# Target home once approved: ops/jobs/dclm-8b-rowscales.sh
#
# What changed against ft8b.sh, and why:
#   1. Bucket names agree with export-fold-8b.sh, which uploads the export to
#      dclm-8b-export-2026-09-21 (export-fold-8b.sh:16). ft8b.sh:41-44 pointed at
#      -2026-09-22 and the job's first size test would have refused on the card.
#      Object folders keep the object's date (dclm-8b-2026-09-21/ already holds
#      the base, 4,364,205,777 B); override with OBJ_DATE if the prereg says so.
#   2. Timeout 2h15m, ceiling $11.25, instead of 3h / $15: the operator's cap
#      for the whole 8B is $20 (proofs/preregistration-dclm-8b-2026-09-21-ECARTS.md
#      E3), and with the census at 135m ($4.05), the FT census 60m ($1.80), the
#      served checks 45m ($1.35) and the bench 30m ($0.90) the sum of ceilings is
#      $19.35. MAX_TRAIN_SECONDS=6300 keeps the loop inside it: it refuses a probe
#      rate above 0.663 s a step (9,507 steps), and the probe read 17 % slow at 4B
#      (0.7573 against 0.6479 in the loop, 7200/0.7573 = 9,507).
#   3. Preflight on the Mac, $0: the prereg's .ots, the export's four files and
#      the tarball in the bucket with their byte counts, and an empty output dir.
#
# Cost: h200 at $5.00/h (`hf jobs hardware`, 2026-09-21); central ~80 min
# running, ~$6.90 (estimated); ceiling 2h15m = $11.25.
#
# DRY_RUN=1 bash ft8b-final.sh   prints the job script, parses it, launches nothing.
set -euo pipefail
REPO=$HOME/Documents/Pro/workspace/poc/llvq
cd "$REPO"

OBJ_DATE=${OBJ_DATE:-2026-09-21}
TGZ_DATE=${TGZ_DATE:-$OBJ_DATE}
PREREG=${PREREG:-proofs/preregistration-dclm-8b-rowscales-$OBJ_DATE.md}
BK=Pier-Jean/jobs-artifacts
O=/out/dclm-8b-rowscales-$OBJ_DATE
E=/out/dclm-8b-export-$OBJ_DATE
S=/out/llvqtune-$TGZ_DATE.tgz
EXPORT_BYTES=16381516776     # computed, export-predict.py predict on Qwen3-8B b968826d
MAX_TRAIN_SECONDS=${MAX_TRAIN_SECONDS:-7000}
MAX_FIRST_KL=${MAX_FIRST_KL:-0.50}   # the prereg fixes this value
TIMEOUT=${TIMEOUT:-135m}
DRY=${DRY_RUN:-0}

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  L=$(hf buckets ls "hf://buckets/$BK/${E#/out/}/")
  printf '%s\n' "$L"
  got() { printf '%s\n' "$L" | awk -v n="$1" '$NF ~ ("/" n "$") {print $1}'; }
  [ "$(got model.safetensors)" = "$EXPORT_BYTES" ] || { echo "refused: model.safetensors is $(got model.safetensors) B, want $EXPORT_BYTES" >&2; exit 1; }
  [ "$(got config.json)" = 728 ] && [ "$(got tokenizer.json)" = 11422654 ] && [ "$(got tokenizer_config.json)" = 64 ] \
    || { echo "refused: config/tokenizer sizes differ from the export" >&2; exit 1; }
  hf buckets ls "hf://buckets/$BK/" | grep -q "llvqtune-$TGZ_DATE\.tgz$" || { echo "refused: $S not in the bucket" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BK/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already holds files" >&2; exit 1
  fi
  # No HF_TOKEN is exported: the teacher (Qwen/Qwen3-8B) is public and the export
  # and tarball come from the bucket mount. run.py passes one only if the caller
  # set it (ops/run.py:1108).
fi

CMDS=(
  'nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader'
  'df -h /tmp | tail -1 ; free -g | head -2'
  'pip install --quiet --no-input transformers==5.17.0 safetensors==0.8.0 pyarrow==25.0.1 huggingface_hub==1.32.0'
  "mkdir -p /tmp/src && tar xzf $S -C /tmp/src && ls /tmp/src"
  "python -c 'import torch, transformers; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), torch.cuda.get_arch_list(), \"transformers\", transformers.__version__)'"
  "test \"\$(stat -c %s $E/model.safetensors)\" = $EXPORT_BYTES || { echo 'the export on the mount is not the uploaded size'; exit 2; }"
  "cd /tmp/src && PYTHONPATH=/tmp/src EXPORT=$E STAGE=/tmp/export OUT=$O TEACHER=Qwen/Qwen3-8B STEPS=9507 MAX_TRAIN_SECONDS=$MAX_TRAIN_SECONDS MAX_FIRST_KL=$MAX_FIRST_KL bash train.sh"
  "echo '== what landed ==' ; ls -la $O"
  "python -c \"import json; r=json.loads(open('$O/journal.jsonl').readlines()[-1]); r.pop('losses', None); print(r)\""
)

if [ "$DRY" = 1 ]; then
  { echo 'set -euo pipefail'; printf '%s\n' "${CMDS[@]}"; } | tee /dev/stderr | bash -n && echo "DRY_RUN: parses; nothing launched" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime \
  --flavor h200 --any-flavor --timeout "$TIMEOUT" \
  --bucket "$BK" --out-mount /out \
  --name dclm-8b-rowscales \
  "${CMDS[@]}"
