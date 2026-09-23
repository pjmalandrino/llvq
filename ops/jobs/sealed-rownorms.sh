#!/usr/bin/env bash
# A second training on the sealed 4B: the 168 Tetra row scales and the 73
# RMSNorm weights, the 84 int4 records left alone. Preregistered:
#   proofs/preregistration-sealed-rownorms-2026-09-23.md, stamped before this.
#
# The 61.11 recipe (ops/jobs/dclm-rowscales.sh) with MODE=row_norms, on the
# export of qwen3-4b-sealed.bin. train.sh stops by itself before training when
# the probe's first KL leaves [0.25, 0.27] (the Mac probe read 0.2607).
#
# Cost, estimated: ~2 h 15 billed on l40sx1, ~$4.00; timeout 4h, ceiling $7.20.
#
#   bash ops/jobs/sealed-rownorms.sh upload    the export and the trainer archive into the bucket
#   DRY_RUN=1 bash ops/jobs/sealed-rownorms.sh prints the job script, parses it, launches nothing
#   bash ops/jobs/sealed-rownorms.sh           launches
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-sealed-rownorms-2026-09-23.md
BUCKET=Pier-Jean/jobs-artifacts
EXPORT_LOCAL=${EXPORT_LOCAL:-$HOME/q4b-sealed-2026-09-23/export}
EXPORT_DIR=sealed-4b-export-2026-09-23
TGZ_LOCAL=${TGZ_LOCAL:-$HOME/q4b-sealed-2026-09-23/llvqtune-2026-09-23.tgz}
TGZ=llvqtune-2026-09-23.tgz
O=/out/sealed-rownorms-2026-09-23
E=/out/$EXPORT_DIR
S=/out/$TGZ
BAND="0.25 0.27"
FILES="model.safetensors config.json tokenizer.json tokenizer_config.json llvq-int4.json"
DRY=${DRY_RUN:-0}

if [ "${1:-}" = upload ]; then
  git diff --quiet HEAD -- ops/llvqtune || { echo "refused: ops/llvqtune differs from HEAD" >&2; exit 1; }
  tar czf "$TGZ_LOCAL" -C ops/llvqtune train.sh llvqtune
  tar tzf "$TGZ_LOCAL" | grep -qE '^llvqtune/trainables/row_norms\.py$'
  tar xzOf "$TGZ_LOCAL" train.sh | grep -q FIRST_LOSS_BAND
  for f in $FILES; do
    hf buckets cp "$EXPORT_LOCAL/$f" "hf://buckets/$BUCKET/$EXPORT_DIR/$f"
  done
  hf buckets cp "$TGZ_LOCAL" "hf://buckets/$BUCKET/$TGZ"
  hf buckets ls "hf://buckets/$BUCKET/$EXPORT_DIR/"
  exit 0
fi

SUMS=$(cd "$EXPORT_LOCAL" && shasum -a 256 $FILES)
TGZ_SHA=$(shasum -a 256 "$TGZ_LOCAL" 2>/dev/null | cut -d' ' -f1 || true)
if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  [ ${#TGZ_SHA} -eq 64 ] || { echo "refused: $TGZ_LOCAL missing, run upload" >&2; exit 1; }
  for f in $FILES; do
    L=$(stat -f %z "$EXPORT_LOCAL/$f")
    R=$(hf buckets ls "hf://buckets/$BUCKET/$EXPORT_DIR/" 2>/dev/null | awk -v n="/$f" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
    [ "$R" = "$L" ] || { echo "refused: $f is ${R:-absent} B in the bucket, $L B here" >&2; exit 1; }
  done
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
fi

PRE=$(printf 'O=%q\nE=%q\nS=%q\nBAND=%q\nSUMS=%q\nTGZ_SHA=%q' "$O" "$E" "$S" "$BAND" "$SUMS" "$TGZ_SHA")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt"
echo '== inputs on the mount: sha256 against the Mac =='
printf '%s\n' "$SUMS" | sed "s#  #  $E/#" > "$O/export.sha256"
sha256sum -c "$O/export.sha256"
echo "$TGZ_SHA  $S" | sha256sum -c -
python -c "import json,sys; n=json.load(open(sys.argv[1]))['int4']; print(len(n), 'int4 names'); sys.exit(0 if len(n)==84 else 3)" "$E/llvq-int4.json"
pip install --quiet --no-input transformers safetensors pyarrow huggingface_hub
mkdir -p /tmp/src && tar xzf "$S" -C /tmp/src && ls /tmp/src
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'
cd /tmp/src && PYTHONPATH=/tmp/src MODE=row_norms EXPORT="$E" OUT="$O" BUDGET=7200 FIRST_LOSS_BAND="$BAND" bash train.sh
echo '== what landed ==' ; ls -la "$O" ; tail -2 "$O/journal.jsonl"
JOB

if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; nothing launched" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime \
  --flavor l40sx1 --timeout 4h \
  --bucket "$BUCKET" --out-mount /out \
  --name sealed-rownorms \
  "$PRE" "$BODY"
