#!/usr/bin/env bash
# The layer judge against the final-output judge, by projection group, on the sealed 4B.
# Preregistered: proofs/preregistration-layer-vs-output-4b-2026-11-13.md, stamped before this.
#
# One Python process (ops/layer_vs_output.py) on the pytorch image of the row-scale trainings: the FP16
# checkpoint Qwen/Qwen3-4B@1cfa9a72 and the f16 export of qwen3-4b-sealed.bin both resident; per window of
# WikiText-2 test, KL to FP16 and NLL for the sealed model, seven mirror arms, seven direct arms; note 1 per
# matrix hooked in the FP16 and sealed passes. Reads no model on the Mac.
#
# Cost, estimated: setup 8 to 11 min, window loop 20 min (the count is fixed from the first window's time,
# at least 12), report 1 min: 30 to 35 min billed on l40sx1, $0.90 to $1.05. Timeout 45m, ceiling $1.35.
#
#   bash ops/jobs/layer-vs-output-4b.sh upload    the committed script into the bucket
#   DRY_RUN=1 bash ops/jobs/layer-vs-output-4b.sh prints the job script, parses it, launches nothing
#   bash ops/jobs/layer-vs-output-4b.sh           launches
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-layer-vs-output-4b-2026-11-13.md
BUCKET=Pier-Jean/jobs-artifacts
PY_LOCAL=ops/layer_vs_output.py
PY=layer_vs_output-2026-11-13.py
EXPORT_DIR=sealed-4b-export-2026-09-23
O=/out/layer-vs-output-4b-2026-11-13
E=/out/$EXPORT_DIR
S=/out/$PY
BUDGET=1200
# Sizes listed in the bucket on 2026-11-13 (docs/mesures/layer-vs-output-4b-2026-11-13-brut/pre-launch/buckets.txt).
SIZES="model.safetensors:8044981648 config.json:726"
PINS="safetensors==0.4.5 tokenizers==0.22.2 pyarrow==20.0.0 huggingface-hub==1.19.0"
DRY=${DRY_RUN:-0}

if [ "${1:-}" = upload ]; then
  git diff --quiet HEAD -- "$PY_LOCAL" || { echo "refused: $PY_LOCAL differs from HEAD" >&2; exit 1; }
  git ls-files --error-unmatch "$PY_LOCAL" >/dev/null || { echo "refused: $PY_LOCAL is not committed" >&2; exit 1; }
  hf buckets cp "$PY_LOCAL" "hf://buckets/$BUCKET/$PY"
  hf buckets ls "hf://buckets/$BUCKET/" | grep "$PY"
  exit 0
fi

PY_SHA=$(shasum -a 256 "$PY_LOCAL" | cut -d' ' -f1)
if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  grep -q "$PY_SHA" "$PREREG" || { echo "refused: $PY_LOCAL sha256 $PY_SHA is not the one $PREREG names" >&2; exit 1; }
  git diff --quiet HEAD -- "$PY_LOCAL" "$0" || { echo "refused: script or job differ from HEAD" >&2; exit 1; }
  L=$(stat -f %z "$PY_LOCAL")
  R=$(hf buckets ls "hf://buckets/$BUCKET/" 2>/dev/null | awk -v n="$PY" '$NF == n {print $1}' || true)
  [ "$R" = "$L" ] || { echo "refused: $PY is ${R:-absent} B in the bucket, $L B here; run upload" >&2; exit 1; }
  for fs in $SIZES; do
    f=${fs%%:*}; want=${fs##*:}
    R=$(hf buckets ls "hf://buckets/$BUCKET/$EXPORT_DIR/" 2>/dev/null | awk -v n="/$f" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
    [ "$R" = "$want" ] || { echo "refused: $EXPORT_DIR/$f is ${R:-absent} B in the bucket, $want expected" >&2; exit 1; }
  done
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
fi

PRE=$(printf 'O=%q\nE=%q\nS=%q\nPY_SHA=%q\nBUDGET=%q\nPINS=%q' "$O" "$E" "$S" "$PY_SHA" "$BUDGET" "$PINS")

IFS= read -r -d '' BODY <<'JOB' || true
export HF_HOME=/tmp/hf
mkdir -p "$O"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt"
echo '== the script on the mount: sha256 =='
echo "$PY_SHA  $S" | sha256sum -c - | tee "$O/script.sha256"
python -V | tee "$O/python.txt"
pip install --quiet --no-input $PINS
pip freeze > "$O/pip-freeze.txt"
cp "$S" /tmp/layer_vs_output.py
echo '== selftest, tiny model, CPU =='
python /tmp/layer_vs_output.py --selftest 2>&1 | tee "$O/selftest.txt"
echo '== the measurement =='
python /tmp/layer_vs_output.py --run --export "$E/model.safetensors" --out "$O" --stage /tmp/lvo-export \
  --budget-seconds "$BUDGET" 2>&1 | tee "$O/run.txt"
echo '== what landed ==' ; ls -la "$O"
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
  --flavor l40sx1 --timeout 45m \
  --bucket "$BUCKET" --out-mount /out \
  --name layer-vs-output-4b-2026-11-13 \
  "$PRE" "$BODY"
