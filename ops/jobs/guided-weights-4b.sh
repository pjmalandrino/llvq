#!/usr/bin/env bash
# Per-token gradient weights of gate_proj and up_proj on the 4B, for the guided encode (V1).
# Preregistered: proofs/preregistration-guided-gateup-4b-2026-11-13.md, stamped before this.
#
# One Python process (ops/guided_weights.py, which imports ops/layer_vs_output.py for its pins and
# helpers) on the pytorch image of the row-scale trainings: the FP16 checkpoint Qwen/Qwen3-4B@1cfa9a72
# cast to fp32, one forward and one backward per calibration window of the encoder (64 x 2,048
# DCLM-edu tokens, the ids `bin/calibtokens` wrote on the Mac), 72 weight files of 8 MiB and a
# manifest. Reads no model on the Mac.
#
# Cost, estimated: setup 6 to 10 min (pip, checkpoint download and sha256), 64 windows at 2 to 4 s
# each with activation checkpointing (2 to 4 min), writing 1 min: 9 to 15 min billed on l40sx1,
# $0.27 to $0.45. Timeout 30m, ceiling $0.90.
#
#   bash ops/jobs/guided-weights-4b.sh upload                  the two committed scripts into the bucket
#   bash ops/jobs/guided-weights-4b.sh upload-tokens <file>    the token file and its .json into the bucket
#   DRY_RUN=1 bash ops/jobs/guided-weights-4b.sh               prints the job script, parses it, launches nothing
#   bash ops/jobs/guided-weights-4b.sh                         launches
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-guided-gateup-4b-2026-11-13.md
BUCKET=Pier-Jean/jobs-artifacts
PY_LOCAL=ops/guided_weights.py
LVO_LOCAL=ops/layer_vs_output.py
PY=guided_weights-2026-11-13.py
LVO=layer_vs_output-2026-11-13.py          # already in the bucket since the layer-vs-output job
TOK_DIR=guided-tokens-4b-2026-11-13
TOK=calib-tokens-4b.u32
TOK_BYTES=524288                          # 64 x 2,048 ids x 4 B
O=/out/guided-weights-4b-2026-11-13
S=/out/$PY
SL=/out/$LVO
T=/out/$TOK_DIR/$TOK
PINS="safetensors==0.4.5 tokenizers==0.22.2 pyarrow==20.0.0 huggingface-hub==1.19.0"
DRY=${DRY_RUN:-0}

if [ "${1:-}" = upload ]; then
  for f in "$PY_LOCAL" "$LVO_LOCAL"; do
    git diff --quiet HEAD -- "$f" || { echo "refused: $f differs from HEAD" >&2; exit 1; }
    git ls-files --error-unmatch "$f" >/dev/null || { echo "refused: $f is not committed" >&2; exit 1; }
  done
  hf buckets cp "$PY_LOCAL" "hf://buckets/$BUCKET/$PY"
  L=$(stat -f %z "$LVO_LOCAL")
  R=$(hf buckets ls "hf://buckets/$BUCKET/" 2>/dev/null | awk -v n="$LVO" '$NF == n {print $1}' || true)
  [ "$R" = "$L" ] || { echo "refused: $LVO is ${R:-absent} B in the bucket, $L B here: not the committed copy" >&2; exit 1; }
  hf buckets ls "hf://buckets/$BUCKET/" | grep -E "$PY|$LVO"
  exit 0
fi

if [ "${1:-}" = upload-tokens ]; then
  F=${2:?usage: upload-tokens <calib-tokens.u32>}
  { test -f "$F" && test -f "$F.json"; } || { echo "refused: $F or $F.json missing" >&2; exit 1; }
  test "$(stat -f %z "$F")" = "$TOK_BYTES" || { echo "refused: $F is not $TOK_BYTES B" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/$TOK_DIR/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: $TOK_DIR/ already exists in the bucket" >&2; exit 1
  fi
  hf buckets cp "$F" "hf://buckets/$BUCKET/$TOK_DIR/$TOK"
  hf buckets cp "$F.json" "hf://buckets/$BUCKET/$TOK_DIR/$TOK.json"
  hf buckets ls "hf://buckets/$BUCKET/$TOK_DIR/"
  exit 0
fi

PY_SHA=$(shasum -a 256 "$PY_LOCAL" | cut -d' ' -f1)
LVO_SHA=$(shasum -a 256 "$LVO_LOCAL" | cut -d' ' -f1)
if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  grep -q 'NOT YET STAMPED' "$PREREG" && { echo "refused: $PREREG still says NOT YET STAMPED" >&2; exit 1; }
  grep -q "$PY_SHA" "$PREREG" || { echo "refused: $PY_LOCAL sha256 $PY_SHA is not the one $PREREG names" >&2; exit 1; }
  grep -q "$LVO_SHA" "$PREREG" || { echo "refused: $LVO_LOCAL sha256 $LVO_SHA is not the one $PREREG names" >&2; exit 1; }
  git diff --quiet HEAD -- "$PY_LOCAL" "$LVO_LOCAL" "$0" || { echo "refused: scripts or job differ from HEAD" >&2; exit 1; }
  for pair in "$PY_LOCAL:$PY" "$LVO_LOCAL:$LVO"; do
    l=${pair%%:*}; r=${pair##*:}
    L=$(stat -f %z "$l")
    R=$(hf buckets ls "hf://buckets/$BUCKET/" 2>/dev/null | awk -v n="$r" '$NF == n {print $1}' || true)
    [ "$R" = "$L" ] || { echo "refused: $r is ${R:-absent} B in the bucket, $L B here; run upload" >&2; exit 1; }
  done
  R=$(hf buckets ls "hf://buckets/$BUCKET/$TOK_DIR/" 2>/dev/null | awk -v n="/$TOK" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
  [ "$R" = "$TOK_BYTES" ] || { echo "refused: $TOK_DIR/$TOK is ${R:-absent} B in the bucket, $TOK_BYTES expected; run upload-tokens" >&2; exit 1; }
  hf buckets ls "hf://buckets/$BUCKET/$TOK_DIR/" 2>/dev/null | grep -q "$TOK.json" \
    || { echo "refused: $TOK_DIR/$TOK.json absent from the bucket" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
fi

PRE=$(printf 'O=%q\nS=%q\nSL=%q\nT=%q\nPY_SHA=%q\nLVO_SHA=%q\nPINS=%q' "$O" "$S" "$SL" "$T" "$PY_SHA" "$LVO_SHA" "$PINS")

IFS= read -r -d '' BODY <<'JOB' || true
export HF_HOME=/tmp/hf
mkdir -p "$O"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt"
echo '== the scripts and the tokens on the mount: sha256 =='
{ echo "$PY_SHA  $S"; echo "$LVO_SHA  $SL"; } | sha256sum -c - | tee "$O/script.sha256"
sha256sum "$T" "$T.json" | tee "$O/tokens.sha256"
python -V | tee "$O/python.txt"
pip install --quiet --no-input $PINS
pip freeze > "$O/pip-freeze.txt"
mkdir -p /tmp/gw && cp "$S" /tmp/gw/guided_weights.py && cp "$SL" /tmp/gw/layer_vs_output.py
cp "$T" /tmp/gw/calib-tokens.u32 && cp "$T.json" /tmp/gw/calib-tokens.u32.json
echo '== selftest, tiny model, CPU =='
python /tmp/gw/guided_weights.py --selftest 2>&1 | tee "$O/selftest.txt"
grep -q 'selftest: GREEN' "$O/selftest.txt"
echo '== the capture =='
python /tmp/gw/guided_weights.py --run --tokens /tmp/gw/calib-tokens.u32 --out /tmp/gw/out 2>&1 | tee "$O/run.txt"
cp /tmp/gw/out/* "$O/"
( cd "$O" && sha256sum manifest.json w-*.f32 ) > "$O/files.sha256"
echo '== what landed ==' ; ls -la "$O" | head -20 ; ls "$O"/w-*.f32 | wc -l
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
  --flavor l40sx1 --timeout 30m \
  --bucket "$BUCKET" --out-mount /out \
  --name guided-weights-4b-2026-11-13 \
  "$PRE" "$BODY"
