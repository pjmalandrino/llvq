#!/usr/bin/env bash
# Where the contrast of the four MMLU answer logits goes, FP16 against the sealed 4B, side by side.
# Preregistered: proofs/preregistration-contrast-map-4b-2026-11-14.md, which must be stamped, and must name the
# sha256 of ops/contrast_map.py and of this script, before the job launches.
#
# One Python process (ops/contrast_map.py, importing ops/layer_vs_output.py) on the pytorch image of the
# 2026-11-13 layer judge: the FP16 checkpoint Qwen/Qwen3-4B@1cfa9a72 and the f16 export of qwen3-4b-sealed.bin both
# resident; N MMLU 5-shot prompts, round robin over the 57 subjects, N fixed from the time of the first 57 (at least
# 570, at most 2,280); rule D against the two Rust census dumps; readings R1 to R5. Reads no model on the Mac.
#
# Inputs travel through the bucket folder $IN (the `upload` subcommand): the script, layer_vs_output.py and the two
# dumps (1.2 MB each), each checked by sha256 inside the job against the committed files.
#
# Cost, estimated: image and pip 3 min, inputs, export copy and load 104 s and two base hashes 50 s (the
# 2026-11-13 job), prompt check 20 s, question loop 600 s (up to 690 s if the 570 floor binds), eigendecompositions
# and report 3 min: 18 to 21 min billed on l40sx1 at $1.80/h, $0.54 to $0.63. Timeout 33m, ceiling $0.99.
#
#   bash ops/jobs/contrast-map-4b.sh upload      the committed inputs into the bucket
#   DRY_RUN=1 bash ops/jobs/contrast-map-4b.sh   prints the job script, parses it, launches nothing
#   bash ops/jobs/contrast-map-4b.sh             launches
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-contrast-map-4b-2026-11-14.md
BUCKET=Pier-Jean/jobs-artifacts
PY_LOCAL=ops/contrast_map.py
LVO_LOCAL=ops/layer_vs_output.py
DUMP_F16=docs/data/mmlu-dumps/mmlu-4b-f16-FULL.csv
DUMP_SEALED=docs/data/mmlu-dumps/mmlu-4b-abl-control-FULL.csv
INPUTS="$PY_LOCAL $LVO_LOCAL $DUMP_F16 $DUMP_SEALED"
IN=contrast-map-4b-inputs-2026-11-14
OUT=contrast-map-4b-2026-11-14
EXPORT_DIR=sealed-4b-export-2026-09-23
FLAVOR=l40sx1
USD_H=1.80
TIMEOUT=${TIMEOUT:-33m}
BUDGET=${BUDGET:-600}
# Sizes listed in the bucket on 2026-11-13 (docs/mesures/layer-vs-output-4b-2026-11-13-brut/pre-launch/buckets.txt).
SIZES="model.safetensors:8044981648 config.json:726"
PINS="safetensors==0.4.5 tokenizers==0.22.2 pyarrow==20.0.0 huggingface-hub==1.19.0"
DRY=${DRY_RUN:-0}

refuse() { echo "refused: $*" >&2; exit 1; }
committed() {
  for f in "$@"; do
    git ls-files --error-unmatch "$f" >/dev/null 2>&1 || refuse "$f is not committed"
    git diff --quiet HEAD -- "$f" || refuse "$f differs from HEAD"
  done
}
remote_size() {  # size of $IN/<basename> in the bucket, empty if absent
  hf buckets ls "hf://buckets/$BUCKET/$IN/" 2>/dev/null \
    | awk -v n="/$1" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true
}

if [ "${1:-}" = upload ]; then
  committed $INPUTS
  for f in $INPUTS; do hf buckets cp "$f" "hf://buckets/$BUCKET/$IN/$(basename "$f")"; done
  hf buckets ls "hf://buckets/$BUCKET/$IN/"
  exit 0
fi
[ -z "${1:-}" ] || refuse "usage: $0 [upload]"

sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
PY_SHA=$(sha "$PY_LOCAL"); LVO_SHA=$(sha "$LVO_LOCAL"); F16_SHA=$(sha "$DUMP_F16"); SEALED_SHA=$(sha "$DUMP_SEALED")
JOB_SHA=$(sha "$0")
# The script pins the dumps itself; a mismatch here would fail inside the job after the card is billed.
grep -q "$F16_SHA" "$PY_LOCAL" && grep -q "$SEALED_SHA" "$PY_LOCAL" || refuse "the dumps are not the ones $PY_LOCAL pins"

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || refuse "$PREREG(.ots) missing"
  # The anchor is the proof: the banner a stamped prereg keeps is not read.
  ots info "$PREREG.ots" 2>/dev/null | grep -q "$(sha "$PREREG")" || refuse "$PREREG.ots does not anchor the sha256 of $PREREG"
  grep -q "$PY_SHA" "$PREREG" || refuse "$PREREG does not name the sha256 of $PY_LOCAL ($PY_SHA)"
  grep -q "$JOB_SHA" "$PREREG" || refuse "$PREREG does not name this script's sha256 ($JOB_SHA)"
  committed "$0" $INPUTS
  for f in $INPUTS; do
    L=$(stat -f %z "$f"); R=$(remote_size "$(basename "$f")")
    [ "$R" = "$L" ] || refuse "$IN/$(basename "$f") is ${R:-absent} B in the bucket, $L B here; run upload"
  done
  for fs in $SIZES; do
    f=${fs%%:*}; want=${fs##*:}
    R=$(hf buckets ls "hf://buckets/$BUCKET/$EXPORT_DIR/" 2>/dev/null | awk -v n="/$f" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
    [ "$R" = "$want" ] || refuse "$EXPORT_DIR/$f is ${R:-absent} B in the bucket, $want expected"
  done
  if hf buckets ls "hf://buckets/$BUCKET/$OUT/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    refuse "$OUT/ already exists in the bucket"
  fi
fi

MIN=${TIMEOUT%m}
echo "cost: $FLAVOR at \$$USD_H/h, timeout $TIMEOUT, ceiling \$$(awk -v m="$MIN" -v r="$USD_H" 'BEGIN {printf "%.2f", m / 60 * r}');" \
     "estimated 18 to 21 min, \$0.54 to \$0.63 (setup as 2026-11-13: 104 s inputs and load, 25 s per base hash;" \
     "loop under --budget-seconds $BUDGET; eigendecompositions and report about 3 min)"
echo "inputs: $IN/ ($PY_LOCAL $PY_SHA, $LVO_LOCAL $LVO_SHA, dumps $F16_SHA $SEALED_SHA); out $OUT/; job sha256 $JOB_SHA"

PRE=$(printf 'O=%q\nE=%q\nI=%q\nPY_SHA=%q\nLVO_SHA=%q\nF16_SHA=%q\nSEALED_SHA=%q\nBUDGET=%q\nPINS=%q' \
  "/out/$OUT" "/out/$EXPORT_DIR" "/out/$IN" "$PY_SHA" "$LVO_SHA" "$F16_SHA" "$SEALED_SHA" "$BUDGET" "$PINS")

IFS= read -r -d '' BODY <<'JOB' || true
export HF_HOME=/tmp/hf
mkdir -p "$O" /tmp/cm
date -u +%FT%TZ | tee "$O/start.txt"
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt"
echo "nproc $(nproc)" | tee -a "$O/gpu.txt"
echo '== the inputs on the mount: sha256 =='
{ echo "$PY_SHA  $I/contrast_map.py"; echo "$LVO_SHA  $I/layer_vs_output.py"
  echo "$F16_SHA  $I/mmlu-4b-f16-FULL.csv"; echo "$SEALED_SHA  $I/mmlu-4b-abl-control-FULL.csv"; } \
  | sha256sum -c - | tee "$O/inputs.sha256"
python -V | tee "$O/python.txt"
pip install --quiet --no-input $PINS
pip freeze > "$O/pip-freeze.txt"
cp "$I/contrast_map.py" "$I/layer_vs_output.py" /tmp/cm/
echo '== selftest, tiny model, CPU =='
python /tmp/cm/contrast_map.py --selftest 2>&1 | tee "$O/selftest.txt"
grep -q '^selftest: GREEN' "$O/selftest.txt" || { echo 'refused: selftest is not green' >&2; exit 1; }
echo '== the measurement =='
set +e
python /tmp/cm/contrast_map.py --run --export "$E/model.safetensors" --out "$O" --stage /tmp/cm-export \
  --dump-f16 "$I/mmlu-4b-f16-FULL.csv" --dump-sealed "$I/mmlu-4b-abl-control-FULL.csv" \
  --budget-seconds "$BUDGET" 2>&1 | tee "$O/run.txt"
rc=${PIPESTATUS[0]}
set -e
echo "exit $rc (0 green, 4 rule D red, 3 an invariant broke)" | tee "$O/exit.txt"
date -u +%FT%TZ | tee "$O/end.txt"
echo '== what landed ==' ; ls -la "$O"
exit "$rc"
JOB

if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; $FLAVOR, timeout $TIMEOUT; nothing launched" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime \
  --flavor "$FLAVOR" --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "$OUT" \
  "$PRE" "$BODY"
