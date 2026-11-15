#!/usr/bin/env bash
# The patch map: which part of FP16's residual stream, put into the sealed 4B's run at one depth, gives back its MMLU
# deficit. FP16 Qwen3-4B and the sealed 4B's f16 export side by side on contrast-map-4b's 570 prompts.
# Preregistered: proofs/preregistration-patch-map-4b-2026-11-15.md, which must be stamped, and must name the sha256
# of ops/patch_map.py and of this script, before the job launches.
#
# One Python process (ops/patch_map.py, importing ops/contrast_map.py and ops/layer_vs_output.py, neither edited) on
# the pytorch image of contrast-map-4b. Pass 1: FP16 alone (the depth-12 band basis, the massive positions, the
# matrices' errors on FP16 inputs). Pass 2: FP16, the sealed control and the 16 arms, each resumed from the
# control's own stream; the first 57 prompts also run the exactness controls. Gates, then readings, then the kill
# rules of the prereg. Reads no model on the Mac.
#
# Inputs travel through the bucket folder $IN (the `upload` subcommand): the three scripts, the two Rust dumps and
# five raw files of contrast-map-4b, each checked by sha256 inside the job against the committed files.
#
# Cost, estimated (prereg section 7): setup about 4 min (image and pip, two selftests, export copy and load 95 s,
# two base hashes 24 s each), pass 1 at most 6 min, pass 2 at most 16 min with the controls, analysis about 2 min:
# 14 to 28 min billed on l40sx1 at $1.80/h, $0.42 to $0.84. The script stops itself after 114 prompts of pass 2 if pass 1 + pass 2 would
# pass 30 min. Timeout 40m, ceiling $1.20.
#
#   bash ops/jobs/patch-map-4b.sh upload      the committed inputs into the bucket
#   DRY_RUN=1 bash ops/jobs/patch-map-4b.sh   prints the job script, parses it, launches nothing
#   bash ops/jobs/patch-map-4b.sh             launches
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-patch-map-4b-2026-11-15.md
BUCKET=Pier-Jean/jobs-artifacts
PY_LOCAL=ops/patch_map.py
CM_LOCAL=ops/contrast_map.py
LVO_LOCAL=ops/layer_vs_output.py
DUMP_F16=docs/data/mmlu-dumps/mmlu-4b-f16-FULL.csv
DUMP_SEALED=docs/data/mmlu-dumps/mmlu-4b-abl-control-FULL.csv
REF=docs/mesures/contrast-map-4b-2026-11-14-brut
REFS="$REF/control.tsv $REF/questions.tsv $REF/matrices.tsv $REF/lens.tsv $REF/stream_bands.tsv"
INPUTS="$PY_LOCAL $CM_LOCAL $LVO_LOCAL $DUMP_F16 $DUMP_SEALED $REFS"
IN=patch-map-4b-inputs-2026-11-15
OUT=patch-map-4b-2026-11-15
EXPORT_DIR=sealed-4b-export-2026-09-23
FLAVOR=l40sx1
USD_H=1.80
TIMEOUT=${TIMEOUT:-40m}
EST="14 to 28 min, \$0.42 to \$0.84"
# The share the kill rules read, as `uv run ops/patch_map.py --instrument` decided after the stamp (prereg section 6).
PRIMARY=${PRIMARY:-cosine}
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
dest() {  # the name of an input in the bucket folder: contrast-map's raw files carry a prefix
  case "$1" in
    "$REF"/*) echo "contrast-map-$(basename "$1")" ;;
    *) basename "$1" ;;
  esac
}
remote_size() {  # size of $IN/<name> in the bucket, empty if absent
  hf buckets ls "hf://buckets/$BUCKET/$IN/" 2>/dev/null \
    | awk -v n="/$1" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true
}

if [ "${1:-}" = upload ]; then
  committed $INPUTS
  for f in $INPUTS; do hf buckets cp "$f" "hf://buckets/$BUCKET/$IN/$(dest "$f")"; done
  hf buckets ls "hf://buckets/$BUCKET/$IN/"
  exit 0
fi
[ -z "${1:-}" ] || refuse "usage: $0 [upload]"
case "$PRIMARY" in cosine|accuracy) ;; *) refuse "PRIMARY=$PRIMARY, cosine or accuracy";; esac

sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
PY_SHA=$(sha "$PY_LOCAL"); CM_SHA=$(sha "$CM_LOCAL"); LVO_SHA=$(sha "$LVO_LOCAL")
F16_SHA=$(sha "$DUMP_F16"); SEALED_SHA=$(sha "$DUMP_SEALED")
JOB_SHA=$(sha "$0")
# The scripts pin the dumps and contrast-map's raw files themselves; a mismatch here would fail inside the job after
# the card is billed.
grep -q "$F16_SHA" "$CM_LOCAL" && grep -q "$SEALED_SHA" "$CM_LOCAL" || refuse "the dumps are not the ones $CM_LOCAL pins"
REF_SUMS=""
for f in $REFS; do
  s=$(sha "$f")
  grep -q "$s" "$PY_LOCAL" || refuse "$f ($s) is not the file $PY_LOCAL pins"
  REF_SUMS="$REF_SUMS$s $(dest "$f")"$'\n'
done

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || refuse "$PREREG(.ots) missing"
  # The anchor is the proof: the banner a stamped prereg keeps is not read.
  ots info "$PREREG.ots" 2>/dev/null | grep -q "$(sha "$PREREG")" || refuse "$PREREG.ots does not anchor the sha256 of $PREREG"
  grep -q "$PY_SHA" "$PREREG" || refuse "$PREREG does not name the sha256 of $PY_LOCAL ($PY_SHA)"
  grep -q "$CM_SHA" "$PREREG" || refuse "$PREREG does not name the sha256 of $CM_LOCAL ($CM_SHA)"
  grep -q "$JOB_SHA" "$PREREG" || refuse "$PREREG does not name this script's sha256 ($JOB_SHA)"
  committed "$0" $INPUTS
  for f in $INPUTS; do
    L=$(stat -f %z "$f"); R=$(remote_size "$(dest "$f")")
    [ "$R" = "$L" ] || refuse "$IN/$(dest "$f") is ${R:-absent} B in the bucket, $L B here; run upload"
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
     "estimated $EST (setup about 4 min; pass 1 at most 6 min; pass 2 at most 16 min; analysis about 2 min); primary share $PRIMARY"
echo "inputs: $IN/ ($PY_LOCAL $PY_SHA, $CM_LOCAL $CM_SHA, $LVO_LOCAL $LVO_SHA, dumps $F16_SHA $SEALED_SHA," \
     "five contrast-map raw files); out $OUT/; job sha256 $JOB_SHA"

PRE=$(printf 'O=%q\nE=%q\nI=%q\nPY_SHA=%q\nCM_SHA=%q\nLVO_SHA=%q\nF16_SHA=%q\nSEALED_SHA=%q\nREF_SUMS=%q\nPINS=%q\nPRIMARY=%q' \
  "/out/$OUT" "/out/$EXPORT_DIR" "/out/$IN" "$PY_SHA" "$CM_SHA" "$LVO_SHA" "$F16_SHA" "$SEALED_SHA" "$REF_SUMS" "$PINS" "$PRIMARY")

IFS= read -r -d '' BODY <<'JOB' || true
export HF_HOME=/tmp/hf
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$O" /tmp/pm/ref
stamp() { echo "== $(date -u +%FT%TZ) $*" | tee -a "$O/timeline-job.txt"; }
stamp start
date -u +%FT%TZ > "$O/start.txt"
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt"
{ echo "nproc $(nproc)"; free -g 2>/dev/null | head -2 || true; } | tee -a "$O/gpu.txt"
echo '== the inputs on the mount: sha256 =='
{ echo "$PY_SHA  $I/patch_map.py"; echo "$CM_SHA  $I/contrast_map.py"; echo "$LVO_SHA  $I/layer_vs_output.py"
  echo "$F16_SHA  $I/mmlu-4b-f16-FULL.csv"; echo "$SEALED_SHA  $I/mmlu-4b-abl-control-FULL.csv"
  printf '%s' "$REF_SUMS" | while read -r s n; do echo "$s  $I/$n"; done; } | sha256sum -c - | tee "$O/inputs.sha256"
python -V | tee "$O/python.txt"
pip install --quiet --no-input $PINS
pip freeze > "$O/pip-freeze.txt"
cp "$I/patch_map.py" "$I/contrast_map.py" "$I/layer_vs_output.py" /tmp/pm/
for n in control questions matrices lens stream_bands; do cp "$I/contrast-map-$n.tsv" "/tmp/pm/ref/$n.tsv"; done
stamp selftests
python /tmp/pm/contrast_map.py --selftest 2>&1 | tee "$O/selftest-contrast-map.txt"
grep -q '^selftest: GREEN' "$O/selftest-contrast-map.txt" || { echo 'refused: contrast_map selftest is not green' >&2; exit 1; }
python /tmp/pm/patch_map.py --selftest 2>&1 | tee "$O/selftest.txt"
grep -q '^selftest: GREEN' "$O/selftest.txt" || { echo 'refused: patch_map selftest is not green' >&2; exit 1; }
stamp measurement
set +e
python /tmp/pm/patch_map.py --run --export "$E/model.safetensors" --out "$O" --stage /tmp/pm-export \
  --dump-f16 "$I/mmlu-4b-f16-FULL.csv" --dump-sealed "$I/mmlu-4b-abl-control-FULL.csv" --ref-dir /tmp/pm/ref \
  --primary "$PRIMARY" 2>&1 | tee "$O/run.txt"
rc=${PIPESTATUS[0]}
set -e
echo "exit $rc (0 green, 4 a gate red, 3 an exactness control or the bases broke, 5 stopped by the loop cap)" | tee "$O/exit.txt"
date -u +%FT%TZ > "$O/end.txt"
stamp end
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
