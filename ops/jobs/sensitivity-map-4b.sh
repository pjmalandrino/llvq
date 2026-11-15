#!/usr/bin/env bash
# The sensitivity map: is FP16's sensitivity at depth 12 uneven over directions in a basis estimated in advance on
# other text, and does a band of it give back more than its share of the sealed 4B's deviation energy? FP16 Qwen3-4B
# and the sealed 4B's f16 export side by side on patch-map-4b's 570 prompts, plus 256 WikiText-2 windows and the 570
# following MMLU questions for the two Fisher bases.
# Preregistered: proofs/preregistration-sensitivity-map-4b-2026-11-15.md (the DRAFT renamed once signed), which must be
# stamped, and must name the sha256 of ops/sensitivity_map.py, ops/patch_map.py, ops/contrast_map.py and of this
# script, before the job launches.
#
# One Python process (ops/sensitivity_map.py, importing ops/patch_map.py, ops/contrast_map.py and
# ops/layer_vs_output.py, none edited) on the pytorch image of patch-map-4b. Pass 1: FP16 on the scored prompts (the
# variance basis). Text: the fisher_text gradients. Fisher: the fisher_mmlu gradients. Pass 2: FP16, the control, the
# deviation energies and the 15 arms, the first 57 prompts with the exactness controls. Gates, readings, the kill
# rule. Reads no model on the Mac.
#
# Inputs travel through the bucket folder $IN (the `upload` subcommand): the four scripts, the two Rust dumps, five
# raw files of contrast-map-4b and two of patch-map-4b, each checked by sha256 inside the job against the committed
# files. WikiText-2, MMLU and the checkpoint are fetched in the job at their pinned revisions, as before.
#
# Cost, estimated (prereg section 8): setup about 4 min (image and pip, three selftests, export copy and load, two
# base hashes), pass 1 about 1 min, text 1 to 2 min, Fisher 6 to 10 min, pass 2 7 to 9 min, analysis about 1 min:
# 19 to 28 min billed on l40sx1 at $1.80/h, $0.57 to $0.84. The script stops itself if a loop projects over its cap
# (pass 1 300 s, text 300 s, Fisher 900 s; every loop together 1,800 s, read after 114 prompts of pass 2).
# Timeout 45m, ceiling $1.35.
#
# Departures: the stamped prereg is never edited. When a script changes after the stamp, the departure file
# $ECARTS must exist, be stamped ($ECARTS.ots anchoring its sha256) and name the new sha256; the job then accepts a
# sha256 named in either file. The first launch (job 6af9be70) stopped before any model loaded and left
# $OUT/ behind: RETRY=2 writes to $OUT-r2/ and names the inputs folder and the job the same way.
#
#   RETRY=2 bash ops/jobs/sensitivity-map-4b.sh upload      the committed inputs into the bucket
#   DRY_RUN=1 RETRY=2 bash ops/jobs/sensitivity-map-4b.sh   prints the job script, parses it, launches nothing
#   RETRY=2 bash ops/jobs/sensitivity-map-4b.sh             launches
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-sensitivity-map-4b-2026-11-15.md
ECARTS=proofs/preregistration-sensitivity-map-4b-2026-11-15-ECARTS.md
BUCKET=Pier-Jean/jobs-artifacts
PY_LOCAL=ops/sensitivity_map.py
PM_LOCAL=ops/patch_map.py
CM_LOCAL=ops/contrast_map.py
LVO_LOCAL=ops/layer_vs_output.py
DUMP_F16=docs/data/mmlu-dumps/mmlu-4b-f16-FULL.csv
DUMP_SEALED=docs/data/mmlu-dumps/mmlu-4b-abl-control-FULL.csv
REF=docs/mesures/contrast-map-4b-2026-11-14-brut
REFS="$REF/control.tsv $REF/questions.tsv $REF/matrices.tsv $REF/lens.tsv $REF/stream_bands.tsv"
PMREF=docs/mesures/patch-map-4b-2026-11-15-brut
PMREFS="$PMREF/logits.tsv $PMREF/shape.tsv"
INPUTS="$PY_LOCAL $PM_LOCAL $CM_LOCAL $LVO_LOCAL $DUMP_F16 $DUMP_SEALED $REFS $PMREFS"
# RETRY=2 names a relaunch after a failed attempt left its directory behind.
RETRY=${RETRY:-}
case "$RETRY" in ""|[2-9]) ;; *) echo "refused: RETRY=$RETRY, empty or 2..9" >&2; exit 1;; esac
IN=sensitivity-map-4b-inputs-2026-11-15${RETRY:+-r$RETRY}
OUT=sensitivity-map-4b-2026-11-15${RETRY:+-r$RETRY}
EXPORT_DIR=sealed-4b-export-2026-09-23
FLAVOR=l40sx1
USD_H=1.80
TIMEOUT=${TIMEOUT:-45m}
EST="19 to 28 min, \$0.57 to \$0.84"
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
dest() {  # the name of an input in the bucket folder: the raw files carry their measurement's prefix
  case "$1" in
    "$REF"/*) echo "contrast-map-$(basename "$1")" ;;
    "$PMREF"/*) echo "patch-map-$(basename "$1")" ;;
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

sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
PY_SHA=$(sha "$PY_LOCAL"); PM_SHA=$(sha "$PM_LOCAL"); CM_SHA=$(sha "$CM_LOCAL"); LVO_SHA=$(sha "$LVO_LOCAL")
F16_SHA=$(sha "$DUMP_F16"); SEALED_SHA=$(sha "$DUMP_SEALED")
JOB_SHA=$(sha "$0")
# The scripts pin the dumps and the raw files themselves; a mismatch here would fail inside the job after the card is
# billed.
grep -q "$F16_SHA" "$CM_LOCAL" && grep -q "$SEALED_SHA" "$CM_LOCAL" || refuse "the dumps are not the ones $CM_LOCAL pins"
REF_SUMS=""
for f in $REFS; do
  s=$(sha "$f")
  grep -q "$s" "$PM_LOCAL" || refuse "$f ($s) is not the file $PM_LOCAL pins"
  REF_SUMS="$REF_SUMS$s $(dest "$f")"$'\n'
done
for f in $PMREFS; do
  s=$(sha "$f")
  grep -q "$s" "$PY_LOCAL" || refuse "$f ($s) is not the file $PY_LOCAL pins"
  REF_SUMS="$REF_SUMS$s $(dest "$f")"$'\n'
done

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || refuse "$PREREG(.ots) missing"
  # The anchor is the proof: the banner a stamped prereg keeps is not read.
  ots info "$PREREG.ots" 2>/dev/null | grep -q "$(sha "$PREREG")" || refuse "$PREREG.ots does not anchor the sha256 of $PREREG"
  NAMED="$PREREG"
  if [ -f "$ECARTS" ]; then
    test -f "$ECARTS.ots" || refuse "$ECARTS exists but $ECARTS.ots is missing"
    ots info "$ECARTS.ots" 2>/dev/null | grep -q "$(sha "$ECARTS")" || refuse "$ECARTS.ots does not anchor the sha256 of $ECARTS"
    NAMED="$PREREG $ECARTS"
    committed "$ECARTS" "$ECARTS.ots"
  fi
  for pair in "$PY_LOCAL:$PY_SHA" "$PM_LOCAL:$PM_SHA" "$CM_LOCAL:$CM_SHA" "$0:$JOB_SHA"; do
    # shellcheck disable=SC2086
    grep -q "${pair##*:}" $NAMED || refuse "neither $NAMED names the sha256 of ${pair%%:*} (${pair##*:})"
  done
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
     "estimated $EST (setup about 4 min; pass 1 about 1; text 1 to 2; Fisher 6 to 10; pass 2 7 to 9; analysis about 1)"
echo "inputs: $IN/ ($PY_LOCAL $PY_SHA, $PM_LOCAL $PM_SHA, $CM_LOCAL $CM_SHA, $LVO_LOCAL $LVO_SHA, dumps $F16_SHA" \
     "$SEALED_SHA, five contrast-map and two patch-map raw files); out $OUT/; job sha256 $JOB_SHA"

PRE=$(printf 'O=%q\nE=%q\nI=%q\nPY_SHA=%q\nPM_SHA=%q\nCM_SHA=%q\nLVO_SHA=%q\nF16_SHA=%q\nSEALED_SHA=%q\nREF_SUMS=%q\nPINS=%q' \
  "/out/$OUT" "/out/$EXPORT_DIR" "/out/$IN" "$PY_SHA" "$PM_SHA" "$CM_SHA" "$LVO_SHA" "$F16_SHA" "$SEALED_SHA" "$REF_SUMS" "$PINS")

IFS= read -r -d '' BODY <<'JOB' || true
export HF_HOME=/tmp/hf
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$O" /tmp/sm/ref /tmp/sm/pmref
stamp() { echo "== $(date -u +%FT%TZ) $*" | tee -a "$O/timeline-job.txt"; }
stamp start
date -u +%FT%TZ > "$O/start.txt"
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt"
{ echo "nproc $(nproc)"; free -g 2>/dev/null | head -2 || true; } | tee -a "$O/gpu.txt"
echo '== the inputs on the mount: sha256 =='
{ echo "$PY_SHA  $I/sensitivity_map.py"; echo "$PM_SHA  $I/patch_map.py"; echo "$CM_SHA  $I/contrast_map.py"
  echo "$LVO_SHA  $I/layer_vs_output.py"
  echo "$F16_SHA  $I/mmlu-4b-f16-FULL.csv"; echo "$SEALED_SHA  $I/mmlu-4b-abl-control-FULL.csv"
  printf '%s' "$REF_SUMS" | while read -r s n; do echo "$s  $I/$n"; done; } | sha256sum -c - | tee "$O/inputs.sha256"
python -V | tee "$O/python.txt"
pip install --quiet --no-input $PINS
pip freeze > "$O/pip-freeze.txt"
cp "$I/sensitivity_map.py" "$I/patch_map.py" "$I/contrast_map.py" "$I/layer_vs_output.py" /tmp/sm/
for n in control questions matrices lens stream_bands; do cp "$I/contrast-map-$n.tsv" "/tmp/sm/ref/$n.tsv"; done
for n in logits shape; do cp "$I/patch-map-$n.tsv" "/tmp/sm/pmref/$n.tsv"; done
stamp selftests
for s in contrast_map patch_map sensitivity_map; do
  python "/tmp/sm/$s.py" --selftest 2>&1 | tee "$O/selftest-$s.txt"
  grep -q '^selftest: GREEN' "$O/selftest-$s.txt" || { echo "refused: $s selftest is not green" >&2; exit 1; }
done
stamp measurement
set +e
python /tmp/sm/sensitivity_map.py --run --export "$E/model.safetensors" --out "$O" --stage /tmp/sm-export \
  --dump-f16 "$I/mmlu-4b-f16-FULL.csv" --dump-sealed "$I/mmlu-4b-abl-control-FULL.csv" --ref-dir /tmp/sm/ref \
  --pm-ref-dir /tmp/sm/pmref 2>&1 | tee "$O/run.txt"
rc=${PIPESTATUS[0]}
set -e
echo "exit $rc (0 green, 4 a gate red, 3 an exactness control or the bases broke, 5 stopped by a loop cap)" | tee "$O/exit.txt"
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
