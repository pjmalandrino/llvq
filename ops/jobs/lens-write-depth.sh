#!/usr/bin/env bash
# Where an ORIGINAL Qwen3 writes its MMLU answer: ops/contrast_map.py --lens-only on one FP16 checkpoint, no
# quantized file. Preregistered: proofs/preregistration-lens-write-depth-2026-11-14.md, which must be stamped, and
# must name the sha256 of ops/contrast_map.py and of this script, before any size launches.
#
# One Python process on the pytorch image of the 2026-11-14 contrast map: the checkpoint at its pin in ORIGINALS
# (every shard checked by size and sha256), N MMLU 5-shot prompts round robin over the 57 subjects, N fixed from
# the time of the first 57 (at least 570, at most 2,280, budget 300 s); rule D against the Rust FP16 census dump of
# that size (picks equal on at least 98 %); lens.tsv, write.tsv, lens_logits.tsv, three stream dumps.
#
# Inputs travel through the bucket folder $IN (the `upload` subcommand): the script, layer_vs_output.py and the three
# FP16 dumps, each checked by sha256 inside the job against the committed files.
#
# Cost, estimated per size on l40sx1 at $1.80/h (setup = image, pip, checkpoint download, shard hashes, load):
#   4b   setup 4 to 5 min, loop at most 300 s, report 1 min: 9 to 12 min, $0.27 to $0.36. Timeout 20m, $0.60.
#   8b   setup 5 to 7 min, loop 300 s, report 1 min: 11 to 14 min, $0.33 to $0.42. Timeout 25m, $0.75.
#   14b  setup 6 to 9 min, loop 300 s, report 1 min: 12 to 16 min, $0.36 to $0.48. Timeout 30m, $0.90.
#
#   bash ops/jobs/lens-write-depth.sh upload             the committed inputs into the bucket
#   DRY_RUN=1 bash ops/jobs/lens-write-depth.sh <size>   prints the job script, parses it, launches nothing
#   bash ops/jobs/lens-write-depth.sh <size>             launches one size: 4b, 8b or 14b
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-lens-write-depth-2026-11-14.md
BUCKET=Pier-Jean/jobs-artifacts
PY_LOCAL=ops/contrast_map.py
LVO_LOCAL=ops/layer_vs_output.py
D=docs/data/mmlu-dumps
INPUTS="$PY_LOCAL $LVO_LOCAL $D/mmlu-4b-f16-FULL.csv $D/mmlu-8b-f16-FULL.csv $D/mmlu-14b-f16-FULL.csv"
IN=lens-write-depth-inputs-2026-11-14
FLAVOR=l40sx1
USD_H=1.80
BUDGET=${BUDGET:-300}
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
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }

if [ "${1:-}" = upload ]; then
  committed $INPUTS
  for f in $INPUTS; do hf buckets cp "$f" "hf://buckets/$BUCKET/$IN/$(basename "$f")"; done
  hf buckets ls "hf://buckets/$BUCKET/$IN/"
  exit 0
fi

SIZE=${1:-}
# The revisions are ORIGINALS in ops/contrast_map.py; the script refuses any other.
case "$SIZE" in
  4b)  MODEL=Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c;  TIMEOUT=${TIMEOUT:-20m}; EST="9 to 12 min, \$0.27 to \$0.36" ;;
  8b)  MODEL=Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218;  TIMEOUT=${TIMEOUT:-25m}; EST="11 to 14 min, \$0.33 to \$0.42" ;;
  14b) MODEL=Qwen/Qwen3-14B@40c069824f4251a91eefaf281ebe4c544efd3e18; TIMEOUT=${TIMEOUT:-30m}; EST="12 to 16 min, \$0.36 to \$0.48" ;;
  *) refuse "usage: $0 upload | $0 4b|8b|14b" ;;
esac
DUMP=mmlu-$SIZE-f16-FULL.csv
OUT=lens-write-depth-$SIZE-2026-11-14

PY_SHA=$(sha "$PY_LOCAL"); LVO_SHA=$(sha "$LVO_LOCAL"); DUMP_SHA=$(sha "$D/$DUMP"); JOB_SHA=$(sha "$0")
grep -q "${MODEL#*@}" "$PY_LOCAL" || refuse "$PY_LOCAL does not pin ${MODEL#*@}"
grep -q "$DUMP_SHA" "$PY_LOCAL" || refuse "$D/$DUMP is not the dump $PY_LOCAL pins"

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
  if hf buckets ls "hf://buckets/$BUCKET/$OUT/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    refuse "$OUT/ already exists in the bucket"
  fi
fi

MIN=${TIMEOUT%m}
echo "cost: $FLAVOR at \$$USD_H/h, timeout $TIMEOUT, ceiling \$$(awk -v m="$MIN" -v r="$USD_H" 'BEGIN {printf "%.2f", m / 60 * r}');" \
     "estimated $EST (setup, loop under --budget-seconds $BUDGET, report)"
echo "size $SIZE: $MODEL; inputs $IN/ ($PY_LOCAL $PY_SHA, $LVO_LOCAL $LVO_SHA, $DUMP $DUMP_SHA); out $OUT/; job sha256 $JOB_SHA"

PRE=$(printf 'O=%q\nI=%q\nMODEL=%q\nDUMP=%q\nPY_SHA=%q\nLVO_SHA=%q\nDUMP_SHA=%q\nBUDGET=%q\nPINS=%q' \
  "/out/$OUT" "/out/$IN" "$MODEL" "$DUMP" "$PY_SHA" "$LVO_SHA" "$DUMP_SHA" "$BUDGET" "$PINS")

IFS= read -r -d '' BODY <<'JOB' || true
export HF_HOME=/tmp/hf
mkdir -p "$O" /tmp/cm
date -u +%FT%TZ | tee "$O/start.txt"
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt"
{ echo "nproc $(nproc)"; df -h /tmp 2>&1 | tail -n +2 || true; } | tee -a "$O/gpu.txt"
echo '== the inputs on the mount: sha256 =='
{ echo "$PY_SHA  $I/contrast_map.py"; echo "$LVO_SHA  $I/layer_vs_output.py"; echo "$DUMP_SHA  $I/$DUMP"; } \
  | sha256sum -c - | tee "$O/inputs.sha256"
python -V | tee "$O/python.txt"
pip install --quiet --no-input $PINS
pip freeze > "$O/pip-freeze.txt"
cp "$I/contrast_map.py" "$I/layer_vs_output.py" /tmp/cm/
echo '== selftest, tiny model, CPU =='
python /tmp/cm/contrast_map.py --selftest 2>&1 | tee "$O/selftest.txt"
grep -q '^selftest: GREEN' "$O/selftest.txt" || { echo 'refused: selftest is not green' >&2; exit 1; }
echo "== the measurement: $MODEL =="
set +e
python /tmp/cm/contrast_map.py --lens-only --model "$MODEL" --out "$O" --dump-f16 "$I/$DUMP" \
  --budget-seconds "$BUDGET" 2>&1 | tee "$O/run.txt"
rc=${PIPESTATUS[0]}
set -e
echo "exit $rc (0 green, 4 rule D red)" | tee "$O/exit.txt"
date -u +%FT%TZ | tee "$O/end.txt"
echo '== what landed ==' ; ls -la "$O"
exit "$rc"
JOB

if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: size $SIZE parses; $FLAVOR, timeout $TIMEOUT; nothing launched" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime \
  --flavor "$FLAVOR" --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "$OUT" \
  "$PRE" "$BODY"
