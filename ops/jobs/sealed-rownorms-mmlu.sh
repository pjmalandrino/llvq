#!/usr/bin/env bash
# MMLU census, dense reconstruction, of the sealed 4B after its second training
# (row scales + norms), folded on the Mac. Preregistered:
#   proofs/preregistration-sealed-rownorms-2026-09-23.md
# Paired against the sealed base's census dump (63.37).
#
# Cost, estimated: 24 min on l40sx1, ~$0.72; timeout 60m, ceiling $1.80.
#
#   bash ops/jobs/sealed-rownorms-mmlu.sh upload    copy the folded file into the bucket
#   DRY_RUN=1 bash ops/jobs/sealed-rownorms-mmlu.sh prints the job script, parses it, launches nothing
#   bash ops/jobs/sealed-rownorms-mmlu.sh           launches
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-sealed-rownorms-2026-09-23.md
IMAGE_SHA=${IMAGE_SHA:-af90741654a5b28c51f6993c24c94993813220bc}
BUCKET=Pier-Jean/jobs-artifacts
NAME=qwen3-4b-sealed-rownorms.bin
LOCAL=${LOCAL:-$HOME/q4b-sealed-2026-09-23/$NAME}
OBJ_DIR=sealed-rownorms-2026-09-23                   # the training's own directory
O=/out/sealed-rownorms-mmlu-2026-09-23
F=/out/$OBJ_DIR/$NAME
BYTES=1418224685                                     # the fold moves no byte count
FP=a74a6d6213602979
DRY=${DRY_RUN:-0}

if [ "$DRY" = 1 ]; then
  SHA=0000000000000000000000000000000000000000000000000000000000000000
else
  test -f "$LOCAL" || { echo "refused: $LOCAL missing, fold first" >&2; exit 1; }
  [ "$(stat -f %z "$LOCAL")" = "$BYTES" ] || { echo "refused: $LOCAL is not $BYTES B" >&2; exit 1; }
  SHA=$(shasum -a 256 "$LOCAL" | cut -d' ' -f1)
fi

if [ "${1:-}" = upload ]; then
  hf buckets cp "$LOCAL" "hf://buckets/$BUCKET/$OBJ_DIR/$NAME"
  hf buckets ls "hf://buckets/$BUCKET/$OBJ_DIR/"
  exit 0
fi

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/$OBJ_DIR/" 2>/dev/null | awk -v n="/$NAME" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
  [ "$REMOTE" = "$BYTES" ] || { echo "refused: bucket copy is ${REMOTE:-absent} B" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space moved to $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "folded object: $BYTES B, sha256 $SHA; image $NOW"
fi

PRE=$(printf 'O=%q\nF=%q\nSHA=%q\nBYTES=%q\nFP=%q' "$O" "$F" "$SHA" "$BYTES" "$FP")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt" || true
echo '== oracle (hard rule 10) =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle-cuda.txt" | tail -2
grep -q MATCH "$O/oracle-cuda.txt"
echo '== the folded file on the mount: bytes and sha256 against the Mac =='
test "$(stat -c %s "$F")" = "$BYTES"
sha256sum "$F" | tee "$O/files.sha256"
test "$(cut -d' ' -f1 "$O/files.sha256")" = "$SHA"
echo '== MMLU dense, FULL split ==' ; date -u
LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP="$O/mmlu-4b-sealed-rownorms-FULL.csv" \
  mmlu "$F" cuda 2>&1 | tee "$O/out.txt" | tail -10
date -u
D="$O/mmlu-4b-sealed-rownorms-FULL.csv"
grep -qxF '# dtype=f16' "$D"
grep -qxF '# limit=census' "$D"
grep -qxF '# alloc=flat, 100..1534 per subject, 14042 questions' "$D"
grep -qxF '# config=none' "$D"
grep -qxF '# arithmetic=dense reconstruction' "$D"
grep -qxF '# kv=f16' "$D"
test "$(tail -n 1 "$D")" = "# end fingerprint=$FP questions=14042"
echo "dump ok: $D"
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
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout 60m \
  --bucket "$BUCKET" --out-mount /out \
  --name sealed-rownorms-mmlu \
  "$PRE" "$BODY"
