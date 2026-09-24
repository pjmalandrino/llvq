#!/usr/bin/env bash
# MMLU census, dense reconstruction, of the sealed 8B brought to ~2.7 b/param,
# arm A (o_proj + down_proj@15-20) or arm B (down_proj@10-26). Preregistered:
#   proofs/preregistration-sealed-8b-27-2026-09-24.md, stamped before this.
#
#   ARM=A|B bash ops/jobs/sealed-8b-27-mmlu.sh upload    the file into the bucket
#   ARM=A|B DRY_RUN=1 bash ops/jobs/sealed-8b-27-mmlu.sh prints the job script, launches nothing
#   ARM=A|B bash ops/jobs/sealed-8b-27-mmlu.sh           launches
#
# Cost, estimated: ~33 min, ~$0.98 an arm, timeout 60m ($1.80).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

ARM=${ARM:?ARM=A or ARM=B}
case "$ARM" in A|B) ;; *) echo "refused: ARM=$ARM, expected A or B" >&2; exit 1 ;; esac
SIZE=8b
TIMEOUT=60m
PREREG=proofs/preregistration-sealed-8b-27-2026-09-24.md
IMAGE_SHA=${IMAGE_SHA:-af90741654a5b28c51f6993c24c94993813220bc}
BUCKET=Pier-Jean/jobs-artifacts
NAME=qwen3-8b-sealed-$ARM.bin
LOCAL=${LOCAL:-$HOME/q8b-14b-sealed-2026-09-23/$NAME}
OBJ_DIR=sealed-8b27-2026-09-24
O=/out/sealed-8b27-$ARM-mmlu-2026-09-24
F=/out/$OBJ_DIR/$NAME
FP=a74a6d6213602979
DRY=${DRY_RUN:-0}

if [ "$DRY" = 1 ]; then
  SHA=0000000000000000000000000000000000000000000000000000000000000000
  BYTES=0
else
  test -f "$LOCAL" || { echo "refused: $LOCAL missing" >&2; exit 1; }
  SHA=$(shasum -a 256 "$LOCAL" | cut -d' ' -f1)
  BYTES=$(stat -f %z "$LOCAL")
fi

if [ "${1:-}" = upload ]; then
  hf buckets cp "$LOCAL" "hf://buckets/$BUCKET/$OBJ_DIR/$NAME"
  hf buckets ls "hf://buckets/$BUCKET/$OBJ_DIR/"
  exit 0
fi

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/$OBJ_DIR/" 2>/dev/null | awk -v n="/$NAME" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
  [ "$REMOTE" = "$BYTES" ] || { echo "refused: bucket copy is ${REMOTE:-absent} B, local $BYTES B" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space moved to $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "$SIZE sealed object: $BYTES B, sha256 $SHA; image $NOW"
fi

PRE=$(printf 'O=%q\nF=%q\nSHA=%q\nBYTES=%q\nFP=%q\nSIZE=%q' "$O" "$F" "$SHA" "$BYTES" "$FP" "8b-$ARM")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt" || true
echo '== oracle (hard rule 10) =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle-cuda.txt" | tail -2
grep -q MATCH "$O/oracle-cuda.txt"
echo '== the sealed file on the mount: bytes and sha256 against the Mac =='
test "$(stat -c %s "$F")" = "$BYTES"
sha256sum "$F" | tee "$O/files.sha256"
test "$(cut -d' ' -f1 "$O/files.sha256")" = "$SHA"
echo "== MMLU dense, FULL split, sealed $SIZE ==" ; date -u
LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP="$O/mmlu-$SIZE-sealed-FULL.csv" \
  mmlu "$F" cuda 2>&1 | tee "$O/out.txt" | tail -10
date -u
D="$O/mmlu-$SIZE-sealed-FULL.csv"
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
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; nothing launched" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "sealed-8b27-$ARM-mmlu" \
  "$PRE" "$BODY"
