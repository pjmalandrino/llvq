#!/usr/bin/env bash
# The two trained bases on a card, dense reconstruction, f16, wikitext-2 test,
# ctx 4096, 12 windows. Closes the device cross that left the cost of sealing
# measured at one size out of three. Preregistered:
#   proofs/preregistration-ppl-bases-carte-2026-10-05.md, sha256 f1abf54c,
#   stamped before this.
#
#   DRY_RUN=1 bash ops/jobs/ppl-bases-carte.sh   prints the job script, launches nothing
#   bash ops/jobs/ppl-bases-carte.sh             launches
#
# Both files are already in the bucket, from the row-scale campaigns of
# 2026-09-19 and 2026-09-21; this script uploads nothing. Cost, estimated:
# ~8 min, ~$0.25, timeout 30m ($0.90 at worst) under a $9 campaign cap of
# which $0.33 is spent.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-ppl-bases-carte-2026-10-05.md
IMAGE_SHA=${IMAGE_SHA:-97a2b62a6d0c8911dcd8e4d26a40461f0196ccc9}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/ppl-bases-carte-2026-10-05
FP=3f1baca9033bf251
DRY=${DRY_RUN:-0}

# size : bucket path : bytes : sha256 prefix of the TRAINED file
#
# The digest is not a courtesy. Folding row scales leaves the byte count
# unchanged, so dclm-4b-2026-09-18/qwen3-4b-dclm.bin (untrained) has the same
# 1,794,564,765 bytes as the trained file below. Only control 3 separates them.
FILES=(
  "4b:dclm-ft-2026-09-19/qwen3-4b-dclm-ft.bin:1794564765:f8c1c903"
  "8b:dclm-8b-ft-2026-09-21/qwen3-8b-dclm-ft.bin:4364205777:783cef67"
)

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  for row in "${FILES[@]}"; do
    IFS=: read -r _ path bytes _ <<<"$row"
    DIR=${path%/*}; NAME=${path##*/}
    REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/$DIR/" 2>/dev/null \
      | awk -v n="/$NAME" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
    [ "$REMOTE" = "$bytes" ] || { echo "refused: $path is ${REMOTE:-absent} B in the bucket, expected $bytes" >&2; exit 1; }
  done
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space moved to $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "two bucket copies verified byte for byte; image $NOW"
fi

PRE=$(printf 'O=%q\nFP=%q\nROWS=%q' "$O" "$FP" "${FILES[*]}")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt" || true
echo '== oracle (hard rule 10) =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle-cuda.txt" | tail -2
grep -q MATCH "$O/oracle-cuda.txt"
for row in $ROWS; do
  SIZE=${row%%:*}; rest=${row#*:}
  PATH_IN=${rest%%:*}; rest=${rest#*:}
  BYTES=${rest%%:*}; SHA8=${rest##*:}
  F=/out/$PATH_IN
  echo "== $SIZE base: $PATH_IN =="
  test "$(stat -c %s "$F")" = "$BYTES"
  sha256sum "$F" | tee -a "$O/files.sha256"
  # Control 3. The untrained base has the same size; only this line tells them
  # apart, so it refuses rather than warns.
  test "$(sha256sum "$F" | cut -c1-8)" = "$SHA8"
  date -u
  LLVQ_DTYPE=f16 ppl 4096 12 cuda "$F" 2>&1 | tee "$O/ppl-$SIZE-base.txt" | tail -4
  date -u
  grep -q "ctx 4096, 12 windows, dtype f16, kv f16, tokens $FP" "$O/ppl-$SIZE-base.txt"
  grep -E '^ppl = ' "$O/ppl-$SIZE-base.txt" | tee -a "$O/results.txt"
done
echo '== the two results =='
cat "$O/results.txt"
test "$(grep -c '^ppl = ' "$O/results.txt")" = 2
JOB

if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; nothing launched" >&2
  printf '%s\n' "$JOBSCRIPT"
  exit 0
fi

uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout 30m \
  --bucket "$BUCKET" --out-mount /out \
  --name "ppl-bases-carte" \
  "$PRE" "$BODY"
