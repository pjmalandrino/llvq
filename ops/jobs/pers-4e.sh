#!/usr/bin/env bash
# 4E's first card reading: the whole decode token in one cooperative launch on the sealed 4B, its tokens
# against the eager path's (kernel attention form, every decode row on the kernel), then decode tok/s at 5
# and 8,192 prompt tokens. Preregistered: proofs/preregistration-pers-4e-2026-11-04.md, to stamp before this.
#
#   DRY_RUN=1 bash ops/jobs/pers-4e.sh                 prints and parses the job script
#   IMAGE_SHA=<space sha> bash ops/jobs/pers-4e.sh      launches, after the operator's go
#
# One job, l40sx1, one fusedrun process after the oracle. Cost, estimated (l40sx1 $1.80/h): about 10 min,
# $0.30 (two prefills of 8,192 tokens a context, 14 decodes of 63 tokens a context). Timeout 40m: $1.20 at
# worst.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-pers-4e-2026-11-04.md
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/pers-4e-2026-11-04${RETRY:+-r$RETRY}
TIMEOUT=40m
DRY=${DRY_RUN:-0}

SEALED="sealed-4b-2026-09-23/qwen3-4b-sealed.bin:1418224685"
SEALED_SHA8=886391a8
if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  for row in "$SEALED"; do
    path=${row%%:*}; bytes=${row##*:}
    DIR=${path%/*}; NAME=${path##*/}
    REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/$DIR/" 2>/dev/null \
      | awk -v n="/$NAME" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
    [ "$REMOTE" = "$bytes" ] || { echo "refused: $path is ${REMOTE:-absent} B in the bucket, expected $bytes" >&2; exit 1; }
  done
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of the rebuild that carries this branch>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "the sealed 4B verified by size; image $NOW"
fi

PRE=$(printf 'O=%q\nS=%q\nSB=%q\nSSHA8=%q' "$O" "/out/${SEALED%%:*}" "${SEALED##*:}" "$SEALED_SHA8")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
test "$(stat -c %s "$S")" = "$SB"
sha256sum "$S" | tee "$O/files.sha256"
test "$(sha256sum "$S" | cut -c1-8)" = "$SSHA8"

echo '== gate 1: oracle =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle.txt" | tail -3
grep -q 'MATCH' "$O/oracle.txt"

echo '== 4E against the eager path, 5 and 8,192 prompt tokens =='
date -u
rc=0
LLVQ_FUSED_LAYOUT=tetra48 LLVQ_EMBED=q4 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_PERS_4E=64 \
  fusedrun "$S" 64 > "$O/pers-4e.txt" 2>&1 || rc=$?
date -u
echo "fusedrun exit $rc"
grep -aE '^4E:|cudarc event|context |tokens equal|served grouped|tok/s|4E / eager' "$O/pers-4e.txt" | tail -40
# gates 2 to 4: each line from where it is printed
grep -q 'cudarc event tracking: off (one stream, ordered by issue)' "$O/pers-4e.txt"
grep -q '^4E: RoPE forms equal to candle' "$O/pers-4e.txt"
test "$(grep -c '4E tokens equal the eager path' "$O/pers-4e.txt")" = 2
ls -l "$O"
exit $rc
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
  --name "pers-4e${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
