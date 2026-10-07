#!/usr/bin/env bash
# The ten-arm bench and the tile sweep with Tetra on all 252 matrices. Preregistered:
#   proofs/preregistration-banc-252-2026-10-06.md
#
# banc-tetra.sh and tuile-l40s.sh with one change: the second file is the bare
# Tetra encode, 252 lattice records and no int4, so every arm times the same
# 252 matrices.
#
# Cost: about $1.05 on l40sx1 (estimated), timeout 1 h 30, $2.70 at worst.
# DRY_RUN=1 prints the job script and checks that it parses; nothing launches.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-banc-252-2026-10-06.md
IMAGE_SHA=${IMAGE_SHA:-97a2b62a6d0c8911dcd8e4d26a40461f0196ccc9}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/banc-252-2026-10-06
DRY=${DRY_RUN:-0}

# path in the bucket : bytes
BALL="ball-ref-2026-09-20/qwen3-4b-llvq.bin:1770527533"
TETRA="tetra-4b-2026-09-06/qwen3-4b-tetra.bin:1770529149"
TETRA_SHA8=0adb7cfd

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  for row in "$BALL" "$TETRA"; do
    path=${row%%:*}; bytes=${row##*:}
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
  echo "two bucket copies verified by size; image $NOW"
fi

PRE=$(printf 'O=%q\nB=%q\nF=%q\nBB=%q\nFB=%q\nSHA8=%q' "$O" \
  "/out/${BALL%%:*}" "/out/${TETRA%%:*}" "${BALL##*:}" "${TETRA##*:}" "$TETRA_SHA8")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt" || true
# Gate 1: bytes, and the digest that names the bare Tetra file.
test "$(stat -c %s "$B")" = "$BB"
test "$(stat -c %s "$F")" = "$FB"
sha256sum "$B" "$F" | tee "$O/files.sha256"
test "$(sha256sum "$F" | cut -c1-8)" = "$SHA8"

P1=slot32,planes14,planes12x,golay70v1,fp16,awq,golay70v2,cublasf16,nullk
echo '== the ten arms at the served tile, phase 2 adds tetra48 =='
date -u
LLVQ_BENCH_ARMS="$P1;$P1,tetra48" planesbench "$B" "$F" 2>&1 | tee "$O/banc.txt" | tail -45
date -u
# Gates 2 and 3.
grep -q 'tetra48: 252 of 252 matrices matched by name' "$O/banc.txt"
grep -qE '^tile 64 ' "$O/banc.txt"

for T in 128 64 32; do
  echo "== sweep, tile $T =="
  date -u
  LLVQ_TILE_BLOCKS=$T LLVQ_BENCH_ARMS=fp16,planes14,nullk,tetra48 planesbench "$B" "$F" 2>&1 \
    | tee "$O/sweep-$T.txt" | grep -aE 'tile |Tetra48|Planes14|floor \(nullk\)'
  grep -q 'tetra48: 252 of 252 matrices matched by name' "$O/sweep-$T.txt"
done
date -u
echo '== raw kept =='
wc -l "$O"/*.txt
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
  --flavor l40sx1 --timeout 90m \
  --bucket "$BUCKET" --out-mount /out \
  --name "banc-252" \
  "$PRE" "$BODY"
