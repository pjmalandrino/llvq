#!/usr/bin/env bash
# Step L2 of the persistent-kernel lead: the fifteen persistent Tetra arms in route P's geometry (one CTA
# a SM), against the served unit's tv_tetra48_h and A4's fused Tetra geometry in the same rounds.
# Preregistered: proofs/preregistration-pers-l2-2026-11-04.md, to stamp before this. It runs only after
# step L1 and the operator's decision 8 (the four settings frozen on 2026-08-05, prereg §0).
#
#   DRY_RUN=1 bash ops/jobs/pers-l2.sh                 prints and parses the job script
#   IMAGE_SHA=<space sha> bash ops/jobs/pers-l2.sh      launches, after the operator's go
#
# One job, l40sx1, one planesbench process on the two banc-252 files: five table arms (control), A4
# with the Tetra fusion, tetra48h and the fifteen arms, every arm checked before it is timed. Three of them
# (pdec_l1, pdec_smem, pl1_t28) read what bounds the stream at 410 GB/s (docs/mesures/pers-sass-2026-11-05.txt).
#
# Cost, estimated (l40sx1 $1.80/h): about 11 min, $0.33. Timeout 90m (five table arms, METHODE §6):
# $2.70 at worst.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-pers-l2-2026-11-04.md
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/pers-l2-2026-11-04${RETRY:+-r$RETRY}
TIMEOUT=90m
DRY=${DRY_RUN:-0}

BALL="ball-ref-2026-09-20/qwen3-4b-llvq.bin:1770527533"
TETRA="tetra-4b-2026-09-06/qwen3-4b-tetra.bin:1770529149"
TETRA_SHA8=0adb7cfd
# The served unit's digest, computed from this tree by the test that pins it
# (llvq-cuda/src/pers.rs::the_served_unit_is_the_published_one).
SERVED_SHA=6193094310bc18acdb0a0c554807ff45f6df09c1f6ab49e0577032874f107812

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
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of the rebuild that carries this branch>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "two bucket copies verified by size; image $NOW"
fi

ARMS=pnull,pword,pdec_l1,pdec_smem,pl1,pl1_t28,psmem,psmem_s25,psmem_t28,psmem_t28_u2,pt28_512,pt28_1024,pchain_w,pchain_w2,pchain_wp
PRE=$(printf 'O=%q\nB=%q\nF=%q\nBB=%q\nFB=%q\nFSHA8=%q\nSERVED_SHA=%q\nARMS=%q' \
  "$O" "/out/${BALL%%:*}" "/out/${TETRA%%:*}" "${BALL##*:}" "${TETRA##*:}" "$TETRA_SHA8" "$SERVED_SHA" "$ARMS")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O" "$O/cubin"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
test "$(stat -c %s "$B")" = "$BB"
test "$(stat -c %s "$F")" = "$FB"
sha256sum "$B" "$F" | tee "$O/files.sha256"
test "$(sha256sum "$F" | cut -c1-8)" = "$FSHA8"

echo '== gate 1: oracle =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle.txt" | tail -3
grep -q 'MATCH' "$O/oracle.txt"

echo '== the bench =='
date -u
rc=0
LLVQ_TILE_BLOCKS=64 LLVQ_BENCH_ARMS=slot32,planes14,fp16,nullk,tetra48 \
LLVQ_SEG_TETRA=1 LLVQ_PERS_ARMS="$ARMS" LLVQ_DUMP_CUBIN="$O/cubin" \
  planesbench "$B" "$F" > "$O/bench.txt" 2>&1 || rc=$?
date -u
echo "planesbench exit $rc"
sed -n '/Persistent arms (L2)/,$p' "$O/bench.txt" | head -80
grep -q 'tetra48: 252 of 252 matrices matched by name' "$O/bench.txt"
grep -q '^tile 64' "$O/bench.txt"
grep -q "served unit: 101625 bytes, sha256 $SERVED_SHA" "$O/bench.txt"
ls -l "$O" "$O/cubin"
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
  --name "pers-l2${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
