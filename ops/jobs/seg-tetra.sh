#!/usr/bin/env bash
# The first card check of tv_tetra48_seg against the unfused Tetra launches, once the fused stream comes
# from the Tetra file (commit c68ba42). Preregistered: proofs/preregistration-seg-tetra-2026-11-04.md,
# stamped before this.
#
#   DRY_RUN=1 bash ops/jobs/seg-tetra.sh                 prints and parses the job script
#   IMAGE_SHA=<space sha> bash ops/jobs/seg-tetra.sh      launches, after the operator's go
#
# The two bench files of banc-252 (ball reference, all-Tetra 4B: 252 of 252 matrices matched by name, no
# int4 record, so every q+k+v and gate+up group has a Tetra stream). One planesbench process, the four
# arms the fusion section needs (slot32 and planes14 switch it on, fp16 is the witness no ratio may
# lose), tetra48, and LLVQ_SEG_TETRA=1.
#
# Cost, estimated (l40sx1 $1.80/h): about 20 min, $0.60 (banc-252 spent 1,495 s proving ten arms). Ceiling 45m, $1.35.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-seg-tetra-2026-11-04.md
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/seg-tetra-2026-11-04${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}

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
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of the rebuild that carries c68ba42>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "two bucket copies verified by size; image $NOW"
fi

PRE=$(printf 'O=%q\nB=%q\nF=%q\nBB=%q\nFB=%q\nSHA8=%q' "$O" \
  "/out/${BALL%%:*}" "/out/${TETRA%%:*}" "${BALL##*:}" "${TETRA##*:}" "$TETRA_SHA8")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt"
test "$(stat -c %s "$B")" = "$BB"
test "$(stat -c %s "$F")" = "$FB"
sha256sum "$B" "$F" | tee "$O/files.sha256"
test "$(sha256sum "$F" | cut -c1-8)" = "$SHA8"
echo '== fusion section, Tetra included =='
date -u
rc=0
LLVQ_SEG_TETRA=1 LLVQ_BENCH_ARMS=slot32,planes14,fp16,tetra48 planesbench "$B" "$F" > "$O/banc.txt" 2>&1 || rc=$?
date -u
echo "planesbench exit $rc"
grep -aE 'tetra48: [0-9]+ of [0-9]+ matrices|Fusion \(A4\)|groups|bit-exact|fused against|Tetra fusion|UNPROVEN|tile ' "$O/banc.txt" | tail -20
grep -q 'tetra48: 252 of 252 matrices matched by name' "$O/banc.txt"
echo '== raw kept ==' ; wc -l "$O/banc.txt"
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
  --flavor l40sx1 --timeout 45m \
  --bucket "$BUCKET" --out-mount /out \
  --name "seg-tetra${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
