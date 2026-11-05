#!/usr/bin/env bash
# The k2 arm of the row-loop microbenchmark on the card: one decode for two activations (pers_trows.cu, TROWS_K2),
# the door to k tokens a pass, with every other row-loop arm in the same rounds (V2 is its base). Its own job, its own
# output directory: the first row-loop job (ops/jobs/pers-rowloop.sh, prereg stamped on integration/lot1 7f32775,
# job 6aec7403) ran without k2.
# Preregistered: proofs/preregistration-pers-rowloop-k2-2026-11-05.md, to stamp before this.
#
#   DRY_RUN=1 bash ops/jobs/pers-rowloop-k2.sh                 parses the job script, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/pers-rowloop-k2.sh      launches, after the operator's go
#
# One job, l40sx1, one process after the oracle: planesbench on the two banc-252 files with LLVQ_TROWS=1 only (no
# one-launch pass, which the first job carries). Cost, estimated (l40sx1 $1.80/h): about 9 min, $0.27. Timeout
# 30m: $0.90 at worst.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-pers-rowloop-k2-2026-11-05.md
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/pers-rowloop-k2-2026-11-05${RETRY:+-r$RETRY}
TIMEOUT=30m
DRY=${DRY_RUN:-0}

BALL="ball-ref-2026-09-20/qwen3-4b-llvq.bin:1770527533"
TETRA="tetra-4b-2026-09-06/qwen3-4b-tetra.bin:1770529149"
TETRA_SHA8=0adb7cfd
# The served unit's digest (llvq-cuda/src/pers.rs::the_served_unit_is_the_published_one): the row loops' reference
# is the served tv_tetra48_h, compiled from this unit.
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

PRE=$(printf 'O=%q\nB=%q\nF=%q\nBB=%q\nFB=%q\nFSHA8=%q\nSERVED_SHA=%q' \
  "$O" "/out/${BALL%%:*}" "/out/${TETRA%%:*}" "${BALL##*:}" "${TETRA##*:}" "$TETRA_SHA8" "$SERVED_SHA")

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

echo '== process 1: the row loops alone, k2 among them =='
date -u
rc=0
LLVQ_TILE_BLOCKS=64 LLVQ_BENCH_ARMS=slot32,planes14,fp16,tetra48 \
LLVQ_SEG_TETRA=1 LLVQ_TROWS=1 LLVQ_DUMP_CUBIN="$O/cubin" \
  planesbench "$B" "$F" > "$O/bench.txt" 2>&1 || rc=$?
date -u
echo "planesbench exit $rc"
grep -aE 'tetra48: [0-9]+ of [0-9]+|^tile |Row-loop|reference: the served|tv_trows|BIT FOR BIT|INVALIDATED|SPILLS|diagnostic|stamps d_in|One-launch|tpersall|speed over|second token|not timed|cubin ' "$O/bench.txt" | tail -90
[ "$rc" = 0 ]
grep -q 'tetra48: 252 of 252 matrices matched by name' "$O/bench.txt"
grep -q '^tile 64' "$O/bench.txt"
# gate 3: the reference is the served unit
grep -q "reference: the served unit, 101625 bytes, sha256 $SERVED_SHA" "$O/bench.txt" \
  || { echo "gate 3: the row loops' reference is not the served unit $SERVED_SHA" >&2; exit 1; }
# gate 4: v2 and v3 owe the bits and are the base of every ratio
grep -q '  v2: [0-9]* rows identical BIT FOR BIT to tv_tetra48_h' "$O/bench.txt"
grep -q '  v3: [0-9]* rows identical BIT FOR BIT to tv_tetra48_h' "$O/bench.txt"
# k2's identity is rule H (it voids k2 only); printed here for the reading
grep -E '^  k2[: ]' "$O/bench.txt" || true
ls -l "$O" "$O/cubin"
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
  --name "pers-rowloop-k2${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
