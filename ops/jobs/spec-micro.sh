#!/usr/bin/env bash
# The verify rows of a speculative round with B1's integer dot, and B1's levers, alone on the card
# (docs/recherche-debit-maths.md §8 and §11, the first gate of the speculative route).
#   v2       4E's Tetra row, the base of every ratio
#   k2       one decode, two f32 activations: shaker-4e's second row, replicated
#   pword    the words read, nothing decoded: the stream rate of Tetra words at 4E's geometry
#   lean     the relabeled word, Golay patterns from two tables by linearity; the served bits
#   rot      lane rotation; the served bits
#   i16      lean with the activation in int16 and the dot in dp2a (B1's row)
#   i16kK    i16 with one decode for K = 2, 3, 4 activations: a verify pass of K rows
# Preregistered: proofs/preregistration-spec-micro-2026-11-08.md, stamped before this.
#
#   DRY_RUN=1 bash ops/jobs/spec-micro.sh                 parses the job script, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/spec-micro.sh      launches, after the operator's go
#
# One job, l40sx1, one process after the oracle: planesbench on the two banc-252 files, step1-micro's command with
# LLVQ_VROWS in place of LLVQ_TROWS and LLVQ_TETRA_OCC. The image must carry this branch: a Space rebuild first
# (ops/run.py publish, no card). Cost, estimated (l40sx1 $1.80/h): about 9 min, $0.27. Timeout 20m: $0.60 at worst,
# the operator's cap.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-spec-micro-2026-11-08.md
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/spec-micro-2026-11-08${RETRY:+-r$RETRY}
TIMEOUT=20m
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

echo '== process 1: the verify rows and B1 levers, interleaved with A4 =='
date -u
rc=0
LLVQ_TILE_BLOCKS=64 LLVQ_BENCH_ARMS=slot32,planes14,fp16,tetra48 \
LLVQ_SEG_TETRA=1 LLVQ_VROWS=1 LLVQ_DUMP_CUBIN="$O/cubin" \
  planesbench "$B" "$F" > "$O/bench.txt" 2>&1 || rc=$?
date -u
echo "planesbench exit $rc"
grep -aE 'tetra48: [0-9]+ of [0-9]+|^tile |Verify-row|reference: the served|tv_vrows|lean tables|BIT FOR BIT|INVALIDATED|SPILLS|diagnostic|against tv_tetra48_h|extra row|second f32 row|verify gate|in situ|speed over|not timed|cubin ' "$O/bench.txt" | tail -80
[ "$rc" = 0 ]
grep -q 'tetra48: 252 of 252 matrices matched by name' "$O/bench.txt"
grep -q '^tile 64' "$O/bench.txt"
# gate 3: the reference is the served unit
grep -q "reference: the served unit, 101625 bytes, sha256 $SERVED_SHA" "$O/bench.txt" \
  || { echo "gate 3: the verify rows' reference is not the served unit $SERVED_SHA" >&2; exit 1; }
# gate 4: the base of every ratio keeps the bits on each activation
for a in 0 1 2 3; do
  grep -q "  v2: [0-9]* rows identical BIT FOR BIT to tv_tetra48_h on activation $a" "$O/bench.txt" \
    || { echo "gate 4: v2 does not keep the served bits on activation $a" >&2; exit 1; }
done
# gate 5: GA ^ GB is the trellis on every path
grep -q 'differs from the trellis on 0 of 4,096 paths' "$O/bench.txt" \
  || { echo "gate 5: the lean tables are not the trellis" >&2; exit 1; }
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
  --name "spec-micro${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
