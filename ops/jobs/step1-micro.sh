#!/usr/bin/env bash
# plan-300 step 1 (docs/plan-300.md §6): one job, three microbenchmarks at 4E's geometry, nothing of 4E timed.
#   K4   the Tetra row loop: v2 against five arms that each change one thing (read-only loads, the same with a
#        256-byte L2 prefetch, a cp.async ring, a 128-bit tile layout, rows claimed dynamically); the nine arms of
#        pers-rowloop ride along, v2 replicates its 379 GB/s.
#   K11  the int4 rows alone: the served tv_q4_hh (84 launches) against 4E's row in one launch, its read-only
#        loads, its tile layout, and a read-only diagnostic, over 84 synthetic records of the 4B's int4 shapes.
#   K3b  barriers against counters: pers-l1's barrier, the same without back-off, release/acquire counters over
#        every CTA and over 8, and an arrival split from its wait by 1 us of work.
# Preregistered: proofs/preregistration-step1-micro-2026-11-07.md, to stamp before this.
#
#   DRY_RUN=1 bash ops/jobs/step1-micro.sh                 parses the job script, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/step1-micro.sh      launches, after the operator's go
#
# One job, l40sx1, one process after the oracle: planesbench on the two banc-252 files, pers-rowloop's command.
# The image must carry this branch: a Space rebuild first (ops/run.py publish, no card).
# Cost, estimated (l40sx1 $1.80/h): about 11 min running, $0.33. pers-rowloop ran 497 s ($0.25); this adds five
# row-loop arms (a few ms a round each), the K11 setup (0.41 GB generated and uploaded, 84 reference launches,
# about 30 s), the counters (6 kernels x 7 reps x 10,000 hops, under 2 s) and a larger module to compile.
# Timeout 30m: $0.90 at worst.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-step1-micro-2026-11-07.md
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/step1-micro-2026-11-07${RETRY:+-r$RETRY}
TIMEOUT=30m
DRY=${DRY_RUN:-0}

BALL="ball-ref-2026-09-20/qwen3-4b-llvq.bin:1770527533"
TETRA="tetra-4b-2026-09-06/qwen3-4b-tetra.bin:1770529149"
TETRA_SHA8=0adb7cfd
# The served unit's digest (llvq-cuda/src/pers.rs::the_served_unit_is_the_published_one): the row loops' reference
# is the served tv_tetra48_h, K11's the served tv_q4_hh, both compiled from this unit.
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

echo '== process 1: K4 row loops, K11 int4 rows, K3b counters, beside the one-launch pass =='
date -u
rc=0
LLVQ_TILE_BLOCKS=64 LLVQ_BENCH_ARMS=slot32,planes14,fp16,tetra48 \
LLVQ_SEG_TETRA=1 LLVQ_TETRA_OCC=1 LLVQ_TROWS=1 LLVQ_DUMP_CUBIN="$O/cubin" \
  planesbench "$B" "$F" > "$O/bench.txt" 2>&1 || rc=$?
date -u
echo "planesbench exit $rc"
grep -aE 'tetra48: [0-9]+ of [0-9]+|^tile |Row-loop|reference: the served|tv_trows|tv_q4rows|BIT FOR BIT|INVALIDATED|REFUSED|SPILLS|diagnostic|stamps d_in|relaid|K4 rule|K11 rule|K3b rule|Int4 rows|Counters|counts OK|ABORTED|void|us a hop|One-launch|tpersall|B_pa|speed over|not timed|cubin ' "$O/bench.txt" | tail -140
[ "$rc" = 0 ]
grep -q 'tetra48: 252 of 252 matrices matched by name' "$O/bench.txt"
grep -q '^tile 64' "$O/bench.txt"
# gate 3: the references are the served unit
grep -q "reference: the served unit, 101625 bytes, sha256 $SERVED_SHA" "$O/bench.txt" \
  || { echo "gate 3: the row loops' reference is not the served unit $SERVED_SHA" >&2; exit 1; }
# gate 4: the bases of every ratio owe the bits and keep them: v2 (K4), q4 (K11)
grep -q '  v2: [0-9]* rows identical BIT FOR BIT to tv_tetra48_h' "$O/bench.txt"
grep -q '  q4: [0-9]* rows identical BIT FOR BIT to tv_q4_hh' "$O/bench.txt"
# gate 5: the barrier anchor of K3b ran every hop
grep -q '  bar: counts OK' "$O/bench.txt"
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
  --name "step1-micro${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
