#!/usr/bin/env bash
# Step L1 of the persistent-kernel lead: the one-launch Tetra pass (tpersall), the barrier and launch
# microbenchmarks, the driver cubins, and the in-situ value of a kernel boundary in the served 4B.
# Preregistered: proofs/preregistration-pers-l1-2026-11-04.md, to stamp before this.
#
#   DRY_RUN=1 bash ops/jobs/pers-l1.sh                 prints and parses the job script
#   IMAGE_SHA=<space sha> bash ops/jobs/pers-l1.sh      launches, after the operator's go
#
# One job, l40sx1, three processes: planesbench on the two banc-252 files (five table arms, A4 with
# the Tetra fusion, tpersall, the microbenchmarks, the cubins); fusedrun through the served door for
# its NVRTC digest; fusedrun LLVQ_BOUNDARY_AB=1 on the sealed 4B. No frozen setting is touched:
# tpersall runs 256-thread CTAs at today's occupancy, tables in L1, stride 24.
#
# Cost, estimated (l40sx1 $1.80/h, the median of the priced jobs of 5 min or more): about 12 min,
# $0.36 (seg-tetra-2026-11-04 ran the same bench in 270 s with four arms). Timeout 90m, because METHODE
# §6 asks it of any planesbench with five arms or more: $2.70 at worst.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-pers-l1-2026-11-04.md
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/pers-l1-2026-11-04${RETRY:+-r$RETRY}
TIMEOUT=90m
DRY=${DRY_RUN:-0}

BALL="ball-ref-2026-09-20/qwen3-4b-llvq.bin:1770527533"
TETRA="tetra-4b-2026-09-06/qwen3-4b-tetra.bin:1770529149"
SEALED="sealed-4b-2026-09-23/qwen3-4b-sealed.bin:1418224685"
TETRA_SHA8=0adb7cfd
SEALED_SHA8=886391a8
CONF_GRP_LOCAL=configs/qwen3-4b-tetra-e4-grouped.json
CONF_GRP=$(cat "$CONF_GRP_LOCAL")
CONF_GRP_SHA=$(printf '%s\n' "$CONF_GRP" | shasum -a 256 | cut -d' ' -f1)
[ "$CONF_GRP_SHA" = "$(shasum -a 256 < "$CONF_GRP_LOCAL" | cut -d' ' -f1)" ] \
  || { echo "refused: $CONF_GRP_LOCAL does not end in exactly one newline" >&2; exit 1; }
# The served unit's digest, computed from this tree by the test that pins it
# (llvq-cuda/src/pers.rs::the_served_unit_is_the_published_one).
SERVED_SHA=6193094310bc18acdb0a0c554807ff45f6df09c1f6ab49e0577032874f107812

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  for row in "$BALL" "$TETRA" "$SEALED"; do
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
  echo "three bucket copies verified by size; config $CONF_GRP_SHA; image $NOW"
fi

PRE=$(printf 'O=%q\nB=%q\nF=%q\nS=%q\nBB=%q\nFB=%q\nSB=%q\nFSHA8=%q\nSSHA8=%q\nCG=%q\nCONF_GRP=%q\nCONF_GRP_SHA=%q\nSERVED_SHA=%q' \
  "$O" "/out/${BALL%%:*}" "/out/${TETRA%%:*}" "/out/${SEALED%%:*}" "${BALL##*:}" "${TETRA##*:}" "${SEALED##*:}" \
  "$TETRA_SHA8" "$SEALED_SHA8" "$O/qwen3-4b-tetra-e4-grouped.json" "$CONF_GRP" "$CONF_GRP_SHA" "$SERVED_SHA")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O" "$O/cubin"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
test "$(stat -c %s "$B")" = "$BB"
test "$(stat -c %s "$F")" = "$FB"
test "$(stat -c %s "$S")" = "$SB"
sha256sum "$B" "$F" "$S" | tee "$O/files.sha256"
test "$(sha256sum "$F" | cut -c1-8)" = "$FSHA8"
test "$(sha256sum "$S" | cut -c1-8)" = "$SSHA8"
printf '%s\n' "$CONF_GRP" > "$CG"
test "$(sha256sum "$CG" | cut -d' ' -f1)" = "$CONF_GRP_SHA"

echo '== gate 1: oracle =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle.txt" | tail -3
grep -q 'MATCH' "$O/oracle.txt"

echo '== process 1: the bench =='
date -u
rc=0
LLVQ_TILE_BLOCKS=64 LLVQ_BENCH_ARMS=slot32,planes14,fp16,nullk,tetra48 \
LLVQ_SEG_TETRA=1 LLVQ_TETRA_OCC=1 LLVQ_PERS_MICRO=1 LLVQ_DUMP_CUBIN="$O/cubin" \
  planesbench "$B" "$F" > "$O/bench.txt" 2>&1 || rc=$?
date -u
echo "planesbench exit $rc"
grep -aE 'tetra48: [0-9]+ of [0-9]+|^tile |bit-exact|tpersall|barrier|ring of|empty |cubin |B_pa|gain over' "$O/bench.txt" | tail -40
grep -q 'tetra48: 252 of 252 matrices matched by name' "$O/bench.txt"
grep -q '^tile 64' "$O/bench.txt"

echo '== process 2a: the served door, for its digest =='
LLVQ_CONFIG="$CG" fusedrun "$S" 16 > "$O/served-16.txt" 2>&1
tail -5 "$O/served-16.txt"

echo '== process 2b: the boundary A/B =='
date -u
LLVQ_FUSED_LAYOUT=tetra48 LLVQ_EMBED=q4 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_BOUNDARY_AB=1 \
  fusedrun "$S" 256 > "$O/boundary.txt" 2>&1
date -u
tail -30 "$O/boundary.txt"
# gate 4: each line from where it is printed, and the two doors compiled one unit
grep -q 'int4: tv_q4_hh equals tv_q4_h bit for bit' "$O/boundary.txt"
grep -q 'cudarc event tracking: off (one stream, ordered by issue)' "$O/boundary.txt"
grep -q '^attention grouped on grow(256)' "$O/boundary.txt"
D_SERVED=$(grep -o 'NVRTC source: [0-9]* bytes, sha256 [0-9a-f]*' "$O/served-16.txt" | head -1)
D_BENCH=$(grep -o 'NVRTC source: [0-9]* bytes, sha256 [0-9a-f]*' "$O/boundary.txt" | head -1)
echo "served door: $D_SERVED" | tee "$O/digests.txt"
echo "bench door : $D_BENCH" | tee -a "$O/digests.txt"
[ -n "$D_SERVED" ] && [ "$D_SERVED" = "$D_BENCH" ]
case "$D_SERVED" in *"$SERVED_SHA"*) ;; *) echo "gate 4: the served unit moved from $SERVED_SHA" >&2; exit 1;; esac
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
  --name "pers-l1${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
