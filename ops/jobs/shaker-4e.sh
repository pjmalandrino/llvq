#!/usr/bin/env bash
# The "shaker": one card job with every 4E reading prepared at $0 on 2026-11-05.
#   A. 4E v3, v4 and v5 against the eager path on the sealed 4B, 5 and 8,192 prompt tokens (fusedrun
#      LLVQ_PERS_4E=64): the row forms 1, 2, 3 (llvq_pers.cuh), each owing the eager tokens, their rates round by round,
#      and one stamped launch each with the new inner stamps of q/k/v and MLP.
#   B. The row-loop microbenchmark with the k2 arm (planesbench LLVQ_TROWS=1 on the banc-252 files).
#   C. The served door under configs/qwen3-4b-tetra-e4-4e.json: the stamped protocol of
#      proofs/preregistration-served-4e-2026-11-05.md, its preamble and body taken from ops/jobs/served-4e.sh at
#      build time and run unchanged, its output under C's own directory (the departure is written beside that
#      prereg, -ECARTS.md).
# Preregistered: proofs/preregistration-shaker-4e-2026-11-05.md for A and B, to stamp before this.
#
#   DRY_RUN=1 bash ops/jobs/shaker-4e.sh                 parses the job script, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/shaker-4e.sh      launches, after the operator's go
#
# Each part runs whatever the others did; the job fails at the end if any part failed. Every LLVQ_* variable is set
# on its own command line. Cost, estimated (l40sx1 $1.80/h, prereg section 8): about 33 min, $0.99. Timeout 60m:
# $1.80 at worst.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-shaker-4e-2026-11-05.md
PREREG_C=proofs/preregistration-served-4e-2026-11-05.md
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/shaker-4e-2026-11-05${RETRY:+-r$RETRY}
TIMEOUT=60m
DRY=${DRY_RUN:-0}

BALL="ball-ref-2026-09-20/qwen3-4b-llvq.bin:1770527533"
TETRA="tetra-4b-2026-09-06/qwen3-4b-tetra.bin:1770529149"
SEALED="sealed-4b-2026-09-23/qwen3-4b-sealed.bin:1418224685"
TETRA_SHA8=0adb7cfd
SEALED_SHA8=886391a8
SERVED_SHA=6193094310bc18acdb0a0c554807ff45f6df09c1f6ab49e0577032874f107812

# Part C: the stamped served-4e job's own preamble and body, read from its file (lines up to the end of its BODY
# heredoc), in a subshell with DRY_RUN=1 so its launch-time checks are skipped here; they are this script's below.
C_SRC=$(sed -n '1,/^JOB$/p' ops/jobs/served-4e.sh)
C_JOB=$(DRY_RUN=1 bash -c "$C_SRC"$'\nprintf "%s\\n" "$PRE"; printf "%s\\n" "__BODY__"; printf "%s" "$BODY"' ops/jobs/served-4e.sh)
C_PRE=${C_JOB%%$'\n'__BODY__*}
C_BODY=${C_JOB#*__BODY__$'\n'}
[ -n "$C_PRE" ] && [ -n "$C_BODY" ] || { echo "refused: could not read ops/jobs/served-4e.sh's job" >&2; exit 1; }

if [ "$DRY" != 1 ]; then
  for p in "$PREREG" "$PREREG_C"; do
    { test -f "$p" && test -f "$p.ots"; } || { echo "refused: $p(.ots) missing" >&2; exit 1; }
  done
  for row in "$BALL" "$TETRA" "$SEALED"; do
    path=${row%%:*}; bytes=${row##*:}
    REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/${path%/*}/" 2>/dev/null \
      | awk -v n="/${path##*/}" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
    [ "$REMOTE" = "$bytes" ] || { echo "refused: $path is ${REMOTE:-absent} B in the bucket, expected $bytes" >&2; exit 1; }
  done
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of the rebuild that carries this branch>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "three bucket copies verified by size; image $NOW"
fi

PRE=$(printf 'O=%q\nB=%q\nF=%q\nS=%q\nBB=%q\nFB=%q\nSB=%q\nFSHA8=%q\nSSHA8=%q\nSERVED_SHA=%q\nC_PRE=%q\nC_BODY=%q' \
  "$O" "/out/${BALL%%:*}" "/out/${TETRA%%:*}" "/out/${SEALED%%:*}" "${BALL##*:}" "${TETRA##*:}" "${SEALED##*:}" \
  "$TETRA_SHA8" "$SEALED_SHA8" "$SERVED_SHA" "$C_PRE" "$C_BODY")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O/A" "$O/B/cubin"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
test "$(stat -c %s "$B")" = "$BB"
test "$(stat -c %s "$F")" = "$FB"
test "$(stat -c %s "$S")" = "$SB"
sha256sum "$F" "$S" | tee "$O/files.sha256"
test "$(sha256sum "$F" | cut -c1-8)" = "$FSHA8"
test "$(sha256sum "$S" | cut -c1-8)" = "$SSHA8"
echo '== gate 1: oracle =='
timeout 5m oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle.txt" | tail -2
grep -q MATCH "$O/oracle.txt"
failed=""

echo '== PART A: 4E v3, v4, v5 against the eager path, 5 and 8,192 prompt tokens =='
date -u
rc=0
LLVQ_FUSED_LAYOUT=tetra48 LLVQ_EMBED=q4 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_PERS_4E=64 \
  timeout 20m fusedrun "$S" 64 > "$O/A/pers-4e.txt" 2>&1 || rc=$?
date -u
grep -aE '^4E|cudarc event|context |tokens equal|served door|tok/s|4E / eager|phase stamps|stamped token|inside q/k/v|^ +(q/k/v|norm|attention|o |MLP|head|token|embedding)' "$O/A/pers-4e.txt" | tail -120
{
  [ "$rc" = 0 ] &&
  grep -q 'cudarc event tracking: off (one stream, ordered by issue)' "$O/A/pers-4e.txt" &&
  grep -q '^4E: RoPE forms equal to candle' "$O/A/pers-4e.txt" &&
  test "$(grep -c '4E v3 at 512 threads: tokens equal the eager path' "$O/A/pers-4e.txt")" = 2
} || { echo 'PART A: a gate failed (v3 identity at both contexts, event tracking, RoPE)'; failed="$failed A"; }
for f in v4 v5; do
  echo "4E $f identity lines: $(grep -c "4E $f at 512 threads: tokens equal the eager path" "$O/A/pers-4e.txt") of 2"
done

echo '== PART B: the row loops alone, the k2 arm among them =='
date -u
rc=0
LLVQ_TILE_BLOCKS=64 LLVQ_BENCH_ARMS=slot32,planes14,fp16,tetra48 \
LLVQ_SEG_TETRA=1 LLVQ_TROWS=1 LLVQ_DUMP_CUBIN="$O/B/cubin" \
  timeout 15m planesbench "$B" "$F" > "$O/B/bench.txt" 2>&1 || rc=$?
date -u
grep -aE 'tetra48: [0-9]+ of [0-9]+|^tile |Row-loop|reference: the served|tv_trows|BIT FOR BIT|INVALIDATED|SPILLS|stamps d_in|speed over|second token|not timed' "$O/B/bench.txt" | tail -50
{
  [ "$rc" = 0 ] &&
  grep -q 'tetra48: 252 of 252 matrices matched by name' "$O/B/bench.txt" &&
  grep -q '^tile 64' "$O/B/bench.txt" &&
  grep -q "reference: the served unit, 101625 bytes, sha256 $SERVED_SHA" "$O/B/bench.txt" &&
  grep -q '  v2: [0-9]* rows identical BIT FOR BIT to tv_tetra48_h' "$O/B/bench.txt"
} || { echo 'PART B: a gate failed'; failed="$failed B"; }

echo '== PART C: the served-4e protocol, unchanged, under its own directory =='
date -u
rc=0
( eval "$C_PRE"; O="$O/C"; eval "$C_BODY" ) > "$O/C.txt" 2>&1 || rc=$?
date -u
tail -40 "$O/C.txt"
[ "$rc" = 0 ] || { echo 'PART C: the served-4e protocol stopped on one of its gates'; failed="$failed C"; }

ls -laR "$O" | head -80
[ -z "$failed" ] || { echo "parts failed:$failed"; exit 1; }
JOB

if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; part C read from ops/jobs/served-4e.sh ($(printf '%s' "$C_BODY" | wc -l | tr -d ' ') lines); nothing launched" >&2
  printf '%s\n' "$C_PRE" | bash -n && printf '%s\n' "$C_BODY" | bash -n
  exit 0
fi

uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "shaker-4e${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
