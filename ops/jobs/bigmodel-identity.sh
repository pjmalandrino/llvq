#!/usr/bin/env bash
# The 32B wall lift on the L40S: rot_apply_dev and tv_q4_hs against the host's bytes, at every width through 31,744.
# Preregistered: proofs/preregistration-bigmodel-2-2026-11-04.md, which must be stamped before this launches.
#
#   DRY_RUN=1 bash ops/jobs/bigmodel-identity.sh                prints and parses the job script
#   IMAGE_SHA=<space sha> bash ops/jobs/bigmodel-identity.sh    launches, after the operator's go
#   IMAGE_SHA=<sha> EMIT=<file> bash ops/jobs/bigmodel-identity.sh  writes the job script for ops/jobs/lot1.sh
#
# Three steps in one job: oracle; rotbench (K0 and R0, 14 widths, hashes against the compiled-in fixture, rows
# twins, 5 interleaved rounds); fusedrun with LLVQ_INT4_WIDE_CHECK=1 on the sealed 4B (tv_q4_hs against tv_q4_hh on
# its first int4 record, and against the host's hashes on three synthetic records). No token is decoded.
#
# Identity failures are outcomes, not gates (prereg §4-5): each binary prints every comparison, and the job keeps
# going after a mismatch so the raw output of both steps is kept. The gates are checked by grep below.
#
# Cost, estimated (l40sx1 $1.80/h): 10-13 min, $0.30-0.39. Ceiling 20m, $0.60.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

OBJ_DIR=sealed-4b-2026-09-23
NAME=qwen3-4b-sealed.bin
LOCAL=$HOME/q4b-sealed-2026-09-23/$NAME
TIMEOUT=20m
PREREG=proofs/preregistration-bigmodel-2-2026-11-04.md
BUCKET=Pier-Jean/jobs-artifacts
# RETRY=2 names a relaunch after a failed attempt left its directory behind.
O=/out/bigmodel-identity-2026-11-04${RETRY:+-r$RETRY}
F=/out/$OBJ_DIR/$NAME
DRY=${DRY_RUN:-0}

if [ "$DRY" = 1 ]; then
  SHA=0000000000000000000000000000000000000000000000000000000000000000
  BYTES=0
else
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  test -f "$LOCAL" || { echo "refused: $LOCAL missing" >&2; exit 1; }
  SHA=$(shasum -a 256 "$LOCAL" | cut -d' ' -f1)
  BYTES=$(stat -f %z "$LOCAL")
  REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/$OBJ_DIR/" 2>/dev/null | awk -v n="/$NAME" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
  [ "$REMOTE" = "$BYTES" ] || { echo "refused: bucket copy is ${REMOTE:-absent} B, local $BYTES B" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of the rebuild from the commit carrying the prereg>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "4b sealed: $BYTES B, sha256 $SHA; image $NOW"
fi

PRE=$(printf 'O=%q\nF=%q\nSHA=%q\nBYTES=%q' "$O" "$F" "$SHA" "$BYTES")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt"
echo '== gate 1: oracle (hard rule 10) =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle-cuda.txt" | tail -2
grep -q MATCH "$O/oracle-cuda.txt"
echo '== the object on the mount: bytes and sha256 against the Mac =='
test "$(stat -c %s "$F")" = "$BYTES"
sha256sum "$F" | tee "$O/files.sha256"
test "$(cut -d' ' -f1 "$O/files.sha256")" = "$SHA"

echo '== STEP 2: rotbench, K0 and R0 against the host fixture =='
date -u
RB=0
rotbench 2>&1 | tee "$O/rotbench.txt" || RB=$?
date -u
echo "rotbench exit $RB" | tee -a "$O/rotbench.txt"
# gate 2: the card and both bounds
grep -qF '49152 B of shared memory per block by default, 101376 after opt-in' "$O/rotbench.txt"
# gate 3: four rotation entry points, none spilling (rotbench stops on a spill)
test "$(grep -cE '^  rot_apply(_rows)?(_dev)? +[0-9]+ registers, 0 local B' "$O/rotbench.txt")" = 4
# gate 4: no K0 arm at 25,600 or 31,744 (an explicit `if`: `! grep` never trips `set -e`)
if grep -E '^  .* (25600|31744) .*rot_apply/' "$O/rotbench.txt"; then echo 'gate 4 FAILED' >&2; exit 1; fi
# gate 5: the card drew the host's inputs (rotbench stops on the first that differs)
if grep -F 'INPUT sha256' "$O/rotbench.txt"; then echo 'gate 5 FAILED: input drift' >&2; exit 1; fi
echo 'gates 2-5 hold'

echo '== STEP 4: LLVQ_INT4_WIDE_CHECK on the sealed 4B =='
FLAGS='LLVQ_FUSED_LAYOUT=tetra48 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_EMBED=q4'
date -u
FR=0
env $FLAGS LLVQ_INT4_WIDE_CHECK=1 fusedrun "$F" 2>&1 | tee "$O/int4-wide.txt" | tail -20 || FR=$?
date -u
echo "fusedrun exit $FR" | tee -a "$O/int4-wide.txt"
# gate 7: the served load's own parity check, as always
grep -qF 'int4: tv_q4_hh equals tv_q4_h bit for bit' "$O/int4-wide.txt"
# gate 3, int4 half: tv_q4_hs does not spill
grep -qE '^  tv_q4_hs: [0-9]+ registers, 0 local B' "$O/int4-wide.txt"
echo 'gates 3 and 7 hold'
echo "== outcomes: rotbench exit $RB, int4 check exit $FR (0 = every comparison equal) =="
echo '== raw kept ==' ; ls -la "$O" ; wc -l "$O"/*.txt
JOB

# EMIT=<file>: write the job script to <file> and launch nothing. ops/jobs/lot1.sh runs it as one part
# of a lot job on one card. Every check above has run, a launch's included unless DRY_RUN=1.
if [ -n "${EMIT:-}" ]; then
  printf 'set -euo pipefail\n%s\n%s\n' "$PRE" "$BODY" > "$EMIT"
  bash -n "$EMIT" && echo "EMIT: job script written to $EMIT; nothing launched" >&2
  exit 0
fi

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
  --name "bigmodel-identity${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
