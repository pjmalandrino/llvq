#!/usr/bin/env bash
# DRAFT (integrator), not launched. The MMLU census of the row-scale-trained 8B,
# one arm, full split, dense reconstruction: the arm that pairs against arm A
# (DCLM base) of census3.sh. Target home: ops/jobs/dclm-8b-ft-mmlu.sh.
#
# Preregistered under the training prereg, as the 4B did (dclm-rowscales.sh:2-4
# and docs/data/jobs.csv:171-172 file training and scoring under one prereg and
# one journal): proofs/preregistration-dclm-8b-rowscales-<OBJ_DATE>.md, stamped
# before the training job. Its signed prediction is the PAIRED delta against the
# base census, since the base may not be read when it is stamped.
#
# Same image as census3.sh, or the pair crosses images: the Space holds
# afaed1e (Space sha a963a020, 2026-09-21 01:12 UTC). The launcher refuses when
# the Space sha differs from IMAGE_SHA, and nobody runs `ops/run.py publish
# --cuda` before this job has finished.
#
# Shape: ops/jobs/tetranu-ft-mmlu.sh (oracle, one arm, LLVQ_MMLU_DUMP), plus
# census3.sh's dump-header check and the sha256 of the object on the mount.
# Cost, estimated: 4B FT census ran 1,375 s (hf jobs inspect 6aaf099f); vod-8b's
# arm A segment 32.6 min with oracle and load (census3.sh:32-37). ~34 min,
# ~$1.02 on l40sx1; timeout 60m, ceiling $1.80.
#
# DRY_RUN=1 bash census-ft8b.sh   prints the job script, parses it, launches nothing.
set -euo pipefail
REPO=$HOME/Documents/Pro/workspace/poc/llvq
cd "$REPO"

OBJ_DATE=${OBJ_DATE:-2026-09-21}
D=${CENSUS_DATE:-2026-09-22}                         # launch date, names the output dir
PREREG=${PREREG:-proofs/preregistration-dclm-8b-rowscales-$OBJ_DATE.md}
IMAGE_SHA=${IMAGE_SHA:-a963a02010cec2d3c342ec52dd38f2dded06f2a1}
BUCKET=Pier-Jean/jobs-artifacts
OBJ_DIR=dclm-8b-ft-$OBJ_DATE                          # export-fold-8b.sh upload-ft
LOCAL=$HOME/qwen3-8b-dclm-ft.bin
SUMS=$HOME/q8b-dclm-2026-09-21/files.sha256          # export-fold-8b.sh fold appends the FT sha
O=/out/census-8b-ft-$D
F=/out/$OBJ_DIR/qwen3-8b-dclm-ft.bin
FP=a74a6d6213602979
DRY=${DRY_RUN:-0}

if [ "$DRY" = 1 ]; then
  SHA=0000000000000000000000000000000000000000000000000000000000000000
  BYTES=4364205777
else
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  SHA=$(awk '$2 ~ /qwen3-8b-dclm-ft\.bin$/ {print $1}' "$SUMS" | tail -1)
  [ ${#SHA} -eq 64 ] || { echo "refused: no sha256 for qwen3-8b-dclm-ft.bin in $SUMS" >&2; exit 1; }
  [ "$SHA" = "$(shasum -a 256 "$LOCAL" | cut -d' ' -f1)" ] || { echo "refused: $LOCAL differs from $SUMS" >&2; exit 1; }
  BYTES=$(stat -f %z "$LOCAL")
  [ "$BYTES" = "$(stat -f %z "$HOME/qwen3-8b-dclm.bin")" ] || { echo "refused: FT and base sizes differ" >&2; exit 1; }
  REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/$OBJ_DIR/" 2>/dev/null | awk '$NF ~ /qwen3-8b-dclm-ft\.bin$/ {print $1}' || true)
  [ "$REMOTE" = "$BYTES" ] || { echo "refused: bucket copy is ${REMOTE:-absent} B, local $BYTES B" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/census-8b-ft-$D/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: census-8b-ft-$D/ already exists" >&2; exit 1
  fi
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space moved to $NOW; the base census ran on $IMAGE_SHA" >&2; exit 1; }
  echo "FT object: $BYTES B, sha256 $SHA; image $NOW"
fi

PRE=$(printf 'O=%q\nF=%q\nSHA=%q\nBYTES=%q\nFP=%q' "$O" "$F" "$SHA" "$BYTES" "$FP")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
check() {
  grep -qxF '# dtype=f16' "$1"
  grep -qxF '# limit=census' "$1"
  grep -qxF '# alloc=flat, 100..1534 per subject, 14042 questions' "$1"
  grep -qxF '# config=none' "$1"
  grep -qxF '# arithmetic=dense reconstruction' "$1"
  grep -qxF '# kv=f16' "$1"
  test "$(tail -n 1 "$1")" = "# end fingerprint=$FP questions=14042"
  echo "dump ok: $1"
}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt" || true
echo '== oracle (hard rule 10) =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle-cuda.txt" | tail -2
grep -q MATCH "$O/oracle-cuda.txt"
echo '== the FT file on the mount: bytes and sha256 against the Mac =='
test "$(stat -c %s "$F")" = "$BYTES"
sha256sum "$F" | tee "$O/files.sha256"
test "$(cut -d' ' -f1 "$O/files.sha256")" = "$SHA"
echo '== arm FT: DCLM base + trained row scales, sealed, dense reconstruction, FULL split ==' ; date -u
LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP="$O/mmlu-8b-dclm-ft-FULL.csv" mmlu "$F" cuda 2>&1 | tee "$O/out-dclm-ft.txt" | tail -10
date -u
check "$O/mmlu-8b-dclm-ft-FULL.csv"
echo '== dump ==' ; wc -l "$O"/mmlu-8b-*-FULL.csv ; ls -la "$O"
JOB

if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; nothing launched" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout 60m \
  --bucket "$BUCKET" --out-mount /out \
  --name dclm-8b-ft-mmlu \
  "$PRE" "$BODY"
