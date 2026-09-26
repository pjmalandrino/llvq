#!/usr/bin/env bash
# DRAFT, not launched. The MMLU census of the MMLU-format arm, full split,
# dense reconstruction. It pairs question by question against the census of the
# 61.11 object (dclm-ft-mmlu, job 6aaf099f, dump docs/data/mmlu-dumps/) — same
# base, same steps, same seed, same plan, one corpus apart.
#
# Preregistered under the training prereg, as the 4B ladder already does
# (dclm-rowscales.sh:2-4, docs/data/jobs.csv:171-172 file training and scoring
# under one prereg and one journal):
#   proofs/preregistration-mmluaux-4b-rowscales-<D>.md, stamped before training.
#
# Shape: ops/jobs/dclm-8b-ft-mmlu.sh — oracle first (hard rule 10), one arm,
# LLVQ_MMLU_DUMP, the dump header checked line by line, and the object's sha256
# read on the mount against the Mac's.
#
# Cost: the 4B FT census billed 24 min for $0.72 (job 6aaf099f). Timeout 60m,
# ceiling $1.80.
#
# DRY_RUN=1 bash ops/jobs/mmluaux-4b-ft-mmlu.sh   prints, parses, launches nothing.
set -euo pipefail
REPO=$HOME/Documents/Pro/workspace/poc/llvq
cd "$REPO"

D=${D:-2026-09-23}
PREREG=${PREREG:-proofs/preregistration-mmluaux-4b-rowscales-$D.md}
BK=Pier-Jean/jobs-artifacts
OBJ=mmluaux-4b-ft-$D
LOCAL=$HOME/qwen3-4b-mmluaux-ft.bin
SUMS=$HOME/mmluaux-$D.sha256
O=/out/census-4b-mmluaux-$D
F=/out/$OBJ/qwen3-4b-mmluaux-ft.bin
FP=a74a6d6213602979              # the census fingerprint both 4B arms printed
DRY=${DRY_RUN:-0}

if [ "$DRY" = 1 ]; then
  SHA=0000000000000000000000000000000000000000000000000000000000000000
  BYTES=1794564765
else
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  SHA=$(awk '$2 ~ /qwen3-4b-mmluaux-ft\.bin$/ {print $1}' "$SUMS" | tail -1)
  [ ${#SHA} -eq 64 ] || { echo "refused: no sha256 for qwen3-4b-mmluaux-ft.bin in $SUMS" >&2; exit 1; }
  [ "$SHA" = "$(shasum -a 256 "$LOCAL" | cut -d' ' -f1)" ] || { echo "refused: $LOCAL differs from $SUMS" >&2; exit 1; }
  BYTES=$(stat -f %z "$LOCAL")
  [ "$BYTES" = "$(stat -f %z "$HOME/qwen3-4b-dclm.bin")" ] || { echo "refused: the arm and its base differ in size" >&2; exit 1; }
  REMOTE=$(hf buckets ls "hf://buckets/$BK/$OBJ/" 2>/dev/null | awk '$NF ~ /qwen3-4b-mmluaux-ft\.bin$/ {print $1}' || true)
  [ "$REMOTE" = "$BYTES" ] || { echo "refused: bucket copy is ${REMOTE:-absent} B, local $BYTES B" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BK/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already holds files" >&2; exit 1
  fi
  echo "arm object: $BYTES B, sha256 $SHA"
fi

PRE=$(printf 'O=%q\nF=%q\nSHA=%q\nBYTES=%q\nFP=%q' "$O" "$F" "$SHA" "$BYTES" "$FP")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
check() {
  grep -qxF '# dtype=f16' "$1"
  grep -qxF '# limit=census' "$1"
  # The plan, verbatim: `mmlupair` refuses two dumps of different plans, and
  # both reference dumps on disk carry this exact line.
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
echo '== the object on the mount: bytes and sha256 against the Mac =='
test "$(stat -c %s "$F")" = "$BYTES"
sha256sum "$F" | tee "$O/files.sha256"
test "$(cut -d' ' -f1 "$O/files.sha256")" = "$SHA"
echo '== arm: DCLM base + row scales trained on MMLU-format text, FULL split ==' ; date -u
LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP="$O/mmlu-4b-mmluaux-ft-FULL.csv" mmlu "$F" cuda 2>&1 | tee "$O/out-mmluaux-ft.txt" | tail -10
date -u
check "$O/mmlu-4b-mmluaux-ft-FULL.csv"
echo '== dump ==' ; wc -l "$O"/mmlu-4b-*-FULL.csv ; ls -la "$O"
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
  --bucket "$BK" --out-mount /out \
  --name mmluaux-4b-ft-mmlu \
  "$PRE" "$BODY"
