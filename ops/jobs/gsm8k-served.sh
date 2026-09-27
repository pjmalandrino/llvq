#!/usr/bin/env bash
# GSM8K on a sealed paper-2 object through the served kernel, the whole test split.
# Preregistered: proofs/preregistration-gsm8k-campaign-2026-09-26.md, stamped before this.
#
#   SIZE=4b|8b|14b DRY_RUN=1 bash ops/jobs/gsm8k-served.sh   prints and parses the job script
#   SIZE=4b|8b|14b IMAGE_SHA=<space sha> bash ops/jobs/gsm8k-served.sh
#   SIZE=4b ARM=f16-dense IMAGE_SHA=<sha> bash ops/jobs/gsm8k-served.sh
#   SIZE=4b LIMIT=50 IMAGE_SHA=<sha> bash ops/jobs/gsm8k-served.sh   the smoke, run 0
#
# LIMIT=50 draws the pilot's sample (gsm8k::select is nested and seeded), so the smoke's dump
# pairs with docs/data/gsm8k-dumps/pilot-4b-sealed-metal.jsonl through gsm8kpair, and its
# per-problem timings measure the served decode at GSM8K lengths before the census runs.
#
# ARM=served (default): oracle, the object's bytes and sha256 against the Mac, the served
# config written from this command line and checked by sha256, then `gsm8k` under
# LLVQ_CONFIG on the 1,319 problems. ARM=f16-dense, 4B only: the FP16 checkpoint through
# our dense path on the same card, the engine gate of the vLLM reference arms.
#
# Every LLVQ_* variable is set on its own command line, never exported: the job refuses
# to start with one in its environment. The dump is flushed per problem, so a job killed
# at its timeout keeps what it paid for, without its trailer.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

SIZE=${SIZE:?SIZE=4b, 8b or 14b}
ARM=${ARM:-served}
LIMIT=${LIMIT:-}
case "$LIMIT" in ''|[0-9]*) ;; *) echo "refused: LIMIT=$LIMIT is not a count" >&2; exit 1 ;; esac
case "$SIZE/$ARM" in
  4b/served)  OBJ_DIR=sealed-4b-2026-09-23;  NAME=qwen3-4b-sealed.bin
              LOCAL=$HOME/q4b-sealed-2026-09-23/$NAME; TIMEOUT=${TIMEOUT:-150m} ;;
  8b/served)  OBJ_DIR=sealed-8b27-2026-09-24; NAME=qwen3-8b-sealed-B.bin
              LOCAL=$HOME/q8b-14b-sealed-2026-09-23/$NAME; TIMEOUT=${TIMEOUT:-180m} ;;
  14b/served) OBJ_DIR=sealed-14b-2026-09-23; NAME=qwen3-14b-sealed.bin
              LOCAL=$HOME/q8b-14b-sealed-2026-09-23/$NAME; TIMEOUT=${TIMEOUT:-300m} ;;
  4b/f16-dense) TIMEOUT=${TIMEOUT:-300m} ;;
  *) echo "refused: SIZE=$SIZE ARM=$ARM (served at 4b, 8b, 14b; f16-dense at 4b only)" >&2; exit 1 ;;
esac
PREREG=proofs/preregistration-gsm8k-campaign-2026-09-26.md
BUCKET=Pier-Jean/jobs-artifacts
DATASET_REV=740312add88f781978c0658806c59bc2815b9866
F16_REPO=Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c
O=/out/gsm8k-$SIZE-$ARM${LIMIT:+-n$LIMIT}-2026-09-26${RETRY:+-r$RETRY}
[ -z "$LIMIT" ] || TIMEOUT=${TIMEOUT_SMOKE:-30m}
DRY=${DRY_RUN:-0}

if [ "$ARM" = served ]; then
  CONF_LOCAL=configs/qwen3-$SIZE-tetra-e4.json
  CONF=$(cat "$CONF_LOCAL")
  CONF_SHA=$(printf '%s\n' "$CONF" | shasum -a 256 | cut -d' ' -f1)
  [ "$CONF_SHA" = "$(shasum -a 256 < "$CONF_LOCAL" | cut -d' ' -f1)" ] \
    || { echo "refused: $CONF_LOCAL does not end in exactly one newline" >&2; exit 1; }
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1]));
assert (d["layout"],d["embed"],d["rot_share"],d["fuse"],d["kv"])==("tetra48","q4","1","0","f16"), d' "$CONF_LOCAL"
  F=/out/$OBJ_DIR/$NAME
  C=$O/qwen3-$SIZE-tetra-e4.json
else
  CONF= ; CONF_SHA= ; F= ; C=
fi

if [ "$DRY" = 1 ]; then
  SHA=0000000000000000000000000000000000000000000000000000000000000000
  BYTES=0
else
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  if [ "$ARM" = served ]; then
    test -f "$LOCAL" || { echo "refused: $LOCAL missing" >&2; exit 1; }
    SHA=$(shasum -a 256 "$LOCAL" | cut -d' ' -f1)
    BYTES=$(stat -f %z "$LOCAL")
    REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/$OBJ_DIR/" 2>/dev/null | awk -v n="/$NAME" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
    [ "$REMOTE" = "$BYTES" ] || { echo "refused: bucket copy is ${REMOTE:-absent} B, local $BYTES B" >&2; exit 1; }
  else
    SHA= ; BYTES=
  fi
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of the rebuild that carries gsm8k>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "$SIZE $ARM: object ${SHA:-none}; config ${CONF_SHA:-none}; image $NOW; timeout $TIMEOUT"
fi

PRE=$(printf 'O=%q\nF=%q\nC=%q\nSHA=%q\nBYTES=%q\nCONF=%q\nCONF_SHA=%q\nSIZE=%q\nARM=%q\nLIMIT=%q\nDATASET_REV=%q\nF16_REPO=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$F" "$C" "$SHA" "$BYTES" "$CONF" "$CONF_SHA" "$SIZE" "$ARM" "$LIMIT" "$DATASET_REV" "$F16_REPO")

IFS= read -r -d '' HEAD <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt"
echo '== oracle (hard rule 10) =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle-cuda.txt" | tail -2
grep -q MATCH "$O/oracle-cuda.txt"
JOB

IFS= read -r -d '' SERVED <<'JOB' || true
echo '== the object on the mount: bytes and sha256 against the Mac =='
test "$(stat -c %s "$F")" = "$BYTES"
sha256sum "$F" | tee "$O/files.sha256"
test "$(cut -d' ' -f1 "$O/files.sha256")" = "$SHA"
echo '== the served config, written from the launch command =='
printf '%s\n' "$CONF" > "$C"
sha256sum "$C" | tee -a "$O/files.sha256"
test "$(sha256sum "$C" | cut -d' ' -f1)" = "$CONF_SHA"
echo "== GSM8K through the served kernel, ${LIMIT:-all 1,319} problems =="
date -u
LLVQ_DATASET_REV="$DATASET_REV" LLVQ_CONFIG="$C" LLVQ_GSM8K_DUMP="$O/gsm8k-$SIZE-served.jsonl" \
  gsm8k "$F" cuda $LIMIT 2>&1 | tee "$O/gsm8k-$SIZE-served.txt" | grep -vE '^  \[ *[0-9]+/' | tail -16
date -u
tail -1 "$O/gsm8k-$SIZE-served.jsonl" | grep -q '"end":true'
echo '== raw kept ==' ; ls -la "$O"
JOB

IFS= read -r -d '' DENSE <<'JOB' || true
echo '== GSM8K, FP16 checkpoint through our dense path, 1,319 problems =='
date -u
LLVQ_DATASET_REV="$DATASET_REV" LLVQ_GSM8K_DUMP="$O/gsm8k-4b-f16-dense.jsonl" \
  gsm8k "$F16_REPO" cuda 2>&1 | tee "$O/gsm8k-4b-f16-dense.txt" | grep -vE '^  \[ *[0-9]+/' | tail -16
date -u
tail -1 "$O/gsm8k-4b-f16-dense.jsonl" | grep -q '"end":true'
echo '== raw kept ==' ; ls -la "$O"
JOB

case "$ARM" in
  served) BODY="$HEAD
$SERVED" ;;
  f16-dense) BODY="$HEAD
$DENSE" ;;
esac

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
  --name "gsm8k-$SIZE-$ARM${LIMIT:+-n$LIMIT}${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
