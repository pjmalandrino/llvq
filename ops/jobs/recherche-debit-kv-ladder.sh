#!/usr/bin/env bash
# Decode-speed research phase at long context (CERVEAU O13), step R3: the quality ladder of a quantized KV cache.
# One job scores ONE arm of LLVQ_KV_LADDER (llvq-llm/src/kvladder.rs) against the f16 cache on the sealed 4B:
# LLVQ_GQA_PPL=8192,12,0,ladder, WikiText-2, 12 windows of 8,192, decode path, both caches in lockstep under the
# served 4E attention form (grow(256), the kernel on every decode row). The arm is a round trip: the cache keeps its
# dtype, no byte is saved and no speed is read. Four jobs, one an arm: k8v8, k8v4, k4v4c, k4v4h.
# A short ladder run (2 windows of 256) comes first and stops the job: the arm's ops have never run on CUDA.
# Preregistered: proofs/preregistration-recherche-debit-kv-ladder-2026-11-14.md, to stamp before this.
#
#   DRY_RUN=1 ARM=k8v4 bash ops/jobs/recherche-debit-kv-ladder.sh                 parses, launches nothing
#   IMAGE_SHA=<space sha> ARM=k8v4 bash ops/jobs/recherche-debit-kv-ladder.sh     launches, after the operator's go
#
# Cost (l40sx1 $1.80/h), *estimated*: LLVQ_GQA_PPL 8192,6,6,kernel fed 6 windows of 8,192 through two arms in
# lockstep in 940 s (attn-t8-r2-2026-11-04-brut/ppl-8192.txt), 157 s a window. The round-trip ops are not in that
# figure and were never timed: 160 to 220 s a window assumed, 32 to 44 min for 12, $1.0 to $1.3 an arm, $4.0 to $5.3
# for the four. k4v4c adds one 4,096-token calibration prefill and the 2.91 GB dclm-edu shard, 1 to 2 min assumed.
# Job timeout 60m: $1.80 an arm at worst, $7.20 for the four.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-recherche-debit-kv-ladder-2026-11-14.md
ARM=${ARM:?ARM=k8v8|k8v4|k4v4c|k4v4h}
case "$ARM" in k8v8|k8v4|k4v4c|k4v4h) ;; *) echo "refused: ARM=$ARM" >&2; exit 1;; esac
OUT_NAME=recherche-debit-kv-ladder-2026-11-14/$ARM
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/$OUT_NAME${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
TIMEOUT=${TIMEOUT:-60m}
CTX=8192
WINDOWS=12
# LLVQ_GQA_PPL's WikiText-2 revision (ops/jobs/attn-t8-r2.sh).
WIKI_REV=740312add88f781978c0658806c59bc2815b9866

SEALED="sealed-4b-2026-09-23/qwen3-4b-sealed.bin:1418224685"
SEALED_SHA=886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  grep -q '^TO WRITE$' "$PREREG" && { echo "refused: $PREREG still holds TO WRITE" >&2; exit 1; }
  path=${SEALED%%:*}; bytes=${SEALED##*:}
  REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/${path%/*}/" 2>/dev/null \
    | awk -v n="/${path##*/}" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
  [ "$REMOTE" = "$bytes" ] || { echo "refused: $path is ${REMOTE:-absent} B in the bucket, expected $bytes" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of the rebuild that carries this branch>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "bucket copy verified by size; image $NOW"
fi

PRE=$(printf 'O=%q\nSEALED=%q\nSEALED_SHA=%q\nCTX=%q\nWINDOWS=%q\nARM=%q\nWIKI_REV=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$SEALED" "$SEALED_SHA" "$CTX" "$WINDOWS" "$ARM" "$WIKI_REV")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
echo '== control 1: oracle =='
timeout 4m oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle.txt" | tail -2
grep -q MATCH "$O/oracle.txt"
echo '== control 2: the object =='
S=/out/${SEALED%%:*}; SB=${SEALED##*:}
test "$(stat -c %s "$S")" = "$SB"
sha256sum "$S" | tee "$O/files.sha256"
test "$(sha256sum "$S" | cut -d' ' -f1)" = "$SEALED_SHA"
FLAGS=(LLVQ_FUSED_LAYOUT=tetra48 LLVQ_EMBED=q4 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_KV=f16 LLVQ_DATASET_REV="$WIKI_REV")
echo "== control 3: the arm runs on this card ($ARM, 2 windows of 256) =="
date -u
env "${FLAGS[@]}" LLVQ_KV_LADDER="$ARM" LLVQ_GQA_PPL=256,2,0,ladder timeout 8m fusedrun "$S" 2>&1 | tee "$O/smoke.txt" \
  | grep -aE '^k4v4c|^KV ladder|^  bytes|^  window|^  decode|^Error|^error'
grep -aq '^  decode top-1 agreement' "$O/smoke.txt"
echo "== the measurement: $ARM against f16, $WINDOWS windows of $CTX =="
date -u
set +e
env "${FLAGS[@]}" LLVQ_KV_LADDER="$ARM" LLVQ_GQA_PPL="$CTX,$WINDOWS,0,ladder" timeout 52m fusedrun "$S" 2>&1 | tee "$O/ladder.txt" \
  | grep -aE '^sealed file|^loaded in|^attn_dec|^k4v4c|^KV ladder|^  bytes|^  window|^  path|^  decode|^Error|^error'
rc=${PIPESTATUS[0]}
set -e
echo "fusedrun ladder $ARM exit $rc" | tee -a "$O/ladder.txt"
date -u
ls -la "$O"
test "$rc" = 0
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
  --name "recherche-debit-kv-ladder-$ARM${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
