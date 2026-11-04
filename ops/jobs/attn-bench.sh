#!/usr/bin/env bash
# The split-KV decode-attention kernel alone, on one L40S: identity against the CPU replay, read
# rate, verify cost, registers. No artifact, no model. Preregistered:
# proofs/preregistration-attn-bench-2026-11-04.md, to be stamped before this runs.
#
#   DRY_RUN=1 bash ops/jobs/attn-bench.sh             prints and parses the job script
#   IMAGE_SHA=<space sha> bash ops/jobs/attn-bench.sh  launches, after the operator's go
#
# Two steps in one job: oracle (hard rule 10); `attnbench 2048,4096,8192`, which loads the four
# entry points, prints their driver reports, checks every one against the CPU replay bit for bit,
# then times them.
#
# Cost, estimated (l40sx1 $1.80/h): 6 to 8 min, $0.18 to $0.24. Ceiling 25m, $0.75.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

TIMEOUT=25m
PREREG=proofs/preregistration-attn-bench-2026-11-04.md
BUCKET=Pier-Jean/jobs-artifacts
LENGTHS=2048,4096,8192
# RETRY=2 names a relaunch after a failed attempt left its directory behind.
O=/out/attn-bench-2026-11-04${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
SRC_SHA=$(shasum -a 256 < llvq-cuda/kernels/attn_dec.cu | cut -d' ' -f1)

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of the rebuild that carries attnbench>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "attn_dec.cu sha256 $SRC_SHA (file); image $NOW"
fi

PRE=$(printf 'O=%q\nLENGTHS=%q' "$O" "$LENGTHS")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env (gate 4)' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
echo '== oracle (hard rule 10) =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle-cuda.txt" | tail -2
grep -q MATCH "$O/oracle-cuda.txt"
echo "== attnbench $LENGTHS =="
date -u
attnbench "$LENGTHS" 2>&1 | tee "$O/attnbench.txt"
date -u
grep -qF 'attn_dec.cu: embedded, sha256 ' "$O/attnbench.txt"
test "$(grep -cE '^report attn_dec_g[45]_u(64|32): ' "$O/attnbench.txt")" = 4
test "$(grep -cE '^identity (load-check attn_dec|attn_dec_g[45]_u(64|32) pass|FAIL)' "$O/attnbench.txt")" -ge 4
test "$(grep -cE '^attn_dec_g[45]_u(64|32) +[0-9]+ +[0-9]+ ' "$O/attnbench.txt")" = 48
echo '== raw kept ==' ; ls -la "$O" ; wc -l "$O"/*.txt
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
  --name "attn-bench${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
