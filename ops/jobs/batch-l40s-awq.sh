#!/usr/bin/env bash
# The batching target, AWQ side: aggregate and per-sequence tok/s of the official AWQ w4g128 Qwen3 4B, 8B and 14B in
# vLLM at B = 1, 8, 16, 32, 64, 128 on one L40S, 256 greedy tokens a sequence after the 5-token door prompt
# (ops/awq_batch.py --size). One job, the three sizes one after another.
# Preregistered: proofs/preregistration-batch-l40s-2026-11-08.md, stamped before this.
#
#   bash ops/jobs/batch-l40s-awq.sh upload      copies the script into the bucket
#   DRY_RUN=1 bash ops/jobs/batch-l40s-awq.sh   parses the job script, launches nothing
#   bash ops/jobs/batch-l40s-awq.sh             launches
#
# Cost, estimated (l40sx1 $1.80/h): about 22 min, $0.66; timeout 45m, $1.35 at worst. The image pull (10 GB) and three
# model loads are most of it.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-batch-l40s-2026-11-08.md
BUCKET=Pier-Jean/jobs-artifacts
IMAGE=vllm/vllm-openai:v0.26.0
DIGEST=sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52
DIR=batch-l40s-awq-2026-11-08${RETRY:+-r$RETRY}
O=/out/$DIR
SSHA=$(shasum -a 256 ops/awq_batch.py | cut -d' ' -f1)
BATCHES=1,8,16,32,64,128
DRY=${DRY_RUN:-0}

if [ "${1:-}" = upload ]; then
  hf buckets cp ops/awq_batch.py "hf://buckets/$BUCKET/$DIR/awq_batch.py"
  hf buckets ls "hf://buckets/$BUCKET/$DIR/"
  exit 0
fi

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  hf buckets ls "hf://buckets/$BUCKET/$DIR/" | grep -q 'awq_batch.py' || { echo "refused: run 'upload' first" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/$DIR/" | grep -q 'awq-batch-'; then
    echo "refused: $DIR already holds results" >&2; exit 1
  fi
fi

PRE=$(printf 'O=%q\nSSHA=%q\nBATCHES=%q\nexport LLVQ_IMAGE_TAG=%q\nexport LLVQ_IMAGE_DIGEST=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$SSHA" "$BATCHES" "$IMAGE" "$DIGEST")

IFS= read -r -d '' BODY <<'JOB' || true
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
python3 -c "import vllm; print('vllm', vllm.__version__)" | tee "$O/vllm-version.txt"
test "$(sha256sum "$O/awq_batch.py" | cut -d' ' -f1)" = "$SSHA"
for SIZE in 4b 8b 14b; do
  echo "== AWQ $SIZE =="
  date -u
  python3 "$O/awq_batch.py" --size "$SIZE" --batches "$BATCHES" --json "$O/awq-batch-$SIZE.json" 2>&1 \
    | tee "$O/out-$SIZE.txt" | grep -v '^INFO\|^WARNING\|^(EngineCore' | tail -12
  test -s "$O/awq-batch-$SIZE.json"
done
date -u
JOB

if [ "$DRY" = 1 ]; then
  printf '%s\n' "set -euo pipefail
$PRE
$BODY" | bash -n && echo "DRY_RUN: job script parses; nothing launched" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image "$IMAGE" \
  --flavor l40sx1 --timeout 45m \
  --bucket "$BUCKET" --out-mount /out \
  --name "batch-l40s-awq${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
