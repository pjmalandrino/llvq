#!/usr/bin/env bash
# DRAFT, never run. Job script sketch for prereg-attention-1 (FlashInfer single-decode read rate on
# the L40S, Qwen3-4B geometry). Modelled on ops/jobs/paper-awq-speed.sh. Not repository code; it
# would go to ops/jobs/fi-decode-anchor.sh only after the operator's go, with the prereg stamped.
#
#   bash ops/jobs/fi-decode-anchor.sh upload     the bench script into the bucket
#   DRY_RUN=1 bash ops/jobs/fi-decode-anchor.sh  prints and parses the job script
#   bash ops/jobs/fi-decode-anchor.sh            launches (needs the stamped prereg)
#
# Image: the repository's pinned vLLM image, the tag every vLLM job has used, with its manifest
# digest recorded beside it (docs/mesures/awq-vllm-4b-2026-08-17.txt). The job cannot read its own
# digest, so it gates on the versions it finds, as the vLLM jobs do.
# Cost, estimated: 5 to 14 min billed on l40sx1 at $1.80/h, $0.15 to $0.42 (pull 1-2 min, JIT of up
# to four FlashInfer modules 0-8 min if flashinfer-jit-cache does not hold them, bench 2-3 min);
# timeout 25 min, $0.75.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
PREREG=proofs/preregistration-fi-decode-anchor-YYYY-MM-DD.md
BUCKET=Pier-Jean/jobs-artifacts
IMAGE=vllm/vllm-openai:v0.26.0
DIGEST=sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52
DIR=fi-decode-anchor-YYYY-MM-DD
O=/out/$DIR
SSHA=$(shasum -a 256 ops/fi_decode_bench.py | cut -d' ' -f1)
DRY=${DRY_RUN:-0}

if [ "${1:-}" = upload ]; then
  hf buckets cp ops/fi_decode_bench.py "hf://buckets/$BUCKET/$DIR/fi_decode_bench.py"
  hf buckets ls "hf://buckets/$BUCKET/$DIR/"
  exit 0
fi

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  hf buckets ls "hf://buckets/$BUCKET/$DIR/" | grep -q 'fi_decode_bench.py' \
    || { echo "refused: run 'upload' first" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/$DIR/" | grep -q 'fi-decode.json'; then
    echo "refused: $DIR/fi-decode.json already exists" >&2; exit 1
  fi
fi

PRE=$(printf 'O=%q\nSSHA=%q\nexport LLVQ_IMAGE_TAG=%q\nexport LLVQ_IMAGE_DIGEST=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$SSHA" "$IMAGE" "$DIGEST")

IFS= read -r -d '' BODY <<'JOB' || true
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,memory.total --format=csv | tee "$O/gpu.txt"
grep -q 'L40S' "$O/gpu.txt"
python3 -c "import vllm, flashinfer, torch; print('vllm', vllm.__version__); print('flashinfer', flashinfer.__version__); print('torch', torch.__version__, torch.version.cuda)" \
  | tee "$O/versions.txt"
grep -qx 'vllm 0.26.0' "$O/versions.txt"
grep -qx 'flashinfer 0.6.14' "$O/versions.txt"
test "$(sha256sum "$O/fi_decode_bench.py" | cut -d' ' -f1)" = "$SSHA"
date -u
python3 "$O/fi_decode_bench.py" --json "$O/fi-decode.json" 2>&1 | tee "$O/out.txt" | tail -60
date -u
test -s "$O/fi-decode.json"
echo "anchor ok: $O"
JOB

if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; nothing launched" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image "$IMAGE" \
  --flavor l40sx1 --timeout 25m \
  --bucket "$BUCKET" --out-mount /out \
  --name fi-decode-anchor \
  "$PRE" "$BODY"
