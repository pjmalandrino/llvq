#!/usr/bin/env bash
# FlashInfer's batch-1 GQA decode attention on one L40S, Qwen3-4B geometry (32 query heads, 8 KV
# heads, head dim 128, 36 layers, f16 cache) at 2,048, 4,096 and 8,192 positions: the read rate
# eta a good decode-attention kernel reaches on this card. No code of ours runs, so no oracle.
# Preregistered: proofs/preregistration-fi-decode-anchor-2026-11-04.md, stamped before this.
#
#   bash ops/jobs/fi-decode-anchor.sh upload     the bench script into the bucket
#   DRY_RUN=1 bash ops/jobs/fi-decode-anchor.sh  prints and parses the job script, checks the pins
#   bash ops/jobs/fi-decode-anchor.sh            launches
#
# Image: vllm/vllm-openai:v0.26.0, the tag every vLLM job of the repository ran, at the manifest
# digest recorded in docs/mesures/awq-vllm-4b-2026-08-17.txt. The launcher resolves the tag on
# Docker Hub and refuses when it no longer points at that digest. Inside the job, every package the
# bench touches is gated on its exact version before anything is timed: vllm 0.26.0, torch 2.11.0,
# CUDA 13.0, flashinfer-python, flashinfer-cubin and flashinfer-jit-cache 0.6.14 (the pins of
# requirements/cuda.txt and docker/Dockerfile at vLLM tag v0.26.0). Nothing is pip-installed.
# The job script goes inline to `ops/run.py bench` (`bash -lc`); the bench script reaches the job
# through the bucket mount, checked by sha256. No Space build, so UPLOAD_ALLOW is not involved.
#
# Cost, estimated: 4 to 10 min billed on l40sx1 at $1.80/h, $0.12 to $0.30; timeout 20 min, $0.60.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-fi-decode-anchor-2026-11-04.md
BUCKET=Pier-Jean/jobs-artifacts
IMAGE=vllm/vllm-openai:v0.26.0
DIGEST=sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52
DIR=fi-decode-anchor-2026-11-04${RETRY:+-r$RETRY}
O=/out/$DIR
SSHA=$(shasum -a 256 ops/fi_decode_bench.py | cut -d' ' -f1)
DRY=${DRY_RUN:-0}

if [ "${1:-}" = upload ]; then
  hf buckets cp ops/fi_decode_bench.py "hf://buckets/$BUCKET/$DIR/fi_decode_bench.py"
  hf buckets ls "hf://buckets/$BUCKET/$DIR/"
  exit 0
fi

# The tag must still name the recorded digest. Docker Hub's tag endpoint returns the index digest.
HUB_DIGEST=$(curl -fsS --max-time 20 \
    "https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags/${IMAGE##*:}" 2>/dev/null \
  | python3 -I -c 'import json,sys; print(json.load(sys.stdin).get("digest",""))' 2>/dev/null || true)
if [ "$HUB_DIGEST" = "$DIGEST" ]; then
  echo "image: $IMAGE resolves to $DIGEST on Docker Hub" >&2
elif [ "$DRY" = 1 ]; then
  echo "DRY_RUN WARNING: $IMAGE resolves to '${HUB_DIGEST:-<no answer>}', expected $DIGEST" >&2
else
  echo "refused: $IMAGE resolves to '${HUB_DIGEST:-<no answer>}', expected $DIGEST" >&2; exit 1
fi

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  hf buckets ls "hf://buckets/$BUCKET/$DIR/" | grep -q 'fi_decode_bench.py' \
    || { echo "refused: run 'upload' first" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/$DIR/" | grep -q 'fi-decode.json'; then
    echo "refused: $DIR/fi-decode.json already exists" >&2; exit 1
  fi
fi

PRE=$(printf 'O=%q\nSSHA=%q\nexport LLVQ_IMAGE_TAG=%q\nexport LLVQ_IMAGE_DIGEST=%q\nexport HF_HOME=/tmp/hf\nexport FLASHINFER_CUDA_ARCH_LIST=8.9' \
  "$O" "$SSHA" "$IMAGE" "$DIGEST")

IFS= read -r -d '' BODY <<'JOB' || true
nvidia-smi --query-gpu=name,compute_cap,driver_version,clocks.max.sm,clocks.max.mem,memory.total --format=csv,noheader | tee "$O/gpu.txt"
# Gate 1.
grep -q 'L40S' "$O/gpu.txt"
# Gate 2: every package at its pinned version, before anything is timed.
python3 - <<'PY' | tee "$O/versions.txt"
import importlib.metadata as md
import torch
for p in ("vllm", "torch", "flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache"):
    try:
        print(p, md.version(p))
    except md.PackageNotFoundError:
        print(p, "MISSING")
print("torch.cuda", torch.version.cuda)
PY
grep -qx 'vllm 0\.26\.0' "$O/versions.txt"
grep -qE '^torch 2\.11\.0(\+cu130)?$' "$O/versions.txt"
grep -qx 'torch\.cuda 13\.0' "$O/versions.txt"
grep -qE '^flashinfer-python 0\.6\.14$' "$O/versions.txt"
grep -qE '^flashinfer-cubin 0\.6\.14$' "$O/versions.txt"
grep -qE '^flashinfer-jit-cache 0\.6\.14(\+cu130)?$' "$O/versions.txt"
pip freeze 2>/dev/null > "$O/pip-freeze.txt" || true
{ command -v nvcc && nvcc --version | tail -2; } | tee "$O/nvcc.txt" || echo "nvcc: absent" | tee "$O/nvcc.txt"
test "$(sha256sum "$O/fi_decode_bench.py" | cut -d' ' -f1)" = "$SSHA"
env | grep '^LLVQ_' | sort | tee "$O/env.txt"
date -u
python3 "$O/fi_decode_bench.py" --json "$O/fi-decode.json" 2>&1 | tee "$O/out.txt" | tail -40
date -u
test -s "$O/fi-decode.json"
echo '== raw kept ==' ; ls -la "$O"
JOB

if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; nothing launched" >&2
  test -f "$PREREG" && echo "DRY_RUN: $PREREG present" >&2 \
    || echo "DRY_RUN WARNING: $PREREG missing" >&2
  test -f "$PREREG.ots" && echo "DRY_RUN: $PREREG.ots present" >&2 \
    || echo "DRY_RUN: $PREREG.ots not yet stamped (a launch refuses without it)" >&2
  echo "DRY_RUN: bench sha256 $SSHA, output $O, bucket $BUCKET" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image "$IMAGE" \
  --flavor l40sx1 --timeout 20m \
  --bucket "$BUCKET" --out-mount /out \
  --name "fi-decode-anchor${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
