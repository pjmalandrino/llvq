#!/usr/bin/env bash
# Stage 4: the served CUDA Tetra kernel reached from PyTorch, on a card.
# Preregistered: proofs/preregistration-hf-cuda-2026-09-30.md, stamped before this.
#
#   bash ops/jobs/hf-cuda-4b.sh upload     the sources and the reference into the bucket
#   DRY_RUN=1 bash ops/jobs/hf-cuda-4b.sh  prints and parses the job script, launches nothing
#   bash ops/jobs/hf-cuda-4b.sh            launches
#
# The image is `nvidia/cuda:12.4.1-devel`, the same base our own build stage uses, because
# our runtime image carries no nvcc and cannot compile a torch extension. So torch arrives
# by pip, which is three minutes and about $0.09 of the estimate.
#
# The model is read from the bucket and not produced here: `hfpack` is a Rust binary and this
# image has no Rust. It was packed on the Mac at commit 0c4197f and its digests travel with it.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-hf-cuda-2026-09-30.md
BUCKET=Pier-Jean/jobs-artifacts
DIR=hf-cuda-4b-2026-09-30${RETRY:+-r$RETRY}
O=/out/$DIR
IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
TIMEOUT=${TIMEOUT:-40m}
NEW=${NEW:-64}
DRY=${DRY_RUN:-0}
REF=/tmp/run-tokens-f32.json

if [ "${1:-}" = upload ]; then
  # The sources, in the repository's own shape: `tv_tetra48_h.cu` includes
  # `../../llvq-cuda/kernels/...`, so the tree matters and a flat copy would not build.
  TAR=$(mktemp -d)/llvq-hf-src.tgz
  tar czf "$TAR" \
    --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' \
    llvq-hf/llvqhf llvq-hf/pyproject.toml \
    llvq-llm/kernels llvq-cuda/kernels \
    ops/hf_cuda_check.py
  echo "sources $(du -h "$TAR" | cut -f1)"
  hf buckets cp "$TAR" "hf://buckets/$BUCKET/$DIR/llvq-hf-src.tgz"
  test -f "$REF" || { echo "refused: $REF missing, run bin/run with LLVQ_RUN_DUMP first" >&2; exit 1; }
  hf buckets cp "$REF" "hf://buckets/$BUCKET/$DIR/run-tokens-f32.json"
  hf buckets ls "hf://buckets/$BUCKET/$DIR/"
  exit 0
fi

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  for f in llvq-hf-src.tgz run-tokens-f32.json; do
    hf buckets ls "hf://buckets/$BUCKET/$DIR/" | grep -q "$f" \
      || { echo "refused: run 'upload' first, $f is not in the bucket" >&2; exit 1; }
  done
  hf buckets ls "hf://buckets/$BUCKET/$DIR/model/" | grep -q 'model.safetensors' \
    || { echo "refused: the packed model is not in the bucket" >&2; exit 1; }
fi

SSHA=$(shasum -a 256 ops/hf_cuda_check.py | cut -d' ' -f1)
PRE=$(printf 'O=%q\nNEW=%q\nSSHA=%q\nexport HF_HOME=/tmp/hf\nexport TORCH_EXTENSIONS_DIR=/tmp/torchext\nexport LLVQ_HF_FUSED=1\nexport LLVQ_HF_TILE=64' \
  "$O" "$NEW" "$SSHA")

IFS= read -r -d '' BODY <<'JOB' || true
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee "$O/gpu.txt"
nvcc --version | tail -2 | tee "$O/nvcc.txt"
df -h /tmp | tee "$O/disk.txt"
# The base image is CUDA and nothing else: python, pip, then torch.
apt-get update -qq && apt-get install -y -qq python3-pip python3-dev git >/dev/null
pip install --quiet --no-input torch transformers safetensors numpy ninja 2>&1 | tail -2
python3 -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)" \
  | tee "$O/versions.txt"
# The sources, in the repository's shape, and the script checked against its own digest.
mkdir -p /tmp/src && tar xzf "$O/llvq-hf-src.tgz" -C /tmp/src
test "$(sha256sum /tmp/src/ops/hf_cuda_check.py | cut -d' ' -f1)" = "$SSHA"
export LLVQ_REPO=/tmp/src
export PYTHONPATH=/tmp/src/llvq-hf
python3 /tmp/src/ops/hf_cuda_check.py \
  --model "$O/model" --ref "$O/run-tokens-f32.json" --new "$NEW" \
  --dump "$O/cuda-tokens.json" 2>&1 | tee "$O/check.txt"
echo '== raw kept ==' ; ls -la "$O"
JOB

if [ "$DRY" = 1 ]; then
  printf '%s\n' "set -euo pipefail
$PRE
$BODY" | bash -n && echo "DRY_RUN: the job script parses; nothing launched" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image "$IMAGE" \
  --flavor l40sx1 --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "hf-cuda-4b${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
