#!/usr/bin/env bash
# A second architecture, the vLLM half. Preregistered:
#   proofs/preregistration-a100-2026-10-06.md
#
# FP16 and AWQ (Marlin) at 4B, 8B and 14B on one a100-large, with the script
# and the pinned image of paper-awq-speed.sh. 80 GB holds both arms at every
# size, so each size runs them interleaved in one process, where the L40S had
# to split the 14B.
#
# Cost: about 20 min, $0.83 (estimated), timeout 35 min, $1.46 at worst.
# `a100-vllm.sh upload` first, then the launch. DRY_RUN=1 parses only.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-a100-2026-10-06.md
BUCKET=Pier-Jean/jobs-artifacts
IMAGE=vllm/vllm-openai:v0.26.0
DIGEST=sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52
DIR=a100-vllm-2026-10-06
O=/out/$DIR
SSHA=$(shasum -a 256 ops/awq_speed.py | cut -d' ' -f1)
DRY=${DRY_RUN:-0}

if [ "${1:-}" = upload ]; then
  hf buckets cp ops/awq_speed.py "hf://buckets/$BUCKET/$DIR/awq_speed.py"
  hf buckets ls "hf://buckets/$BUCKET/$DIR/"
  exit 0
fi
if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  hf buckets ls "hf://buckets/$BUCKET/$DIR/" | grep -q 'awq_speed.py' \
    || { echo "refused: run 'upload' first" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/$DIR/" | grep -q 'awq-speed-.*\.json'; then
    echo "refused: $DIR already holds a result" >&2; exit 1
  fi
fi

PRE=$(printf 'O=%q\nSSHA=%q\nexport LLVQ_IMAGE_TAG=%q\nexport LLVQ_IMAGE_DIGEST=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$SSHA" "$IMAGE" "$DIGEST")

IFS= read -r -d '' BODY <<'JOB' || true
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader | tee "$O/gpu.txt"
grep -q 'A100' "$O/gpu.txt"
python3 -c "import vllm; print('vllm', vllm.__version__)" | tee "$O/vllm-version.txt"
# Gate 7.
grep -q '^vllm 0\.26\.0' "$O/vllm-version.txt"
test "$(sha256sum "$O/awq_speed.py" | cut -d' ' -f1)" = "$SSHA"
df -h /tmp | tee "$O/disk.txt"
for S in 4b 8b 14b; do
  case "$S" in
    4b)  U16=0.20 ; UQ=0.10 ;;
    8b)  U16=0.35 ; UQ=0.15 ;;
    14b) U16=0.50 ; UQ=0.25 ;;
  esac
  echo "== $S: f16 and awq_marlin, interleaved =="
  date -u
  python3 "$O/awq_speed.py" --size "$S" --arms f16,awq_marlin \
    --gpu-util-f16 "$U16" --gpu-util-quant "$UQ" --json "$O/awq-speed-$S.json" 2>&1 \
    | tee "$O/out-$S.txt" | tail -30
  test -s "$O/awq-speed-$S.json"
  rm -rf /tmp/hf/hub/models--Qwen--Qwen3-*
done
date -u
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
  --flavor a100-large --any-flavor --timeout 35m \
  --bucket "$BUCKET" --out-mount /out \
  --name "a100-vllm" \
  "$PRE" "$BODY"
