#!/usr/bin/env bash
# AWQ w4 g128 tok/s at 8B and 14B in its own engine, vLLM (runs 6-7 of the paper table).
# Preregistered: proofs/preregistration-paper-table-2026-09-25.md, stamped before this.
# The protocol is ops/awq_speed.py's, frozen by preregistration-awq-vllm-2026-08-17.md;
# the command line is its §2.7, at another size.
#
#   SIZE=8b|14b bash ops/jobs/paper-awq-speed.sh upload     the script into the bucket
#   SIZE=8b|14b DRY_RUN=1 bash ops/jobs/paper-awq-speed.sh  prints and parses the job script
#   SIZE=8b|14b bash ops/jobs/paper-awq-speed.sh            launches
#
# 8B: f16 (16.4 GB) and awq_marlin (6.1 GB) fit one process, interleaved rounds, at
# 0.55 + 0.25 of the card. 14B: 29.5 + 10 GB do not, so one arm per process, the f16
# cache deleted before the AWQ download, then --merge, labelled "rounds NOT
# interleaved" by the script itself. awq_marlin only: at 4B the `awq` arm routed to
# Marlin anyway (awq-vllm-4b-2026-08-17.txt).
#
# Cost, estimated: 8B ~9 min $0.27, timeout 30m ($0.90); 14B ~15 min $0.45, timeout
# 40m ($1.20). Mostly image pull (10 GB) and checkpoint downloads (22 and 40 GB).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

SIZE=${SIZE:?SIZE=8b or SIZE=14b}
case "$SIZE" in
  8b)  TIMEOUT=30m ;;
  14b) TIMEOUT=40m ;;
  *) echo "refused: SIZE=$SIZE, expected 8b or 14b" >&2; exit 1 ;;
esac
PREREG=proofs/preregistration-paper-table-2026-09-25.md
BUCKET=Pier-Jean/jobs-artifacts
IMAGE=vllm/vllm-openai:v0.26.0
DIGEST=sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52
DIR=paper-awq-speed-$SIZE-2026-09-25
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
  if hf buckets ls "hf://buckets/$BUCKET/$DIR/" | grep -q "awq-speed-$SIZE.json"; then
    echo "refused: $DIR/awq-speed-$SIZE.json already exists" >&2; exit 1
  fi
fi

PRE=$(printf 'O=%q\nSIZE=%q\nSSHA=%q\nexport LLVQ_IMAGE_TAG=%q\nexport LLVQ_IMAGE_DIGEST=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$SIZE" "$SSHA" "$IMAGE" "$DIGEST")

IFS= read -r -d '' HEAD <<'JOB' || true
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee "$O/gpu.txt"
python3 -c "import vllm; print('vllm', vllm.__version__)" | tee "$O/vllm-version.txt"
test "$(sha256sum "$O/awq_speed.py" | cut -d' ' -f1)" = "$SSHA"
df -h /tmp | tee "$O/disk.txt"
JOB

IFS= read -r -d '' ARMS8 <<'JOB' || true
python3 "$O/awq_speed.py" --size 8b --arms f16,awq_marlin \
  --gpu-util-f16 0.55 --gpu-util-quant 0.25 --json "$O/awq-speed-8b.json" 2>&1 | tee "$O/out.txt" | tail -40
JOB

IFS= read -r -d '' ARMS14 <<'JOB' || true
python3 "$O/awq_speed.py" --size 14b --arms f16,awq_marlin --one-arm f16 \
  --gpu-util-f16 0.85 --json "$O/arm-f16.json" 2>&1 | tee "$O/out-f16.txt" | tail -25
rm -rf /tmp/hf/hub/models--Qwen--Qwen3-14B
python3 "$O/awq_speed.py" --size 14b --arms f16,awq_marlin --one-arm awq_marlin \
  --gpu-util-quant 0.85 --json "$O/arm-awq_marlin.json" 2>&1 | tee "$O/out-awq.txt" | tail -25
python3 "$O/awq_speed.py" --merge "$O/arm-f16.json" "$O/arm-awq_marlin.json" \
  --json "$O/awq-speed-14b.json" 2>&1 | tee "$O/out-merge.txt"
JOB

case "$SIZE" in
  8b)  BODY="$HEAD
$ARMS8" ;;
  14b) BODY="$HEAD
$ARMS14" ;;
esac

if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; nothing launched" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image "$IMAGE" \
  --flavor l40sx1 --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "paper-awq-speed-$SIZE" \
  "$PRE" "$BODY"
