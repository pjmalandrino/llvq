#!/usr/bin/env bash
# GSM8K for the FP16 and AWQ references of one size, in their own engine, vLLM.
# Preregistered: proofs/preregistration-gsm8k-campaign-2026-09-26.md, stamped before this.
#
#   SIZE=4b|8b|14b bash ops/jobs/gsm8k-vllm.sh upload     the script into the bucket
#   SIZE=4b|8b|14b DRY_RUN=1 bash ops/jobs/gsm8k-vllm.sh  prints and parses the job script
#   SIZE=4b|8b|14b bash ops/jobs/gsm8k-vllm.sh            launches
#
# ops/gsm8k_vllm.py generates, graded by nothing: `gsm8kpair` grades every row with the
# Rust rules. One arm a process, the f16 cache deleted before the AWQ download at 14B
# (29.5 + 10 GB). The revisions are the ones `ops/awq_speed.py` pins, the same the MMLU
# census references read.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

SIZE=${SIZE:?SIZE=4b, 8b or 14b}
case "$SIZE" in
  4b)  BASE=Qwen/Qwen3-4B;  BASE_REV=1cfa9a7208912126459214e8b04321603b3df60c
       AWQ=Qwen/Qwen3-4B-AWQ;  AWQ_REV=74d4bd2bd4bff9cafc9345221320bffb08b406a3
       TIMEOUT=${TIMEOUT:-45m} ;;
  8b)  BASE=Qwen/Qwen3-8B;  BASE_REV=b968826d9c46dd6066d109eabc6255188de91218
       AWQ=Qwen/Qwen3-8B-AWQ;  AWQ_REV=4da05a8edb55c6046cce958586c33b61da07bb79
       TIMEOUT=${TIMEOUT:-60m} ;;
  14b) BASE=Qwen/Qwen3-14B; BASE_REV=40c069824f4251a91eefaf281ebe4c544efd3e18
       AWQ=Qwen/Qwen3-14B-AWQ; AWQ_REV=31c69efc29464b6bb0aee1398b5a7b50a99340c3
       TIMEOUT=${TIMEOUT:-90m} ;;
  *) echo "refused: SIZE=$SIZE, expected 4b, 8b or 14b" >&2; exit 1 ;;
esac
PREREG=proofs/preregistration-gsm8k-campaign-2026-09-26.md
BUCKET=Pier-Jean/jobs-artifacts
IMAGE=vllm/vllm-openai:v0.26.0
DIGEST=sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52
DATASET_REV=740312add88f781978c0658806c59bc2815b9866
DIR=gsm8k-vllm-$SIZE-2026-09-26${RETRY:+-r$RETRY}
O=/out/$DIR
SSHA=$(shasum -a 256 ops/gsm8k_vllm.py | cut -d' ' -f1)
DRY=${DRY_RUN:-0}

if [ "${1:-}" = upload ]; then
  hf buckets cp ops/gsm8k_vllm.py "hf://buckets/$BUCKET/$DIR/gsm8k_vllm.py"
  hf buckets ls "hf://buckets/$BUCKET/$DIR/"
  exit 0
fi

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  hf buckets ls "hf://buckets/$BUCKET/$DIR/" | grep -q 'gsm8k_vllm.py' \
    || { echo "refused: run 'upload' first" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/$DIR/" | grep -q '\.jsonl'; then
    echo "refused: $DIR already holds a dump" >&2; exit 1
  fi
fi

PRE=$(printf 'O=%q\nSIZE=%q\nSSHA=%q\nBASE=%q\nBASE_REV=%q\nAWQ=%q\nAWQ_REV=%q\nDATASET_REV=%q\nexport LLVQ_IMAGE_TAG=%q\nexport LLVQ_IMAGE_DIGEST=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$SIZE" "$SSHA" "$BASE" "$BASE_REV" "$AWQ" "$AWQ_REV" "$DATASET_REV" "$IMAGE" "$DIGEST")

IFS= read -r -d '' BODY <<'JOB' || true
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee "$O/gpu.txt"
python3 -c "import vllm; print('vllm', vllm.__version__)" | tee "$O/vllm-version.txt"
test "$(sha256sum "$O/gsm8k_vllm.py" | cut -d' ' -f1)" = "$SSHA"
df -h /tmp | tee "$O/disk.txt"
echo '== FP16 =='
date -u
python3 "$O/gsm8k_vllm.py" --arm f16 --model "$BASE" --revision "$BASE_REV" \
  --dataset-rev "$DATASET_REV" --out "$O/gsm8k-$SIZE-f16-vllm.jsonl" 2>&1 \
  | tee "$O/f16.txt" | grep -vE 'it/s|%\|' | tail -12
date -u
rm -rf "/tmp/hf/hub/models--${BASE//\//--}"
echo '== AWQ, awq_marlin =='
python3 "$O/gsm8k_vllm.py" --arm awq_marlin --model "$AWQ" --revision "$AWQ_REV" \
  --quantization awq_marlin --dataset-rev "$DATASET_REV" \
  --out "$O/gsm8k-$SIZE-awq-vllm.jsonl" 2>&1 | tee "$O/awq.txt" | grep -vE 'it/s|%\|' | tail -12
date -u
echo '== raw kept ==' ; ls -la "$O"
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
  --flavor l40sx1 --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "gsm8k-vllm-$SIZE${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
