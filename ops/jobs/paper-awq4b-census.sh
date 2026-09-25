#!/usr/bin/env bash
# MMLU census of AWQ w4 g128 at 4B, dequantized to f16 (run 4 of the paper table): the
# 70.04 of the 4B table is on the 2,280-question sample, every other 4B number on the
# 14,042-question census. Preregistered: proofs/preregistration-paper-table-2026-09-25.md.
#
#   DRY_RUN=1 bash ops/jobs/paper-awq4b-census.sh           prints and parses the job script
#   IMAGE_SHA=<space sha> bash ops/jobs/paper-awq4b-census.sh
#
# The checkpoint is Pier-Jean/qwen3-4b-awq-deq at 2c78de3b, built by ops/awq_dequant.py
# from Qwen/Qwen3-4B-AWQ 74d4bd2b; its three shards on the Hub equal the Mac's copy by
# sha256 (checked 2026-09-25). Same arm shape as arm C of census-8b.sh.
#
# Cost, estimated: ~23 min (f16-full at 4B, 2026-09-18), ~$0.70, timeout 45m ($1.35).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-paper-table-2026-09-25.md
BUCKET=Pier-Jean/jobs-artifacts
REPO_ID=Pier-Jean/qwen3-4b-awq-deq
REV=2c78de3b5a9572e4aadff02fe53f4ab4aad86c55
O=/out/paper-awq4b-census-2026-09-25
FP=a74a6d6213602979
DRY=${DRY_RUN:-0}

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  HUB=$(uv run --quiet --with huggingface_hub python -c \
    "from huggingface_hub import HfApi; print(HfApi().model_info('$REPO_ID').sha)")
  [ "$HUB" = "$REV" ] || { echo "refused: $REPO_ID moved to $HUB, pinned $REV" >&2; exit 1; }
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of the q4 rebuild>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "AWQ 4B deq $REPO_ID @ $HUB; image $NOW"
fi

PRE=$(printf 'O=%q\nM=%q\nFP=%q' "$O" "$REPO_ID" "$FP")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt" || true
echo '== oracle (hard rule 10) =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle-cuda.txt" | tail -2
grep -q MATCH "$O/oracle-cuda.txt"
echo '== MMLU dense, FULL split, AWQ w4 g128 dequantized, 4B ==' ; date -u
LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP="$O/mmlu-4b-awq-FULL.csv" \
  mmlu "$M" cuda 2>&1 | tee "$O/out.txt" | tail -10
date -u
D="$O/mmlu-4b-awq-FULL.csv"
grep -qxF '# dtype=f16' "$D"
grep -qxF '# limit=census' "$D"
grep -qxF '# alloc=flat, 100..1534 per subject, 14042 questions' "$D"
grep -qxF '# config=none' "$D"
grep -qxF '# arithmetic=dense reconstruction' "$D"
grep -qxF '# kv=f16' "$D"
test "$(tail -n 1 "$D")" = "# end fingerprint=$FP questions=14042"
echo "dump ok: $D"
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
  --flavor l40sx1 --timeout 45m \
  --bucket "$BUCKET" --out-mount /out \
  --name paper-awq4b-census \
  "$PRE" "$BODY"
