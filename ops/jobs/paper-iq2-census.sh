#!/usr/bin/env bash
# MMLU census of IQ2_XXS at 4B in llama.cpp (run 5 of the paper table): the 38.87 of
# the 4B table is on the 2,280-question sample. Preregistered:
#   proofs/preregistration-paper-table-2026-09-25.md, stamped before this.
#
#   bash ops/jobs/paper-iq2-census.sh upload     prompts + scorer into the bucket
#   DRY_RUN=1 bash ops/jobs/paper-iq2-census.sh  prints and parses the job script
#   bash ops/jobs/paper-iq2-census.sh            launches
#
# The command of m4-iq2-cuda (6a951e9f45686a1580c1a70c, 142 s for 2,280 questions),
# unchanged but for the prompts: mmlu-prompts-FULL.jsonl, built by ops/mmlu_prompts.py
# from docs/data/mmlu-dumps/mmlu-4b-f16-FULL.csv with 14,042 / 14,042 qhash checked.
# -c 65536 -np 4 instead of 32768: four slots of 16k, the census carries longer
# five-shot prompts than the sample (KV ~9.6 GB at 4B, *computed*).
#
# Cost, estimated: ~15 min, ~$0.45, timeout 40m ($1.20).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-paper-table-2026-09-25.md
BUCKET=Pier-Jean/jobs-artifacts
SCRATCH=${SCRATCH:?SCRATCH=<dir holding mmlu-prompts-FULL.jsonl>}
PROMPTS=$SCRATCH/mmlu-prompts-FULL.jsonl
PSHA=9bd7a441888424f7a8b207366e45ede34d4b150e11a2e28a4bc1be0043cf5556
GSHA=19a8ed4946353b6fdc5d19ba9766ffe903923cb7aefe848dcbcd5dba1de27605
DIR=paper-iq2-census-2026-09-25
M=/out/$DIR
G=/out/m4-iq2-cuda-1b57b7d3/qwen3-4b-iq2xxs.gguf
DRY=${DRY_RUN:-0}

test "$(shasum -a 256 "$PROMPTS" | cut -d' ' -f1)" = "$PSHA" \
  || { echo "refused: $PROMPTS is not the prereg's prompts file" >&2; exit 1; }

if [ "${1:-}" = upload ]; then
  hf buckets cp "$PROMPTS" "hf://buckets/$BUCKET/$DIR/mmlu-prompts-FULL.jsonl"
  hf buckets cp ops/gguf_mmlu_thin.py "hf://buckets/$BUCKET/$DIR/gguf_mmlu_thin.py"
  hf buckets ls "hf://buckets/$BUCKET/$DIR/"
  exit 0
fi

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  hf buckets ls "hf://buckets/$BUCKET/$DIR/" | grep -q 'mmlu-prompts-FULL.jsonl' \
    || { echo "refused: run 'upload' first" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/$DIR/" | grep -q 'mmlu-4b-iq2xxs-FULL.csv'; then
    echo "refused: $DIR/mmlu-4b-iq2xxs-FULL.csv already exists" >&2; exit 1
  fi
fi

PRE=$(printf 'M=%q\nG=%q\nPSHA=%q\nGSHA=%q' "$M" "$G" "$PSHA" "$GSHA")

IFS= read -r -d '' BODY <<'JOB' || true
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee "$M/gpu.txt" || true
sha256sum "$G" "$M/mmlu-prompts-FULL.jsonl" | tee "$M/files.sha256"
test "$(sha256sum "$G" | cut -d' ' -f1)" = "$GSHA"
test "$(sha256sum "$M/mmlu-prompts-FULL.jsonl" | cut -d' ' -f1)" = "$PSHA"
S=$(command -v llama-server || echo /app/llama-server)
echo '== serve IQ2_XXS ==' ; date -u
$S -m "$G" -ngl 99 -c 65536 -np 4 --host 127.0.0.1 --port 8080 > "$M/server.log" 2>&1 &
SRV=$!
ok=0
for i in $(seq 1 200); do
  if curl -sf http://127.0.0.1:8080/health | grep -q ok; then ok=1; break; fi
  kill -0 $SRV 2>/dev/null || break
  sleep 3
done
[ "$ok" = 1 ] || { echo 'server never became healthy' >&2; tail -20 "$M/server.log" >&2; exit 1; }
echo 'server ready' ; date -u
python3 "$M/gguf_mmlu_thin.py" "$M/mmlu-prompts-FULL.jsonl" "$M/mmlu-4b-iq2xxs-FULL.csv" http://127.0.0.1:8080 4 2>&1 | tee "$M/score.txt" | tail -14
date -u
kill $SRV ; wait $SRV 2>/dev/null || true
test "$(grep -vc '^#' "$M/mmlu-4b-iq2xxs-FULL.csv")" = 14043
echo "dump ok: $M/mmlu-4b-iq2xxs-FULL.csv"
JOB

if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; nothing launched" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image ghcr.io/ggml-org/llama.cpp:full-cuda \
  --flavor l40sx1 --timeout 40m \
  --bucket "$BUCKET" --out-mount /out \
  --name paper-iq2-census \
  "$PRE" "$BODY"
