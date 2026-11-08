#!/usr/bin/env bash
# The batched decode under the served v9 configs with a batch field, on one L40S, one job, each size in turn:
#   (a) batchrun check gemm: the GEMM on 4E v9's tiled int4 words, against the multi-row path, 1 to 128 rows;
#   (b) batchrun door at B = 1, 8, 32, 64, 128 under gemm (the kernel alone, the prompt fed through the batch);
#   (c) at 4B, batchrun ppl 2048 16 gemm: the band on the tiled words;
#   (d) the client's units: `llvq serve` under LLVQ_BATCH=128:1024 and ops/batch_serve_bench.py, B concurrent
#       /api/generate requests of 256 tokens at B = 1, 8, 16, 32, 64, 128: aggregate tok/s and tok/s a request. A
#       request alone decodes on 4E v9; from the second on, the GEMM.
# Preregistered: proofs/preregistration-batch-l40s-v3-2026-11-08.md, stamped before this.
#
#   DRY_RUN=1 bash ops/jobs/batch-l40s-v3.sh               parses the job script, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/batch-l40s-v3.sh    launches
#
# Binaries: oracle, batchrun, llvq, in the build list and the runtime COPY of ops/Dockerfile.cuda (check_image_lists).
# Every LLVQ_* variable is set on its own command line, never exported.
# Cost, estimated (l40sx1 $1.80/h): about 43 min, $1.29; timeout 65m, $1.95 at worst.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-batch-l40s-v3-2026-11-08.md
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/batch-l40s-v3-2026-11-08${RETRY:+-r$RETRY}
TIMEOUT=65m
DRY=${DRY_RUN:-0}
# size:bucket path:bytes:sha256:served door fingerprint (batch-l40s-2026-11-08):library tag
OBJECTS="4b:sealed-4b-2026-09-23/qwen3-4b-sealed.bin:1418224685:886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8:61d0654532d35120:qwen3:4b
8b:sealed-8b27-2026-09-24/qwen3-8b-sealed-B.bin:2815098745:7bdb9a5503518081b1f652a93215250898caae5cc1b6a2f33ee9fb6094565c13:c817aeaf2f5d1116:qwen3:8b
14b:sealed-14b-2026-09-23/qwen3-14b-sealed.bin:5087000541:61db37fe7ce8e6a57c5c59f257319c796574981ecb5e450a02c73dd3c17e2ca0:dc4f9ab9aa24c4d3:qwen3:14b"
for s in 4b 8b 14b; do
  c=configs/qwen3-$s-tetra-e4-4e-v9-batch.json
  [ "$(printf '%s\n' "$(cat "$c")" | shasum -a 256 | cut -d' ' -f1)" = "$(shasum -a 256 < "$c" | cut -d' ' -f1)" ] \
    || { echo "refused: $c does not end in exactly one newline" >&2; exit 1; }
done
CB_4B=$(cat configs/qwen3-4b-tetra-e4-4e-v9-batch.json)
CB_8B=$(cat configs/qwen3-8b-tetra-e4-4e-v9-batch.json)
CB_14B=$(cat configs/qwen3-14b-tetra-e4-4e-v9-batch.json)
CFG_SHAS=$(shasum -a 256 configs/qwen3-{4b,8b,14b}-tetra-e4-4e-v9-batch.json | awk '{print $1}' | tr '\n' ' ')
BENCH=$(cat ops/batch_serve_bench.py)
BENCH_SHA=$(shasum -a 256 < ops/batch_serve_bench.py | cut -d' ' -f1)

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  while IFS=: read -r size path bytes sha fp name tag; do
    REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/${path%/*}/" 2>/dev/null \
      | awk -v n="/${path##*/}" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
    [ "$REMOTE" = "$bytes" ] || { echo "refused: $path is ${REMOTE:-absent} B in the bucket, expected $bytes" >&2; exit 1; }
  done <<< "$OBJECTS"
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of the rebuild that carries this branch>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "bucket copies verified by size; configs $CFG_SHAS; bench $BENCH_SHA; image $NOW"
fi

PRE=$(printf 'O=%q\nOBJECTS=%q\nCB_4B=%q\nCB_8B=%q\nCB_14B=%q\nCFG_SHAS=%q\nBENCH=%q\nBENCH_SHA=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$OBJECTS" "$CB_4B" "$CB_8B" "$CB_14B" "$CFG_SHAS" "$BENCH" "$BENCH_SHA")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
echo '== gate 1: oracle =='
timeout 5m oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle.txt" | tail -2
grep -q MATCH "$O/oracle.txt"
echo '== gate 2: the objects, the configs, the bench =='
while IFS=: read -r size path bytes sha fp name tag; do
  test "$(stat -c %s "/out/$path")" = "$bytes"
  sha256sum "/out/$path" | tee -a "$O/files.sha256"
  test "$(sha256sum "/out/$path" | cut -d' ' -f1)" = "$sha"
done <<< "$OBJECTS"
printf '%s\n' "$CB_4B" > "$O/qwen3-4b-tetra-e4-4e-v9-batch.json"
printf '%s\n' "$CB_8B" > "$O/qwen3-8b-tetra-e4-4e-v9-batch.json"
printf '%s\n' "$CB_14B" > "$O/qwen3-14b-tetra-e4-4e-v9-batch.json"
GOT=$(cd "$O" && sha256sum qwen3-{4b,8b,14b}-tetra-e4-4e-v9-batch.json | awk '{print $1}' | tr '\n' ' ')
test "$GOT" = "$CFG_SHAS"
printf '%s\n' "$BENCH" > "$O/batch_serve_bench.py"
test "$(sha256sum < "$O/batch_serve_bench.py" | cut -d' ' -f1)" = "$BENCH_SHA"
echo "configs and bench at their bytes"

while IFS=: read -r size path bytes sha FP name tag; do
  S=/out/$path; PB=$O/qwen3-$size-tetra-e4-4e-v9-batch.json; MODEL=$name:$tag
  case $size in 4b) T=1;; 8b) T=2;; 14b) T=3;; esac
  echo "== $size (a): batchrun check gemm on the tiled words =="
  date -u
  LLVQ_CONFIG="$PB" timeout $((6 * T))m batchrun "$S" check gemm > "$O/check-$size.txt" 2>&1 || { tail -20 "$O/check-$size.txt"; exit 1; }
  grep -E '^served config|^int4 layout|^tv_bg|^check: ' "$O/check-$size.txt"
  grep -qE '^int4 layout: tiled' "$O/check-$size.txt" || { echo "gate: $size did not load its int4 words tiled" >&2; exit 1; }

  echo "== $size (b): the batched door, gemm =="
  date -u
  LLVQ_CONFIG="$PB" LLVQ_BATCH_REF="$FP" timeout $((8 * T))m batchrun "$S" door 1,8,32,64,128 gemm 256 3 > "$O/door-batch-$size.txt" 2>&1 \
    || { tail -20 "$O/door-batch-$size.txt"; exit 1; }
  grep -A12 '^batched door' "$O/door-batch-$size.txt"

  if [ "$size" = 4b ]; then
    echo "== 4b (c): decode-path perplexity on the tiled words, gemm against rows =="
    date -u
    LLVQ_CONFIG="$PB" timeout 10m batchrun "$S" ppl 2048 16 gemm > "$O/ppl-2048x16-$size.txt" 2>&1 || { tail -20 "$O/ppl-2048x16-$size.txt"; exit 1; }
    grep -E '^batched ppl|^ppl|^T8' "$O/ppl-2048x16-$size.txt"
  fi

  echo "== $size (d): llvq serve under LLVQ_BATCH=128:1024, concurrent requests =="
  date -u
  H=/tmp/llvq-home-$size
  LLVQ_HOME=$H timeout 5m llvq pull "$MODEL" --from "$S" > "$O/pull-$size.txt" 2>&1 || { tail -5 "$O/pull-$size.txt"; exit 1; }
  LLVQ_HOME=$H LLVQ_HOST=127.0.0.1:11435 LLVQ_BATCH=128:1024 LLVQ_KEEP_ALIVE=-1 llvq serve > "$O/serve-$size.txt" 2>&1 &
  SP=$!
  # The runtime image has no curl: python3 asks.
  UP='import sys,urllib.request
try: urllib.request.urlopen("http://127.0.0.1:11435/api/version",timeout=2); sys.exit(0)
except Exception: sys.exit(1)'
  for i in $(seq 1 60); do python3 -c "$UP" && break; sleep 2; done
  timeout $((10 * T))m python3 "$O/batch_serve_bench.py" --host 127.0.0.1:11435 --model "$MODEL" \
    --batches 1,8,16,32,64,128 --tokens 256 --rounds 2 --json "$O/serve-bench-$size.json" > "$O/serve-bench-$size.txt" 2>&1 \
    || { tail -20 "$O/serve-bench-$size.txt"; tail -20 "$O/serve-$size.txt"; kill $SP; exit 1; }
  grep -A10 '^llvq serve' "$O/serve-bench-$size.txt"
  python3 -c 'import urllib.request;print(urllib.request.urlopen("http://127.0.0.1:11435/api/ps").read().decode())' > "$O/ps-$size.json" || true
  kill $SP; wait $SP 2>/dev/null || true
  grep -E 'batch:|tv_bg|4E: persistent module' "$O/serve-$size.txt" | head -8 || true
done <<< "$OBJECTS"
date -u
ls -la "$O"
JOB

if [ "$DRY" = 1 ]; then
  printf '%s\n' "set -euo pipefail
$PRE
$BODY" | bash -n && echo "DRY_RUN: job script parses; nothing launched; configs $CFG_SHAS; bench $BENCH_SHA" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "batch-l40s-v3${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
