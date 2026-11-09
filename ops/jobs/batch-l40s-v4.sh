#!/usr/bin/env bash
# Batch v4, the 4B only, one job: what slowed batch-l40s-v3, the int4 word order or the two-CTA entries.
#   (a) batchrun check gemm under the v9-batch config (the served entries, back to v2's, on the tiled words);
#   (b) batchrun door 1,32,64,128 gemm,gemm-occ under the batch config (file order) and under the v9-batch config
#       (tiled): two arms by two orders in one job;
#   (c) llvq serve under LLVQ_BATCH=128:1024 (v9-batch) and ops/batch_serve_bench.py at B = 1, 8, 32, 128.
# Preregistered: proofs/preregistration-batch-l40s-v4-2026-11-08.md, stamped before this.
#
#   DRY_RUN=1 bash ops/jobs/batch-l40s-v4.sh               parses the job script, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/batch-l40s-v4.sh    launches
#
# Binaries: oracle, batchrun, llvq, in the build list and the runtime COPY of ops/Dockerfile.cuda (check_image_lists).
# Every LLVQ_* variable is set on its own command line, never exported.
# Cost, estimated (l40sx1 $1.80/h): about 17 min, $0.51; timeout 30m, $0.90 at worst.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-batch-l40s-v4-2026-11-08.md
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/batch-l40s-v4-2026-11-08${RETRY:+-r$RETRY}
TIMEOUT=30m
DRY=${DRY_RUN:-0}
# size:bucket path:bytes:sha256:served door fingerprint (batch-l40s-2026-11-08):library tag
OBJECTS="4b:sealed-4b-2026-09-23/qwen3-4b-sealed.bin:1418224685:886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8:61d0654532d35120:qwen3:4b"
for c in configs/qwen3-4b-tetra-e4-4e-v9-batch.json configs/qwen3-4b-tetra-e4-batch.json; do
  [ "$(printf '%s\n' "$(cat "$c")" | shasum -a 256 | cut -d' ' -f1)" = "$(shasum -a 256 < "$c" | cut -d' ' -f1)" ] \
    || { echo "refused: $c does not end in exactly one newline" >&2; exit 1; }
done
CB_4B=$(cat configs/qwen3-4b-tetra-e4-4e-v9-batch.json)
CF_4B=$(cat configs/qwen3-4b-tetra-e4-batch.json)
CFG_SHAS=$(shasum -a 256 configs/qwen3-4b-tetra-e4-4e-v9-batch.json configs/qwen3-4b-tetra-e4-batch.json | awk '{print $1}' | tr '\n' ' ')
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

PRE=$(printf 'O=%q\nOBJECTS=%q\nCB_4B=%q\nCF_4B=%q\nCFG_SHAS=%q\nBENCH=%q\nBENCH_SHA=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$OBJECTS" "$CB_4B" "$CF_4B" "$CFG_SHAS" "$BENCH" "$BENCH_SHA")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
echo '== gate 1: oracle =='
timeout 5m oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle.txt" | tail -2
grep -q MATCH "$O/oracle.txt"
echo '== gate 2: the object, the configs, the bench =='
IFS=: read -r size path bytes sha FP name tag <<< "$OBJECTS"
S=/out/$path; MODEL=$name:$tag
test "$(stat -c %s "$S")" = "$bytes"
sha256sum "$S" | tee -a "$O/files.sha256"
test "$(sha256sum "$S" | cut -d' ' -f1)" = "$sha"
PB=$O/qwen3-4b-tetra-e4-4e-v9-batch.json; PF=$O/qwen3-4b-tetra-e4-batch.json
printf '%s\n' "$CB_4B" > "$PB"; printf '%s\n' "$CF_4B" > "$PF"
GOT=$(sha256sum "$PB" "$PF" | awk '{print $1}' | tr '\n' ' ')
test "$GOT" = "$CFG_SHAS"
printf '%s\n' "$BENCH" > "$O/batch_serve_bench.py"
test "$(sha256sum < "$O/batch_serve_bench.py" | cut -d' ' -f1)" = "$BENCH_SHA"

echo '== (a) batchrun check gemm on the tiled words, the served entries =='
date -u
LLVQ_CONFIG="$PB" timeout 6m batchrun "$S" check gemm > "$O/check.txt" 2>&1 || { tail -20 "$O/check.txt"; exit 1; }
grep -E '^int4 layout|^tv_bg|^check: ' "$O/check.txt"
grep -qE '^int4 layout: tiled' "$O/check.txt" || { echo "gate: the int4 words were not loaded tiled" >&2; exit 1; }

for C in file:"$PF" tiled:"$PB"; do
  W=${C%%:*}; P=${C#*:}
  echo "== (b) door, gemm and gemm-occ, int4 words in $W order =="
  date -u
  LLVQ_CONFIG="$P" LLVQ_BATCH_REF="$FP" timeout 8m batchrun "$S" door 1,32,64,128 gemm,gemm-occ 256 3 > "$O/door-$W.txt" 2>&1 \
    || { tail -20 "$O/door-$W.txt"; exit 1; }
  grep -E '^int4 layout' "$O/door-$W.txt" || true
  grep -A12 '^batched door' "$O/door-$W.txt"
done

echo '== (c) llvq serve under LLVQ_BATCH=128:1024, concurrent requests =='
date -u
H=/tmp/llvq-home
LLVQ_HOME=$H timeout 5m llvq pull "$MODEL" --from "$S" > "$O/pull.txt" 2>&1 || { tail -5 "$O/pull.txt"; exit 1; }
LLVQ_HOME=$H LLVQ_HOST=127.0.0.1:11435 LLVQ_BATCH=128:1024 LLVQ_KEEP_ALIVE=-1 llvq serve > "$O/serve.txt" 2>&1 &
SP=$!
UP='import sys,urllib.request
try: urllib.request.urlopen("http://127.0.0.1:11435/api/version",timeout=2); sys.exit(0)
except Exception: sys.exit(1)'
for i in $(seq 1 60); do python3 -c "$UP" && break; sleep 2; done
timeout 10m python3 "$O/batch_serve_bench.py" --host 127.0.0.1:11435 --model "$MODEL" \
  --batches 1,8,32,128 --tokens 256 --rounds 2 --json "$O/serve-bench.json" > "$O/serve-bench.txt" 2>&1 \
  || { tail -20 "$O/serve-bench.txt"; tail -20 "$O/serve.txt"; kill $SP; exit 1; }
grep -A8 '^llvq serve' "$O/serve-bench.txt"
kill $SP; wait $SP 2>/dev/null || true
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
  --name "batch-l40s-v4${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
