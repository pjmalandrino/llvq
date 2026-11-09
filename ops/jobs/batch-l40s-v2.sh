#!/usr/bin/env bash
# The batched GEMM's second pass (f16 activation, 128-row tiles) on one L40S, one job: the gate again, the step's
# attribution, the bench at 4B, 8B and 14B.
#   4B  (a) batchrun kernels: the projections alone at B = 1, 8, 32, 128, per group, ms a step;
#       (b) batchrun check gemm: the GEMM against the multi-row path on the real weights, 1 to 128 rows, the groups;
#       (c) batchrun door at B = 1, 8, 16, 32, 64, 128 under gemm, against the served fingerprint of batch-l40s;
#       (d) decode-path perplexity, gemm against rows: 16 windows of 2,048 (the 16-row entry) and 128 windows of 256
#           (the 128-row entry), the T8 band;
#       (e) the 50 GSM8K pilot problems in one batch of 50 under gemm, against the served pilot dump.
#   8B, 14B: (a), (b), (c) under gemm only, and (d) at 16 windows of 2,048.
# Preregistered: proofs/preregistration-batch-l40s-v2-2026-11-08.md, stamped before this.
#
#   DRY_RUN=1 bash ops/jobs/batch-l40s-v2.sh               parses the job script, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/batch-l40s-v2.sh    launches
#
# Every LLVQ_* variable is set on its own command line, never exported. Binaries: oracle, fusedrun, batchrun, all in
# the build list and the runtime COPY of ops/Dockerfile.cuda; configs/** and llvq-*/** in UPLOAD_ALLOW.
# Cost, estimated (l40sx1 $1.80/h): about 45 min, $1.35; timeout 75m, $2.25 at worst.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-batch-l40s-v2-2026-11-08.md
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/batch-l40s-v2-2026-11-08${RETRY:+-r$RETRY}
TIMEOUT=75m
DRY=${DRY_RUN:-0}
DATASET_REV=740312add88f781978c0658806c59bc2815b9866
BATCHES=1,8,16,32,64,128
# size:bucket path:bytes:sha256:the served door's fingerprint (batch-l40s-2026-11-08)
OBJECTS="4b:sealed-4b-2026-09-23/qwen3-4b-sealed.bin:1418224685:886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8:61d0654532d35120
8b:sealed-8b27-2026-09-24/qwen3-8b-sealed-B.bin:2815098745:7bdb9a5503518081b1f652a93215250898caae5cc1b6a2f33ee9fb6094565c13:c817aeaf2f5d1116
14b:sealed-14b-2026-09-23/qwen3-14b-sealed.bin:5087000541:61db37fe7ce8e6a57c5c59f257319c796574981ecb5e450a02c73dd3c17e2ca0:dc4f9ab9aa24c4d3"
DOOR_FP_4B=61d0654532d35120
REPLAY_FP=94c7b98876288901
PILOT=docs/mesures/served-v4-2026-11-05-brut/gsm8k-4b-v4-n50.jsonl
PILOT_SHA=a7203fa6a7f19faf91b46a1907ebe08989c7354582f2c10e306a40e557cfcab8
REF_PY='import hashlib,json,sys
ls=[json.loads(l) for l in open(sys.argv[1],encoding="utf-8").read().split("\n") if l][1:]
print(",".join("%d:%d:%s"%(d["index"],int(d["correct"]),hashlib.sha256(d["completion"].encode()).hexdigest()[:12]) for d in ls if not d.get("end")))'

[ "$(shasum -a 256 < "$PILOT" | cut -d' ' -f1)" = "$PILOT_SHA" ] || { echo "refused: $PILOT is not $PILOT_SHA" >&2; exit 1; }
PILOT_REF=$(python3 -c "$REF_PY" "$PILOT")
CFGS=""
for s in 4b 8b 14b; do
  for c in configs/qwen3-$s-tetra-e4-4e-v9.json configs/qwen3-$s-tetra-e4-batch.json; do
    [ "$(printf '%s\n' "$(cat "$c")" | shasum -a 256 | cut -d' ' -f1)" = "$(shasum -a 256 < "$c" | cut -d' ' -f1)" ] \
      || { echo "refused: $c does not end in exactly one newline" >&2; exit 1; }
  done
done
C9_4B=$(cat configs/qwen3-4b-tetra-e4-4e-v9.json); CB_4B=$(cat configs/qwen3-4b-tetra-e4-batch.json)
C9_8B=$(cat configs/qwen3-8b-tetra-e4-4e-v9.json); CB_8B=$(cat configs/qwen3-8b-tetra-e4-batch.json)
C9_14B=$(cat configs/qwen3-14b-tetra-e4-4e-v9.json); CB_14B=$(cat configs/qwen3-14b-tetra-e4-batch.json)
CFG_SHAS=$(shasum -a 256 configs/qwen3-{4b,8b,14b}-tetra-e4-{4e-v9,batch}.json | awk '{print $1}' | tr '\n' ' ')

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  while IFS=: read -r size path bytes sha fp; do
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
  echo "bucket copies verified by size; configs $CFG_SHAS; image $NOW"
fi

CMP_PY='import hashlib,json,sys
ref={int(i):(int(c),h) for i,c,h in (x.split(":") for x in sys.argv[2].split(","))}
ls=[json.loads(l) for l in open(sys.argv[1],encoding="utf-8").read().split("\n") if l][1:]
rows=[d for d in ls if not d.get("end")]
same=sum(hashlib.sha256(d["completion"].encode()).hexdigest()[:12]==ref.get(d["index"],(0,""))[1] for d in rows)
a=sum(int(d["correct"]) for d in rows); g=sum(c for c,_ in ref.values())
b=sum(1 for d in rows if d["correct"] and not ref[d["index"]][0]); c=sum(1 for d in rows if ref[d["index"]][0] and not d["correct"])
print("GSM8K pilot, %d problems, %s: %d correct against %d served; completions identical %d/%d; discordant %d up, %d down"%(len(rows),sys.argv[3],a,g,same,len(rows),b,c))'

PRE=$(printf 'O=%q\nOBJECTS=%q\nBATCHES=%q\nDATASET_REV=%q\nDOOR_FP_4B=%q\nREPLAY_FP=%q\nPILOT_REF=%q\nCMP_PY=%q\nC9_4B=%q\nCB_4B=%q\nC9_8B=%q\nCB_8B=%q\nC9_14B=%q\nCB_14B=%q\nCFG_SHAS=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$OBJECTS" "$BATCHES" "$DATASET_REV" "$DOOR_FP_4B" "$REPLAY_FP" "$PILOT_REF" "$CMP_PY" \
  "$C9_4B" "$CB_4B" "$C9_8B" "$CB_8B" "$C9_14B" "$CB_14B" "$CFG_SHAS")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
echo '== gate 1: oracle =='
timeout 5m oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle.txt" | tail -2
grep -q MATCH "$O/oracle.txt"
echo '== gate 2: the objects and the configs =='
while IFS=: read -r size path bytes sha fp; do
  test "$(stat -c %s "/out/$path")" = "$bytes"
  sha256sum "/out/$path" | tee -a "$O/files.sha256"
  test "$(sha256sum "/out/$path" | cut -d' ' -f1)" = "$sha"
done <<< "$OBJECTS"
printf '%s\n' "$C9_4B" > "$O/qwen3-4b-tetra-e4-4e-v9.json"; printf '%s\n' "$CB_4B" > "$O/qwen3-4b-tetra-e4-batch.json"
printf '%s\n' "$C9_8B" > "$O/qwen3-8b-tetra-e4-4e-v9.json"; printf '%s\n' "$CB_8B" > "$O/qwen3-8b-tetra-e4-batch.json"
printf '%s\n' "$C9_14B" > "$O/qwen3-14b-tetra-e4-4e-v9.json"; printf '%s\n' "$CB_14B" > "$O/qwen3-14b-tetra-e4-batch.json"
GOT=$(cd "$O" && sha256sum qwen3-{4b,8b,14b}-tetra-e4-{4e-v9,batch}.json | awk '{print $1}' | tr '\n' ' ')
test "$GOT" = "$CFG_SHAS"
echo "configs at their bytes: $GOT"

while IFS=: read -r size path bytes sha FP; do
  S=/out/$path; PB=$O/qwen3-$size-tetra-e4-batch.json
  case $size in 4b) T=1;; 8b) T=2;; 14b) T=3;; esac
  echo "== $size (a): the projections alone =="
  date -u
  LLVQ_CONFIG="$PB" timeout $((6 * T))m batchrun "$S" kernels 1,8,32,128 gemm 10 > "$O/kernels-$size.txt" 2>&1 || { tail -20 "$O/kernels-$size.txt"; exit 1; }
  grep -A6 '^projections alone' "$O/kernels-$size.txt"

  echo "== $size (b): batchrun check gemm =="
  date -u
  LLVQ_CONFIG="$PB" timeout $((6 * T))m batchrun "$S" check gemm > "$O/check-$size.txt" 2>&1 || { tail -20 "$O/check-$size.txt"; exit 1; }
  grep -E '^batch|^tv_bg|^attn_slot|^check: ' "$O/check-$size.txt"

  ARMS=gemm
  echo "== $size (c): the batched door, $ARMS, B = $BATCHES =="
  date -u
  LLVQ_CONFIG="$PB" LLVQ_BATCH_REF="$FP" timeout $((12 * T))m batchrun "$S" door "$BATCHES" "$ARMS" 256 3 > "$O/door-batch-$size.txt" 2>&1 \
    || { tail -20 "$O/door-batch-$size.txt"; exit 1; }
  grep -A20 '^batched door' "$O/door-batch-$size.txt"

  echo "== $size (d): decode-path perplexity, gemm against rows =="
  date -u
  LLVQ_CONFIG="$PB" timeout $((10 * T))m batchrun "$S" ppl 2048 16 gemm > "$O/ppl-2048x16-$size.txt" 2>&1 || { tail -20 "$O/ppl-2048x16-$size.txt"; exit 1; }
  grep -E '^batched ppl|^ppl|^T8' "$O/ppl-2048x16-$size.txt"
  if [ "$size" = 4b ]; then
    LLVQ_CONFIG="$PB" timeout 10m batchrun "$S" ppl 256 128 gemm > "$O/ppl-256x128-$size.txt" 2>&1 || { tail -20 "$O/ppl-256x128-$size.txt"; exit 1; }
    grep -E '^batched ppl|^ppl|^T8' "$O/ppl-256x128-$size.txt"

    echo "== 4b (e): the GSM8K pilot, one batch of 50, gemm =="
    date -u
    LLVQ_CONFIG="$PB" LLVQ_DATASET_REV="$DATASET_REV" LLVQ_GSM8K_DUMP="$O/gsm8k-b50-gemm.jsonl" \
      timeout 10m batchrun "$S" gsm8k 50 gemm 50 > "$O/gsm8k-b50-gemm.txt" 2>&1 || { tail -20 "$O/gsm8k-b50-gemm.txt"; exit 1; }
    tail -1 "$O/gsm8k-b50-gemm.txt"
    tail -1 "$O/gsm8k-b50-gemm.jsonl" | grep -qF "\"fingerprint\":\"$REPLAY_FP\""
    python3 -c "$CMP_PY" "$O/gsm8k-b50-gemm.jsonl" "$PILOT_REF" gemm | tee -a "$O/compare.txt"
  fi
done <<< "$OBJECTS"
date -u
ls -la "$O"
JOB

if [ "$DRY" = 1 ]; then
  printf '%s\n' "set -euo pipefail
$PRE
$BODY" | bash -n && echo "DRY_RUN: job script parses; nothing launched; pilot reference of $(echo "$PILOT_REF" | tr ',' '\n' | wc -l | tr -d ' ') rows; configs $CFG_SHAS" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "batch-l40s-v2${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
