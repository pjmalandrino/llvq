#!/usr/bin/env bash
# Batching on the sealed 4B, one job: the starting point and the gate.
#   (1) the served door at batch 1 (fusedrun under the v4 config), its token fingerprint the reference;
#   (2) batchrun check: the tensor-core projections against the multi-row path on the real weights;
#   (3) batchrun door at B = 1, 4, 8, 16, 32, arm `rows` (the existing multi-row path, the baseline) and arm `mma`
#       (decode once a block, tensor-core MMA), 256 tokens, every row's tokens against the batch-1 fingerprint;
#   (4) batchrun ppl: 16 WikiText-2 windows of 2,048, every position a decode row, mma against rows (T8 band);
#   (5) the 50 GSM8K pilot problems, batched 16 at a time, under rows then mma, against the served pilot dump.
# Preregistered: proofs/preregistration-batch-tetra-2026-11-07.md, stamped before this.
#
#   DRY_RUN=1 bash ops/jobs/batch-tetra.sh                parses the job script, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/batch-tetra.sh     launches
#
# Every LLVQ_* variable is set on its own command line, never exported. Cost, estimated (l40sx1 $1.80/h): about
# 22 min, $0.66. Timeout 45m: $1.35 at worst.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-batch-tetra-2026-11-07.md
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/batch-tetra-2026-11-07${RETRY:+-r$RETRY}
TIMEOUT=45m
DRY=${DRY_RUN:-0}
DATASET_REV=740312add88f781978c0658806c59bc2815b9866
SEALED="sealed-4b-2026-09-23/qwen3-4b-sealed.bin:1418224685"
SEALED_SHA=886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8
CV4_LOCAL=configs/qwen3-4b-tetra-e4-4e-v4.json
DOOR_FP=61d0654532d35120
REPLAY_FP=94c7b98876288901
PILOT=docs/mesures/served-v4-2026-11-05-brut/gsm8k-4b-v4-n50.jsonl
PILOT_SHA=a7203fa6a7f19faf91b46a1907ebe08989c7354582f2c10e306a40e557cfcab8
REF_PY='import hashlib,json,sys
ls=[json.loads(l) for l in open(sys.argv[1],encoding="utf-8").read().split("\n") if l][1:]
print(",".join("%d:%d:%s"%(d["index"],int(d["correct"]),hashlib.sha256(d["completion"].encode()).hexdigest()[:12]) for d in ls if not d.get("end")))'

CV4=$(cat "$CV4_LOCAL"); CV4_SHA=$(shasum -a 256 < "$CV4_LOCAL" | cut -d' ' -f1)
[ "$(shasum -a 256 < "$PILOT" | cut -d' ' -f1)" = "$PILOT_SHA" ] || { echo "refused: $PILOT is not $PILOT_SHA" >&2; exit 1; }
PILOT_REF=$(python3 -c "$REF_PY" "$PILOT")

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  path=${SEALED%%:*}; bytes=${SEALED##*:}
  REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/${path%/*}/" 2>/dev/null \
    | awk -v n="/${path##*/}" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
  [ "$REMOTE" = "$bytes" ] || { echo "refused: $path is ${REMOTE:-absent} B in the bucket, expected $bytes" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of the rebuild that carries this branch>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "bucket copy verified by size; config $CV4_SHA; image $NOW"
fi

CMP_PY='import hashlib,json,sys
ref={int(i):(int(c),h) for i,c,h in (x.split(":") for x in sys.argv[2].split(","))}
ls=[json.loads(l) for l in open(sys.argv[1],encoding="utf-8").read().split("\n") if l][1:]
rows=[d for d in ls if not d.get("end")]
same=sum(hashlib.sha256(d["completion"].encode()).hexdigest()[:12]==ref.get(d["index"],(0,""))[1] for d in rows)
a=sum(int(d["correct"]) for d in rows); g=sum(c for c,_ in ref.values())
b=sum(1 for d in rows if d["correct"] and not ref[d["index"]][0]); c=sum(1 for d in rows if ref[d["index"]][0] and not d["correct"])
print("GSM8K pilot, %d problems, %s: %d correct against %d served; completions identical %d/%d; discordant %d up, %d down"%(len(rows),sys.argv[3],a,g,same,len(rows),b,c))'

PRE=$(printf 'O=%q\nS=%q\nSB=%q\nSSHA=%q\nCV4=%q\nCV4_SHA=%q\nDATASET_REV=%q\nDOOR_FP=%q\nREPLAY_FP=%q\nPILOT_REF=%q\nCMP_PY=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "/out/${SEALED%%:*}" "${SEALED##*:}" "$SEALED_SHA" "$CV4" "$CV4_SHA" "$DATASET_REV" "$DOOR_FP" "$REPLAY_FP" \
  "$PILOT_REF" "$CMP_PY")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
echo '== gate 1: oracle =='
timeout 5m oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle.txt" | tail -2
grep -q MATCH "$O/oracle.txt"
echo '== gate 2: the object and the config =='
test "$(stat -c %s "$S")" = "$SB"
sha256sum "$S" | tee "$O/files.sha256"
test "$(cut -d' ' -f1 "$O/files.sha256")" = "$SSHA"
P4=$O/qwen3-4b-tetra-e4-4e-v4.json
printf '%s\n' "$CV4" > "$P4"
test "$(sha256sum "$P4" | cut -d' ' -f1)" = "$CV4_SHA"

echo '== PART 1: the served door at batch 1 =='
date -u
LLVQ_CONFIG="$P4" timeout 8m fusedrun "$S" 256 > "$O/door-b1.txt" 2>&1
grep -E '^served door|^served :|^  (4E|eager)|^  token ids' "$O/door-b1.txt" | cut -c1-160
grep -qE "^  token ids: $DOOR_FP " "$O/door-b1.txt"
echo "gate 3: the served door decoded $DOOR_FP"

echo '== PART 2: tensor-core projections against the multi-row path =='
date -u
LLVQ_CONFIG="$P4" timeout 6m batchrun "$S" check > "$O/check.txt" 2>&1 || { tail -20 "$O/check.txt"; exit 1; }
grep -E '^batch|^tv_|^attn_batch|^check' "$O/check.txt"

echo '== PART 3: the batched door, rows and mma =='
date -u
LLVQ_CONFIG="$P4" LLVQ_BATCH_REF="$DOOR_FP" timeout 14m batchrun "$S" door 1,4,8,16,32 rows,mma 256 3 > "$O/door-batch.txt" 2>&1 \
  || { tail -20 "$O/door-batch.txt"; exit 1; }
grep -A20 '^batched door' "$O/door-batch.txt"
grep -E 'warm-up' "$O/door-batch.txt" | cut -c1-160

echo '== PART 4: decode-path perplexity, mma against rows =='
date -u
LLVQ_CONFIG="$P4" timeout 8m batchrun "$S" ppl 2048 16 > "$O/ppl.txt" 2>&1 || { tail -20 "$O/ppl.txt"; exit 1; }
grep -E '^batched ppl|^ppl|^T8' "$O/ppl.txt"

echo '== PART 5: the GSM8K pilot, batched 16 at a time =='
for ARM in rows mma; do
  date -u
  LLVQ_CONFIG="$P4" LLVQ_DATASET_REV="$DATASET_REV" LLVQ_GSM8K_DUMP="$O/gsm8k-b16-$ARM.jsonl" \
    timeout 8m batchrun "$S" gsm8k 16 "$ARM" 50 > "$O/gsm8k-b16-$ARM.txt" 2>&1 || { tail -20 "$O/gsm8k-b16-$ARM.txt"; exit 1; }
  tail -1 "$O/gsm8k-b16-$ARM.txt"
  tail -1 "$O/gsm8k-b16-$ARM.jsonl" | grep -qF "\"fingerprint\":\"$REPLAY_FP\""
  python3 -c "$CMP_PY" "$O/gsm8k-b16-$ARM.jsonl" "$PILOT_REF" "$ARM" | tee -a "$O/compare.txt"
done
date -u
ls -la "$O"
JOB

if [ "$DRY" = 1 ]; then
  printf '%s\n' "set -euo pipefail
$PRE
$BODY" | bash -n && echo "DRY_RUN: job script parses; nothing launched; pilot reference of $(echo "$PILOT_REF" | tr ',' '\n' | wc -l | tr -d ' ') rows" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "batch-tetra${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
