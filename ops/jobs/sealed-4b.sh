#!/usr/bin/env bash
# The sealed 4B through the served kernel. Preregistered:
#   proofs/preregistration-sealed-4b-2026-09-23.md, stamped (ots stamp) before this.
#
# One l40sx1 job, cheapest stages first: oracle and sha256, the prefill gate
# through LLVQ_CONFIG, arm S (the sealed file, q4 embedding) and arm R (the
# 61.11 file, q8) at 256 tokens against their dense arms, the dense census of
# the sealed file, then MMLU through the kernel on 2,280 questions.
#
# Needs an image carrying EmbedMode::Q4 (emb_q4.cu); IMAGE_SHA names it and the
# launcher refuses a Space that moved.
#
# Cost, estimated: ~2 h 10 on l40sx1, ~$3.90; timeout 165m, ceiling $4.95.
#
#   bash ops/jobs/sealed-4b.sh upload          copy the sealed file into the bucket
#   DRY_RUN=1 bash ops/jobs/sealed-4b.sh       prints the job script, parses it, launches nothing
#   IMAGE_SHA=<sha> bash ops/jobs/sealed-4b.sh launches
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-sealed-4b-2026-09-23.md
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
OBJ_DIR=sealed-4b-2026-09-23                         # the sealed file, written from the Mac
LOCAL=${LOCAL:-$HOME/q4b-sealed-2026-09-23/qwen3-4b-sealed.bin}
SHA=886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8
BYTES=1418224685
O=/out/sealed-4b-served-2026-09-23                   # the outputs
F=/out/$OBJ_DIR/qwen3-4b-sealed.bin
R=/out/dclm-ft-2026-09-19/qwen3-4b-dclm-ft.bin       # the 61.11 object, arm R
R_SHA=f8c1c903b753fe3427f395f53b79e1e4ab59d212a6bc1c4c5ceedf8aea971300
R_BYTES=1794564765
C=/usr/local/share/llvq/configs/qwen3-4b-tetra-e4.json
FP=a74a6d6213602979
FP40=65dcd53655e8bfa5
DRY=${DRY_RUN:-0}

local_ok() {
  [ "$(stat -f %z "$LOCAL")" = "$BYTES" ] || { echo "refused: $LOCAL is not $BYTES B" >&2; exit 1; }
  [ "$(shasum -a 256 "$LOCAL" | cut -d' ' -f1)" = "$SHA" ] || { echo "refused: $LOCAL sha256 differs" >&2; exit 1; }
}

if [ "${1:-}" = upload ]; then
  local_ok
  hf buckets cp "$LOCAL" "hf://buckets/$BUCKET/$OBJ_DIR/qwen3-4b-sealed.bin"
  hf buckets ls "hf://buckets/$BUCKET/$OBJ_DIR/"
  exit 0
fi

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  local_ok
  REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/$OBJ_DIR/" 2>/dev/null | awk '$NF ~ /qwen3-4b-sealed\.bin$/ {print $1}' || true)
  [ "$REMOTE" = "$BYTES" ] || { echo "refused: bucket copy is ${REMOTE:-absent} B, local $BYTES B" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  [ ${#IMAGE_SHA} -eq 40 ] || { echo "refused: IMAGE_SHA=<the Space sha of the image carrying q4> is required" >&2; exit 1; }
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "sealed object: $BYTES B, sha256 $SHA; image $NOW"
fi

PRE=$(printf 'O=%q\nF=%q\nSHA=%q\nBYTES=%q\nR=%q\nR_SHA=%q\nR_BYTES=%q\nC=%q\nFP=%q\nFP40=%q' \
  "$O" "$F" "$SHA" "$BYTES" "$R" "$R_SHA" "$R_BYTES" "$C" "$FP" "$FP40")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt"
echo '== oracle (hard rule 10) =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle-cuda.txt" | tail -2
grep -q MATCH "$O/oracle-cuda.txt"
echo '== both files on the mount: bytes and sha256 against the Mac =='
test "$(stat -c %s "$F")" = "$BYTES"
test "$(stat -c %s "$R")" = "$R_BYTES"
sha256sum "$F" "$R" | tee "$O/files.sha256"
test "$(sed -n 1p "$O/files.sha256" | cut -d' ' -f1)" = "$SHA"
test "$(sed -n 2p "$O/files.sha256" | cut -d' ' -f1)" = "$R_SHA"
sha256sum "$C" | tee -a "$O/files.sha256"
cat "$C"
grep -qF '"embed": "q4"' "$C"

echo '== prefill gate, 203 tokens, through the served config ==' ; date -u
LLVQ_CONFIG="$C" LLVQ_PREFILL_TOKENS=203 fusedrun "$F" 2>&1 | tee "$O/prefill-203.txt" | tail -14
grep -qF 'tile 64 (served: measured optimum for sm_89)' "$O/prefill-203.txt"
grep -qF '(0 groups + 168 lone + 84 int4)' "$O/prefill-203.txt"
grep -qF 'embedding: q4 g64' "$O/prefill-203.txt"

FLAGS='LLVQ_FUSED_LAYOUT=tetra48 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_KV=f16'
echo '== ARM S: the sealed file, served flags, q4 embedding, 256 tokens against its dense arm ==' ; date -u
env $FLAGS LLVQ_EMBED=q4 fusedrun "$F" 256 2>&1 | tee "$O/fused-S-256.txt" | tail -30
grep -qF 'embedding: q4 g64' "$O/fused-S-256.txt"
echo '== ARM R: the 61.11 file, served flags, q8 embedding, 256 tokens against its dense arm ==' ; date -u
env $FLAGS LLVQ_EMBED=q8 fusedrun "$R" 256 2>&1 | tee "$O/fused-R-256.txt" | tail -30
grep -qF 'embedding: q8 g64' "$O/fused-R-256.txt"

echo '== MMLU dense, FULL split, the sealed file ==' ; date -u
LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP="$O/mmlu-4b-sealed-FULL.csv" \
  mmlu "$F" cuda 2>&1 | tee "$O/mmlu-dense.txt" | tail -8
D="$O/mmlu-4b-sealed-FULL.csv"
grep -qxF '# dtype=f16' "$D"
grep -qxF '# limit=census' "$D"
grep -qxF '# alloc=flat, 100..1534 per subject, 14042 questions' "$D"
grep -qxF '# config=none' "$D"
grep -qxF '# arithmetic=dense reconstruction' "$D"
grep -qxF '# kv=f16' "$D"
test "$(tail -n 1 "$D")" = "# end fingerprint=$FP questions=14042"

echo '== MMLU through the served kernel, 2,280 questions ==' ; date -u
LLVQ_CONFIG="$C" LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP="$O/mmlu-4b-sealed-served-l40.csv" \
  mmlu "$F" cuda 40 2>&1 | tee "$O/mmlu-served.txt" | tail -8
K="$O/mmlu-4b-sealed-served-l40.csv"
grep -qxF '# arithmetic=served kernel' "$K"
grep -qxF "# config=$C" "$K"
test "$(tail -n 1 "$K")" = "# end fingerprint=$FP40 questions=2280"
date -u
echo '== outputs ==' ; ls -la "$O"
JOB

if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; nothing launched" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout 165m \
  --bucket "$BUCKET" --out-mount /out \
  --name sealed-4b-served \
  "$PRE" "$BODY"
