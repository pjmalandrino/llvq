#!/usr/bin/env bash
# The served speed of a sealed paper-2 object (runs 1-3 of the paper table).
# Preregistered: proofs/preregistration-paper-table-2026-09-25.md, stamped before this.
#
#   SIZE=4b|8b|14b DRY_RUN=1 bash ops/jobs/paper-served.sh   prints and parses the job script
#   SIZE=4b|8b|14b IMAGE_SHA=<space sha> bash ops/jobs/paper-served.sh
#
# ops/jobs/served-14b.sh without the door, at q4: the prefill gate through the
# served config, then fusedrun 256 tokens against the dense arm of the same process at
# the served flags spelled out (EMBED=q4), then at EMBED=f16, the same-head arm of hard
# rule 4. The flags are spelled out because LLVQ_CONFIG puts fusedrun on a one-arm path
# with no dense reference. The config is written by the job from this command line and
# checked by sha256, so the dumps and the provenance line name the object's own file.
#
# IMAGE_SHA is the Space sha of the rebuild that carries EmbedMode::Q4; the launcher
# refuses if the Space is elsewhere, so the three sizes run on one image.
#
# Cost, estimated (l40sx1 $1.80/h): 4B ~10 min $0.30, 8B ~13 min $0.40, 14B ~30 min
# $0.90. Ceilings 30m / 30m / 45m.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

SIZE=${SIZE:?SIZE=4b, 8b or 14b}
case "$SIZE" in
  4b)  OBJ_DIR=sealed-4b-2026-09-23;   NAME=qwen3-4b-sealed.bin
       LOCAL=$HOME/q4b-sealed-2026-09-23/$NAME; TIMEOUT=30m
       TABLES='1 table (model.embed_tokens.weight, lm_head tied on it)' ;;
  8b)  OBJ_DIR=sealed-8b27-2026-09-24;  NAME=qwen3-8b-sealed-B.bin
       LOCAL=$HOME/q8b-14b-sealed-2026-09-23/$NAME; TIMEOUT=30m
       TABLES='2 tables (model.embed_tokens.weight + lm_head.weight)' ;;
  14b) OBJ_DIR=sealed-14b-2026-09-23;  NAME=qwen3-14b-sealed.bin
       LOCAL=$HOME/q8b-14b-sealed-2026-09-23/$NAME; TIMEOUT=45m
       TABLES='2 tables (model.embed_tokens.weight + lm_head.weight)' ;;
  *) echo "refused: SIZE=$SIZE, expected 4b, 8b or 14b" >&2; exit 1 ;;
esac
PREREG=proofs/preregistration-paper-table-2026-09-25.md
BUCKET=Pier-Jean/jobs-artifacts
CONF_LOCAL=configs/qwen3-$SIZE-tetra-e4.json
# RETRY=2 names a relaunch after a failed attempt left its directory behind.
O=/out/paper-served-$SIZE-2026-09-25${RETRY:+-r$RETRY}
F=/out/$OBJ_DIR/$NAME
C=$O/qwen3-$SIZE-tetra-e4.json
DRY=${DRY_RUN:-0}

CONF=$(cat "$CONF_LOCAL")
CONF_SHA=$(printf '%s\n' "$CONF" | shasum -a 256 | cut -d' ' -f1)
[ "$CONF_SHA" = "$(shasum -a 256 < "$CONF_LOCAL" | cut -d' ' -f1)" ] \
  || { echo "refused: $CONF_LOCAL does not end in exactly one newline" >&2; exit 1; }
python3 -c 'import json,sys; d=json.load(open(sys.argv[1]));
assert (d["layout"],d["embed"],d["rot_share"],d["fuse"],d["kv"])==("tetra48","q4","1","0","f16"), d' "$CONF_LOCAL"

if [ "$DRY" = 1 ]; then
  SHA=0000000000000000000000000000000000000000000000000000000000000000
  BYTES=0
else
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  test -f "$LOCAL" || { echo "refused: $LOCAL missing" >&2; exit 1; }
  SHA=$(shasum -a 256 "$LOCAL" | cut -d' ' -f1)
  BYTES=$(stat -f %z "$LOCAL")
  REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/$OBJ_DIR/" 2>/dev/null | awk -v n="/$NAME" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
  [ "$REMOTE" = "$BYTES" ] || { echo "refused: bucket copy is ${REMOTE:-absent} B, local $BYTES B" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of the q4 rebuild>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "$SIZE sealed: $BYTES B, sha256 $SHA; config $CONF_LOCAL sha256 $CONF_SHA; image $NOW"
fi

PRE=$(printf 'O=%q\nF=%q\nC=%q\nSHA=%q\nBYTES=%q\nCONF=%q\nCONF_SHA=%q\nTABLES=%q' \
  "$O" "$F" "$C" "$SHA" "$BYTES" "$CONF" "$CONF_SHA" "$TABLES")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt"
echo '== oracle (hard rule 10) =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle-cuda.txt" | tail -2
grep -q MATCH "$O/oracle-cuda.txt"
echo '== the object on the mount: bytes and sha256 against the Mac =='
test "$(stat -c %s "$F")" = "$BYTES"
sha256sum "$F" | tee "$O/files.sha256"
test "$(cut -d' ' -f1 "$O/files.sha256")" = "$SHA"
echo '== the served config, written from the launch command =='
printf '%s\n' "$CONF" > "$C"
sha256sum "$C" | tee -a "$O/files.sha256"
test "$(sha256sum "$C" | cut -d' ' -f1)" = "$CONF_SHA"
cat "$C"
echo '== prefill gate, 203 tokens, through the served config =='
date -u
LLVQ_CONFIG="$C" LLVQ_PREFILL_TOKENS=203 fusedrun "$F" 2>&1 | tee "$O/prefill-203.txt" | tail -14
date -u
grep -qF 'tile 64 (served: measured optimum for sm_89)' "$O/prefill-203.txt"
grep -F 'embedding: q4 g64' "$O/prefill-203.txt" | grep -qF "$TABLES"
grep -E 'lone|int4' "$O/prefill-203.txt" | head -3 || true
FLAGS='LLVQ_FUSED_LAYOUT=tetra48 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_KV=f16'
echo '== ARM q4: the served flags spelled out, 256 tokens against the dense arm, same process =='
date -u
env $FLAGS LLVQ_EMBED=q4 fusedrun "$F" 256 2>&1 | tee "$O/fused-q4-256.txt" | tail -30
date -u
grep -qF 'tile 64 (served: measured optimum for sm_89)' "$O/fused-q4-256.txt"
grep -F 'embedding: q4 g64 (LLVQ_EMBED)' "$O/fused-q4-256.txt" | grep -qF "$TABLES"
# Identity is READ, not gated: the prereg says where a divergence is a defect.
grep -E 'tokens identical to the dense arm|divergence at token' "$O/fused-q4-256.txt" || true
echo '== ARM f16: the same-head arm, LLVQ_EMBED=f16 (hard rule 4), 256 tokens =='
date -u
env $FLAGS LLVQ_EMBED=f16 fusedrun "$F" 256 2>&1 | tee "$O/fused-f16-256.txt" | tail -30
date -u
grep -E 'tokens identical to the dense arm|divergence at token' "$O/fused-f16-256.txt" || true
echo '== raw kept ==' ; ls -la "$O" ; wc -l "$O"/*.txt
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
  --flavor l40sx1 --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "paper-served-$SIZE${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
