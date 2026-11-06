#!/usr/bin/env bash
# The quality control of the tolerance gate (decision T8): WikiText-2 perplexity of the grouped attention
# against the published one, on the same 12 windows of 4,096, by the prefill path and by the decode path,
# through the served kernel. Preregistered: proofs/preregistration-gqa-ppl-2026-11-01.md, stamped before
# this.
#
#   DRY_RUN=1 bash ops/jobs/gqa-ppl.sh                 prints and parses the job script
#   IMAGE_SHA=<space sha> bash ops/jobs/gqa-ppl.sh     launches, after the operator's go
#
# The sealed 4B only. Three steps in one job: oracle; the published arm replayed (paper-table run 1,
# flags spelled out, against the dense arm, decoded line checked by sha256 against that run), which
# checks that the default path did not move; LLVQ_GQA_PPL=4096,12 on the same flags. The config is
# written by the job from this command line and checked by sha256.
#
# Cost, estimated (l40sx1 $1.80/h): about 35 min, $1.05. Ceiling 60m, $1.80.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

OBJ_DIR=sealed-4b-2026-09-23
NAME=qwen3-4b-sealed.bin
LOCAL=$HOME/q4b-sealed-2026-09-23/$NAME
TIMEOUT=60m
TABLES='1 table (model.embed_tokens.weight, lm_head tied on it)'
PREREG=proofs/preregistration-gqa-ppl-2026-11-01.md
BUCKET=Pier-Jean/jobs-artifacts
CONF_LOCAL=configs/qwen3-4b-tetra-e4.json
# The decoded line of run 1 of the paper table, `grep -F '  fused F0:  '` of
# docs/mesures/paper-table-2026-09-25-brut/paper-served-4b-2026-09-25-r2/fused-q4-256.txt.
REF_LINE_SHA=8784ffc423bfbeeaeb9a92abd426573cc45f9a083ba56e61e52d9ff40e3b018f
PPL_SPEC=4096,12
# The scored stream of ppl-scelles-2026-10-04: wikitext-2 test, 12 windows of 4,096, Qwen3 tokenizer.
TOKENS=3f1baca9033bf251
# RETRY=2 names a relaunch after a failed attempt left its directory behind.
O=/out/gqa-ppl-4b-2026-11-01${RETRY:+-r$RETRY}
F=/out/$OBJ_DIR/$NAME
C=$O/qwen3-4b-tetra-e4.json
DRY=${DRY_RUN:-0}

CONF=$(cat "$CONF_LOCAL")
CONF_SHA=$(printf '%s\n' "$CONF" | shasum -a 256 | cut -d' ' -f1)
[ "$CONF_SHA" = "$(shasum -a 256 < "$CONF_LOCAL" | cut -d' ' -f1)" ] \
  || { echo "refused: $CONF_LOCAL does not end in exactly one newline" >&2; exit 1; }
python3 -c 'import json,sys; d=json.load(open(sys.argv[1]));
assert (d["layout"],d["embed"],d["rot_share"],d["fuse"],d["kv"])==("tetra48","q4","1","0","f16"), d' "$CONF_LOCAL"
REF=docs/mesures/paper-table-2026-09-25-brut/paper-served-4b-2026-09-25-r2/fused-q4-256.txt
[ "$(grep -F '  fused F0:  ' "$REF" | shasum -a 256 | cut -d' ' -f1)" = "$REF_LINE_SHA" ] \
  || { echo "refused: the reference line of $REF does not hash to $REF_LINE_SHA" >&2; exit 1; }

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
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of the rebuild that carries the attention forms>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "4b sealed: $BYTES B, sha256 $SHA; config $CONF_LOCAL sha256 $CONF_SHA; image $NOW"
fi

PRE=$(printf 'O=%q\nF=%q\nC=%q\nSHA=%q\nBYTES=%q\nCONF=%q\nCONF_SHA=%q\nTABLES=%q\nREF_LINE_SHA=%q\nPPL_SPEC=%q\nTOKENS=%q' \
  "$O" "$F" "$C" "$SHA" "$BYTES" "$CONF" "$CONF_SHA" "$TABLES" "$REF_LINE_SHA" "$PPL_SPEC" "$TOKENS")

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
PARITY='int4: tv_q4_hh equals tv_q4_h bit for bit'
TILE='tile 64 (served: measured optimum for sm_89)'
FLAGS='LLVQ_FUSED_LAYOUT=tetra48 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_KV=f16 LLVQ_EMBED=q4'
echo '== STEP 2: the published arm replayed, 256 tokens against the dense arm, same process =='
date -u
env $FLAGS fusedrun "$F" 256 2>&1 | tee "$O/published-q4-256.txt" | tail -30
date -u
grep -qF "$TILE" "$O/published-q4-256.txt"
grep -qF "$PARITY" "$O/published-q4-256.txt"
grep -F 'embedding: q4 g64 (LLVQ_EMBED)' "$O/published-q4-256.txt" | grep -qF "$TABLES"
grep -qF 'fused F0: 256 tokens identical to the dense arm' "$O/published-q4-256.txt"
test "$(grep -cF '  fused F0:  ' "$O/published-q4-256.txt")" = 1
test "$(grep -F '  fused F0:  ' "$O/published-q4-256.txt" | sha256sum | cut -d' ' -f1)" = "$REF_LINE_SHA"
echo 'gate 4: decoded line identical to paper-table run 1'
echo "== STEP 3: LLVQ_GQA_PPL=$PPL_SPEC, two arms, prefill and decode paths =="
date -u
env $FLAGS LLVQ_GQA_PPL="$PPL_SPEC" fusedrun "$F" 2>&1 | tee "$O/gqa-ppl.txt" | tail -24
date -u
grep -qF "$TILE" "$O/gqa-ppl.txt"
grep -qF "$PARITY" "$O/gqa-ppl.txt"
grep -qF 'cudarc event tracking: off (one stream, ordered by issue)' "$O/gqa-ppl.txt"
grep -qF "(tokens $TOKENS)" "$O/gqa-ppl.txt"
test "$(grep -cE '^  window +[0-9]+/12  prefill nll ' "$O/gqa-ppl.txt")" = 12
test "$(grep -cE '^  (prefill|decode) +[0-9.]+ +[0-9.]+ +[-+][0-9.]+ % \[' "$O/gqa-ppl.txt")" = 2
# The harness: both paths score the same tokens, so on the published arm they agree to rounding.
awk '/^  harness: decode over prefill/ {r = $NF; ok = (r >= 0.99 && r <= 1.01)} END {exit !ok}' "$O/gqa-ppl.txt"
echo 'gate: decode and prefill agree on the published arm'
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
  --name "gqa-ppl-4b${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
