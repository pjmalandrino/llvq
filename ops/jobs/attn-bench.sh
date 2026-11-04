#!/usr/bin/env bash
# The split-KV decode-attention kernel on one L40S: alone (identity against the CPU replay, read rate,
# verify cost, registers), then in the served 4B (the served door's context sweep, kernel arm then
# grouped arm). Preregistered: proofs/preregistration-attn-bench-2026-11-04.md, stamped before this.
#
#   DRY_RUN=1 bash ops/jobs/attn-bench.sh             prints and parses the job script
#   IMAGE_SHA=<space sha> bash ops/jobs/attn-bench.sh  launches, after the operator's go
#
# Steps: oracle (hard rule 10); `attnbench 2048,4096,8192`; then LLVQ_CONTEXTS=5,2048,4096,8192 beside
# two configs that differ in the attention field only, the kernel one written by the job from this
# command line (it is not a file of configs/ and changes no served default).
#
# Cost, estimated (l40sx1 $1.80/h): about 21 min, $0.63. Ceiling 33m, $0.99, under the go's $1.00.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

OBJ_DIR=sealed-4b-2026-09-23
NAME=qwen3-4b-sealed.bin
LOCAL=$HOME/q4b-sealed-2026-09-23/$NAME
TIMEOUT=33m
PREREG=proofs/preregistration-attn-bench-2026-11-04.md
BUCKET=Pier-Jean/jobs-artifacts
LENGTHS=2048,4096,8192
CONTEXTS=5,2048,4096,8192
CONF_GRP_LOCAL=configs/qwen3-4b-tetra-e4-grouped.json
# RETRY=2 names a relaunch after a failed attempt left its directory behind.
O=/out/attn-bench-2026-11-04${RETRY:+-r$RETRY}
F=/out/$OBJ_DIR/$NAME
CG=$O/qwen3-4b-tetra-e4-grouped.json
CK=$O/qwen3-4b-tetra-e4-kernel.json
DRY=${DRY_RUN:-0}

CONF_GRP=$(cat "$CONF_GRP_LOCAL")
CONF_GRP_SHA=$(printf '%s\n' "$CONF_GRP" | shasum -a 256 | cut -d' ' -f1)
[ "$CONF_GRP_SHA" = "$(shasum -a 256 < "$CONF_GRP_LOCAL" | cut -d' ' -f1)" ] \
  || { echo "refused: $CONF_GRP_LOCAL does not end in exactly one newline" >&2; exit 1; }
# The kernel config: the grouped one with the attention field and the note changed, nothing else.
CONF_KER=$(python3 -c 'import json,sys
g=json.load(open(sys.argv[1]))
assert (g["layout"],g["embed"],g["rot_share"],g["fuse"],g["kv"],g["attention"])==("tetra48","q4","1","0","f16","grouped"), g
g["attention"]="kernel"
g["note"]="MEASUREMENT ONLY, written by ops/jobs/attn-bench.sh: the qwen3-4b-tetra-e4 object with decode rows sent to the split-KV attention kernel. Not licensed: the kernel is not bit for bit with the grouped form, and its T8 decision is the operator'"'"'s."
print(json.dumps(g, indent=2))' "$CONF_GRP_LOCAL")
CONF_KER_SHA=$(printf '%s\n' "$CONF_KER" | shasum -a 256 | cut -d' ' -f1)

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
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of the rebuild that carries attnbench>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "attn_dec.cu sha256 $(shasum -a 256 < llvq-cuda/kernels/attn_dec.cu | cut -d' ' -f1) (file); 4b sealed: $BYTES B, sha256 $SHA; configs $CONF_KER_SHA (kernel) / $CONF_GRP_SHA (grouped); image $NOW"
fi

PRE=$(printf 'O=%q\nF=%q\nCG=%q\nCK=%q\nSHA=%q\nBYTES=%q\nCONF_GRP=%q\nCONF_KER=%q\nCONF_GRP_SHA=%q\nCONF_KER_SHA=%q\nLENGTHS=%q\nCONTEXTS=%q' \
  "$O" "$F" "$CG" "$CK" "$SHA" "$BYTES" "$CONF_GRP" "$CONF_KER" "$CONF_GRP_SHA" "$CONF_KER_SHA" "$LENGTHS" "$CONTEXTS")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env (gate 4)' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
echo '== STEP 1: oracle (hard rule 10) =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle-cuda.txt" | tail -2
grep -q MATCH "$O/oracle-cuda.txt"
echo "== STEP 2: attnbench $LENGTHS =="
date -u
attnbench "$LENGTHS" 2>&1 | tee "$O/attnbench.txt"
date -u
grep -qF 'attn_dec.cu: embedded, sha256 ' "$O/attnbench.txt"
test "$(grep -cE '^report attn_dec_g[45]_u(64|32): ' "$O/attnbench.txt")" = 4
test "$(grep -cE '^identity ' "$O/attnbench.txt")" -ge 4
test "$(grep -cE '^attn_dec_g[45]_u(64|32) +[0-9]+ +[0-9]+ ' "$O/attnbench.txt")" = 48
echo '== the object on the mount: bytes and sha256 against the Mac =='
test "$(stat -c %s "$F")" = "$BYTES"
sha256sum "$F" | tee "$O/files.sha256"
test "$(cut -d' ' -f1 "$O/files.sha256")" = "$SHA"
echo '== the two configs, written from the launch command =='
printf '%s\n' "$CONF_KER" > "$CK"
printf '%s\n' "$CONF_GRP" > "$CG"
sha256sum "$CK" "$CG" | tee -a "$O/files.sha256"
test "$(sha256sum "$CK" | cut -d' ' -f1)" = "$CONF_KER_SHA"
test "$(sha256sum "$CG" | cut -d' ' -f1)" = "$CONF_GRP_SHA"
cat "$CK" "$CG"
PARITY='int4: tv_q4_hh equals tv_q4_h bit for bit'
TILE='tile 64 (served: measured optimum for sm_89)'
echo "== STEP 3: LLVQ_CONTEXTS=$CONTEXTS beside each config, 256 new tokens, kernel arm first =="
for arm in kernel grouped; do
  case $arm in
    kernel)  C=$CK; WANT='attention kernel on grow(256)' ;;
    grouped) C=$CG; WANT='attention grouped on grow(256)' ;;
  esac
  date -u
  env LLVQ_CONFIG="$C" LLVQ_CONTEXTS="$CONTEXTS" fusedrun "$F" 256 2>&1 \
    | tee "$O/sweep-$arm.txt" | tail -16
  date -u
  grep -qF "$TILE" "$O/sweep-$arm.txt"
  grep -qF "$PARITY" "$O/sweep-$arm.txt"
  grep -E '^served config: ' "$O/sweep-$arm.txt" | grep -qF "$WANT"
  test "$(grep -cE '^ +[0-9]+ +[0-9.]+ \[' "$O/sweep-$arm.txt")" = 4
  if [ "$arm" = kernel ]; then
    grep -qF 'the card equals the CPU replay bit for bit' "$O/sweep-$arm.txt"
  fi
done
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
  --name "attn-bench${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
