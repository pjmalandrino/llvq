#!/usr/bin/env bash
# P2: t(k), one forward of k rows through the served 4B, k = 1 to 8, on the grouped config. It prices one extra
# verification row of exact speculative decoding. Preregistered: proofs/preregistration-tk-sweep-2026-11-04.md,
# stamped before this.
#
#   DRY_RUN=1 bash ops/jobs/tk-sweep.sh                       prints and parses the job script
#   IMAGE_SHA=<space sha> bash ops/jobs/tk-sweep.sh            launches, after the operator's go
#
# The sealed 4B only, one config. Three steps in one job: oracle; LLVQ_PREFILL_TOKENS=k beside the config, one
# process for each k, k = 1..8 then 8..1, so drift across the job shows as a gap between the two orders; then
# LLVQ_CONTEXTS=5,2048 beside the same config, for the one-token decode step at 5 and 2,048 positions in this job.
# LLVQ_PREFILL_TOKENS takes ONE length a process (fusedrun.rs, `v.parse::<usize>()`), and times k rows from
# position 0 on fresh caches: no mode times k rows at an offset, so t(k) at 2,048 is not measured here.
#
# Cost, estimated (l40sx1 $1.80/h): about 3.5 min billed, $0.11. Ceiling 15m, $0.45.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

OBJ_DIR=sealed-4b-2026-09-23
NAME=qwen3-4b-sealed.bin
LOCAL=$HOME/q4b-sealed-2026-09-23/$NAME
# The object as the prereg names it, measured on the Mac on 2026-11-04.
OBJ_SHA=886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8
OBJ_BYTES=1418224685
TIMEOUT=15m
TABLES='1 table (model.embed_tokens.weight, lm_head tied on it)'
PREREG=proofs/preregistration-tk-sweep-2026-11-04.md
ECARTS=proofs/preregistration-tk-sweep-2026-11-04-ECARTS.md
BUCKET=Pier-Jean/jobs-artifacts
CONF_GRP_LOCAL=configs/qwen3-4b-tetra-e4-grouped.json
CONF_GRP_PIN=2455fe858076217664f5e57083db0c6f5eb6ad3b16cdffa210265e41a0d42a42
# The image of served-grouped (built from f808856). From f808856 to 06e1602 the code differs only in a comment
# and an error string of the LLVQ_CONTEXTS block of fusedrun.rs, which fires only under LLVQ_KV_PREALLOC.
WANT_IMAGE=f77f803eab81bc286225df6ca5a235eb08b66862
KS_FWD='1 2 3 4 5 6 7 8'
KS_REV='8 7 6 5 4 3 2 1'
CTX=5,2048
# RETRY=2 names a relaunch after a failed attempt left its directory behind.
O=/out/tk-sweep-4b-2026-11-04${RETRY:+-r$RETRY}
F=/out/$OBJ_DIR/$NAME
CG=$O/qwen3-4b-tetra-e4-grouped.json
DRY=${DRY_RUN:-0}

CONF_GRP=$(cat "$CONF_GRP_LOCAL")
CONF_GRP_SHA=$(printf '%s\n' "$CONF_GRP" | shasum -a 256 | cut -d' ' -f1)
[ "$CONF_GRP_SHA" = "$(shasum -a 256 < "$CONF_GRP_LOCAL" | cut -d' ' -f1)" ] \
  || { echo "refused: $CONF_GRP_LOCAL does not end in exactly one newline" >&2; exit 1; }
[ "$CONF_GRP_SHA" = "$CONF_GRP_PIN" ] \
  || { echo "refused: $CONF_GRP_LOCAL is $CONF_GRP_SHA, the prereg names $CONF_GRP_PIN" >&2; exit 1; }
python3 -c 'import json,sys
g=json.load(open(sys.argv[1]))
k=("layout","embed","rot_share","fuse","kv","attention")
assert tuple(g.get(x) for x in k)==("tetra48","q4","1","0","f16","grouped"), g' "$CONF_GRP_LOCAL"

if [ "$DRY" = 1 ]; then
  SHA=0000000000000000000000000000000000000000000000000000000000000000
  BYTES=0
else
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  test -f "$LOCAL" || { echo "refused: $LOCAL missing" >&2; exit 1; }
  SHA=$(shasum -a 256 "$LOCAL" | cut -d' ' -f1)
  BYTES=$(stat -f %z "$LOCAL")
  [ "$SHA" = "$OBJ_SHA" ] && [ "$BYTES" = "$OBJ_BYTES" ] \
    || { echo "refused: $LOCAL is $BYTES B sha256 $SHA, the prereg names $OBJ_BYTES B $OBJ_SHA" >&2; exit 1; }
  REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/$OBJ_DIR/" 2>/dev/null | awk -v n="/$NAME" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
  [ "$REMOTE" = "$BYTES" ] || { echo "refused: bucket copy is ${REMOTE:-absent} B, local $BYTES B" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  # Another image is a deviation, written beside the prereg before the run.
  IMAGE_SHA=${IMAGE_SHA:-$WANT_IMAGE}
  if [ "$IMAGE_SHA" != "$WANT_IMAGE" ] && ! test -f "$ECARTS"; then
    echo "refused: image $IMAGE_SHA is not the prereg's $WANT_IMAGE, and $ECARTS is missing" >&2; exit 1
  fi
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "4b sealed: $BYTES B, sha256 $SHA; config $CONF_GRP_SHA; image $NOW"
fi

PRE=$(printf 'O=%q\nF=%q\nCG=%q\nSHA=%q\nBYTES=%q\nCONF_GRP=%q\nCONF_GRP_SHA=%q\nTABLES=%q\nKS_FWD=%q\nKS_REV=%q\nCTX=%q' \
  "$O" "$F" "$CG" "$SHA" "$BYTES" "$CONF_GRP" "$CONF_GRP_SHA" "$TABLES" "$KS_FWD" "$KS_REV" "$CTX")

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
printf '%s\n' "$CONF_GRP" > "$CG"
sha256sum "$CG" | tee -a "$O/files.sha256"
test "$(sha256sum "$CG" | cut -d' ' -f1)" = "$CONF_GRP_SHA"
cat "$CG"
PARITY='int4: tv_q4_hh equals tv_q4_h bit for bit'
TILE='tile 64 (served: measured optimum for sm_89)'
PAIR='prefill 4x128 (served pair)'
WANT='attention grouped on grow(256)'
# Gates 3 and 4 of the prereg, on one output file: they void the run when they fail.
gate() {
  grep -qF "$TILE" "$1"
  grep -qF "$PAIR" "$1"
  grep -qF "$PARITY" "$1"
  grep -F 'embedding: q4 g64 (LLVQ_CONFIG)' "$1" | grep -qF "$TABLES"
  grep -E '^served config: ' "$1" | grep -qF "$WANT"
}
echo '== STEP 2: t(k), one process a k, forward order then reverse =='
: > "$O/rc.txt"
for order in fwd rev; do
  case $order in fwd) LIST=$KS_FWD ;; rev) LIST=$KS_REV ;; esac
  for k in $LIST; do
    T=$O/pf-$k-$order.txt
    date -u
    # A guard refusal exits non-zero AFTER the timing lines; the sweep goes on and the code is kept.
    rc=0
    env LLVQ_CONFIG="$CG" LLVQ_PREFILL_TOKENS="$k" fusedrun "$F" > "$T" 2>&1 || rc=$?
    if grep -qF 'pick different tokens' "$T"; then why=guard-refused
    elif [ "$rc" = 0 ]; then why=ok
    else why=failed; fi
    echo "k=$k order=$order rc=$rc $why" | tee -a "$O/rc.txt"
    grep -E 'ms a prefill of|ms a token of prompt|max \|delta logit\|' "$T" || echo "  no timing line for k=$k ($order)"
    gate "$T"
    grep -E '^prefill: ' "$T" | grep -qF "prefill: $k tokens, layout tetra48, $WANT"
  done
done
echo '-- t(k) medians, both orders (reported; the rule is read on the Mac) --'
for k in $KS_FWD; do
  for order in fwd rev; do
    printf 'k=%s %s ' "$k" "$order"
    grep -E "^ +[0-9.]+ ms a prefill of $k tokens " "$O/pf-$k-$order.txt" | tr -s ' ' || echo 'missing'
  done
done | tee "$O/tk-summary.txt"
echo "== STEP 3: LLVQ_CONTEXTS=$CTX beside the same config, 256 new tokens =="
date -u
rc=0
env LLVQ_CONFIG="$CG" LLVQ_CONTEXTS="$CTX" fusedrun "$F" 256 > "$O/contexts.txt" 2>&1 || rc=$?
date -u
echo "contexts rc=$rc" | tee -a "$O/rc.txt"
tail -6 "$O/contexts.txt"
gate "$O/contexts.txt"
grep -E '^layout tetra48, ' "$O/contexts.txt" | grep -qF "$WANT"
grep -cE '^ +[0-9]+ +[0-9.]+ \[' "$O/contexts.txt" || true
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
  --name "tk-sweep-4b${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
