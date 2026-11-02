#!/usr/bin/env bash
# The served door under the grouped attention: the reference line it owes, and the served object against
# the context, both forms. Preregistered: proofs/preregistration-served-grouped-2026-11-02.md, stamped
# before this.
#
#   DRY_RUN=1 bash ops/jobs/served-grouped.sh                 prints and parses the job script
#   IMAGE_SHA=<space sha> bash ops/jobs/served-grouped.sh      launches, after the operator's go
#
# The sealed 4B only, two configs that differ in one field. Five steps in one job: oracle; the bench door
# replayed (decoded line checked by sha256 against paper-table run 1), which checks that the default path
# did not move when the `attention` field landed; the served door under each config, whose decoded lines
# are hashed and compared; then LLVQ_CONTEXTS beside each config, the one measurement mode the served
# door accepts.
#
# Cost, estimated (l40sx1 $1.80/h): about 29 min, $0.87. Ceiling 60m, $1.80.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

OBJ_DIR=sealed-4b-2026-09-23
NAME=qwen3-4b-sealed.bin
LOCAL=$HOME/q4b-sealed-2026-09-23/$NAME
TIMEOUT=60m
TABLES='1 table (model.embed_tokens.weight, lm_head tied on it)'
PREREG=proofs/preregistration-served-grouped-2026-11-02.md
BUCKET=Pier-Jean/jobs-artifacts
CONF_EXP_LOCAL=configs/qwen3-4b-tetra-e4.json
CONF_GRP_LOCAL=configs/qwen3-4b-tetra-e4-grouped.json
# The decoded line of run 1 of the paper table, `grep -F '  fused F0:  '` of
# docs/mesures/paper-table-2026-09-25-brut/paper-served-4b-2026-09-25-r2/fused-q4-256.txt.
REF_LINE_SHA=8784ffc423bfbeeaeb9a92abd426573cc45f9a083ba56e61e52d9ff40e3b018f
LENGTHS=5,2048,4096,8192
# RETRY=2 names a relaunch after a failed attempt left its directory behind.
O=/out/served-grouped-4b-2026-11-02${RETRY:+-r$RETRY}
F=/out/$OBJ_DIR/$NAME
CE=$O/qwen3-4b-tetra-e4.json
CG=$O/qwen3-4b-tetra-e4-grouped.json
DRY=${DRY_RUN:-0}

CONF_EXP=$(cat "$CONF_EXP_LOCAL")
CONF_GRP=$(cat "$CONF_GRP_LOCAL")
CONF_EXP_SHA=$(printf '%s\n' "$CONF_EXP" | shasum -a 256 | cut -d' ' -f1)
CONF_GRP_SHA=$(printf '%s\n' "$CONF_GRP" | shasum -a 256 | cut -d' ' -f1)
for pair in "$CONF_EXP_LOCAL:$CONF_EXP_SHA" "$CONF_GRP_LOCAL:$CONF_GRP_SHA"; do
  f=${pair%:*}; want=${pair#*:}
  [ "$want" = "$(shasum -a 256 < "$f" | cut -d' ' -f1)" ] \
    || { echo "refused: $f does not end in exactly one newline" >&2; exit 1; }
done
# The two configs must differ in the attention field and in nothing else: that is what makes this run an
# A/B of one thing, and a drift in any other field would read as a gain of the form.
python3 -c 'import json,sys
e=json.load(open(sys.argv[1])); g=json.load(open(sys.argv[2]))
k=("layout","embed","rot_share","fuse","kv")
assert tuple(e[x] for x in k)==("tetra48","q4","1","0","f16"), e
assert tuple(g[x] for x in k)==tuple(e[x] for x in k), (e,g)
assert e.get("attention") is None, e
assert g.get("attention")=="grouped", g' "$CONF_EXP_LOCAL" "$CONF_GRP_LOCAL"
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
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of the rebuild that carries the attention field>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "4b sealed: $BYTES B, sha256 $SHA; configs $CONF_EXP_SHA / $CONF_GRP_SHA; image $NOW"
fi

PRE=$(printf 'O=%q\nF=%q\nCE=%q\nCG=%q\nSHA=%q\nBYTES=%q\nCONF_EXP=%q\nCONF_GRP=%q\nCONF_EXP_SHA=%q\nCONF_GRP_SHA=%q\nTABLES=%q\nREF_LINE_SHA=%q\nLENGTHS=%q' \
  "$O" "$F" "$CE" "$CG" "$SHA" "$BYTES" "$CONF_EXP" "$CONF_GRP" "$CONF_EXP_SHA" "$CONF_GRP_SHA" "$TABLES" "$REF_LINE_SHA" "$LENGTHS")

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
echo '== the two served configs, written from the launch command =='
printf '%s\n' "$CONF_EXP" > "$CE"
printf '%s\n' "$CONF_GRP" > "$CG"
sha256sum "$CE" "$CG" | tee -a "$O/files.sha256"
test "$(sha256sum "$CE" | cut -d' ' -f1)" = "$CONF_EXP_SHA"
test "$(sha256sum "$CG" | cut -d' ' -f1)" = "$CONF_GRP_SHA"
cat "$CE" "$CG"
PARITY='int4: tv_q4_hh equals tv_q4_h bit for bit'
TILE='tile 64 (served: measured optimum for sm_89)'
FLAGS='LLVQ_FUSED_LAYOUT=tetra48 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_KV=f16 LLVQ_EMBED=q4'
echo '== STEP 2: the bench door replayed, 256 tokens against the dense arm, same process =='
date -u
env $FLAGS fusedrun "$F" 256 2>&1 | tee "$O/bench-q4-256.txt" | tail -30
date -u
grep -qF "$TILE" "$O/bench-q4-256.txt"
grep -qF "$PARITY" "$O/bench-q4-256.txt"
grep -F 'embedding: q4 g64 (LLVQ_EMBED)' "$O/bench-q4-256.txt" | grep -qF "$TABLES"
grep -qF 'fused F0: 256 tokens identical to the dense arm' "$O/bench-q4-256.txt"
test "$(grep -cF '  fused F0:  ' "$O/bench-q4-256.txt")" = 1
test "$(grep -F '  fused F0:  ' "$O/bench-q4-256.txt" | sha256sum | cut -d' ' -f1)" = "$REF_LINE_SHA"
echo 'gate 4: decoded line identical to paper-table run 1'
echo '== STEPS 3 and 4: the served door under each config, one arm each =='
for arm in expanded grouped; do
  case $arm in
    expanded) C=$CE; WANT='attention expanded on cat' ;;
    grouped)  C=$CG; WANT='attention grouped on grow(256)' ;;
  esac
  date -u
  env LLVQ_CONFIG="$C" fusedrun "$F" 256 2>&1 | tee "$O/served-$arm-256.txt" | tail -20
  date -u
  grep -qF "$TILE" "$O/served-$arm-256.txt"
  grep -qF "$PARITY" "$O/served-$arm-256.txt"
  grep -E '^served config: ' "$O/served-$arm-256.txt" | grep -qF "$WANT"
  grep -qF 'No dense arm on this path' "$O/served-$arm-256.txt"
  test "$(grep -cE '^served : ' "$O/served-$arm-256.txt")" = 1
  # The WHOLE decoded text, not its first line: 256 tokens can hold newlines,
  # and a comparison of first lines would call two different texts equal.
  awk '/^served : /{f=1;next} /No dense arm on this path/{f=0} f' \
    "$O/served-$arm-256.txt" > "$O/line-$arm.txt"
  test -s "$O/line-$arm.txt"
done
sha256sum "$O/line-expanded.txt" "$O/line-grouped.txt" | tee "$O/lines.sha256"
echo '-- the two served texts, equal or not (reported, not a gate) --'
if cmp -s "$O/line-expanded.txt" "$O/line-grouped.txt"; then
  echo "SERVED LINES EQUAL: the grouped config inherits the reference line"
  test "$(sha256sum < "$O/line-expanded.txt" | cut -d' ' -f1)" \
     = "$(sha256sum < "$O/line-grouped.txt" | cut -d' ' -f1)"
else
  echo "SERVED LINES DIFFER: the grouped config needs a reference line of its own"
  diff "$O/line-expanded.txt" "$O/line-grouped.txt" || true
fi
echo "== STEP 5: LLVQ_CONTEXTS=$LENGTHS beside each config, 256 new tokens =="
for arm in expanded grouped; do
  case $arm in
    expanded) C=$CE; WANT='attention expanded on cat' ;;
    grouped)  C=$CG; WANT='attention grouped on grow(256)' ;;
  esac
  date -u
  env LLVQ_CONFIG="$C" LLVQ_CONTEXTS="$LENGTHS" fusedrun "$F" 256 2>&1 \
    | tee "$O/sweep-$arm.txt" | tail -12
  date -u
  grep -qF "$TILE" "$O/sweep-$arm.txt"
  grep -qF "$PARITY" "$O/sweep-$arm.txt"
  grep -E '^served config: ' "$O/sweep-$arm.txt" | grep -qF "$WANT"
  grep -qE '^  *5 ' "$O/sweep-$arm.txt"
  test "$(grep -cE '^ +[0-9]+ +[0-9.]+ \[' "$O/sweep-$arm.txt")" = 4
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
  --name "served-grouped-4b${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
