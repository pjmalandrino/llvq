#!/usr/bin/env bash
# The 14B under the grouped config: the served door and a light MMLU (5 questions a
# subject, 285) under both configs. Preregistered: proofs/preregistration-served-grouped-14b-2026-11-04.md,
# stamped before this. The operator granted the T8 at 14B on 2026-11-04 on condition of this MMLU check.
#
#   DRY_RUN=1 bash ops/jobs/served-grouped-14b.sh                 prints and parses the job script
#   IMAGE_SHA=<space sha> bash ops/jobs/served-grouped-14b.sh      launches, after the operator's go
#
# Every LLVQ_* variable is set on its own command line, never exported.
#
# Cost, estimated (l40sx1 $1.80/h): about 56 min, $1.68. Ceiling 65m, $1.95. No context sweep: the 14B's speeds
# against context were measured the same day (gqa-14b-2026-11-04).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

OBJ_DIR=sealed-14b-2026-09-23
NAME=qwen3-14b-sealed.bin
LOCAL=$HOME/q8b-14b-sealed-2026-09-23/$NAME
SHA_REF=61db37fe7ce8e6a57c5c59f257319c796574981ecb5e450a02c73dd3c17e2ca0
BYTES_REF=5087000541
TIMEOUT=65m
TABLES='2 tables (model.embed_tokens.weight + lm_head.weight)'
PREREG=proofs/preregistration-served-grouped-14b-2026-11-04.md
BUCKET=Pier-Jean/jobs-artifacts
CONF_EXP_LOCAL=configs/qwen3-14b-tetra-e4.json
CONF_GRP_LOCAL=configs/qwen3-14b-tetra-e4-grouped.json
MMLU_LIMIT=5
O=/out/served-grouped-14b-2026-11-04${RETRY:+-r$RETRY}
F=/out/$OBJ_DIR/$NAME
CE=$O/qwen3-14b-tetra-e4.json
CG=$O/qwen3-14b-tetra-e4-grouped.json
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
# The two configs must differ in the attention field and in nothing else.
python3 -c 'import json,sys
e=json.load(open(sys.argv[1])); g=json.load(open(sys.argv[2]))
k=("layout","embed","rot_share","fuse","kv")
assert tuple(e[x] for x in k)==("tetra48","q4","1","0","f16"), e
assert tuple(g[x] for x in k)==tuple(e[x] for x in k), (e,g)
assert e.get("attention") is None, e
assert g.get("attention")=="grouped", g
assert "Qwen3-14B" in e["note"] and "qwen3-14b" in g["note"], "not the 14B configs"' "$CONF_EXP_LOCAL" "$CONF_GRP_LOCAL"

if [ "$DRY" = 1 ]; then
  SHA=$SHA_REF
  BYTES=$BYTES_REF
else
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  test -f "$LOCAL" || { echo "refused: $LOCAL missing" >&2; exit 1; }
  SHA=$(shasum -a 256 "$LOCAL" | cut -d' ' -f1)
  BYTES=$(stat -f %z "$LOCAL")
  [ "$SHA" = "$SHA_REF" ] && [ "$BYTES" = "$BYTES_REF" ] || { echo "refused: $LOCAL is not the sealed 14B" >&2; exit 1; }
  REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/$OBJ_DIR/" 2>/dev/null | awk -v n="/$NAME" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
  [ "$REMOTE" = "$BYTES" ] || { echo "refused: bucket copy is ${REMOTE:-absent} B, local $BYTES B" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha the prereg names>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "14b sealed: $BYTES B, sha256 $SHA; configs $CONF_EXP_SHA / $CONF_GRP_SHA; image $NOW"
fi

PRE=$(printf 'O=%q\nF=%q\nCE=%q\nCG=%q\nSHA=%q\nBYTES=%q\nCONF_EXP=%q\nCONF_GRP=%q\nCONF_EXP_SHA=%q\nCONF_GRP_SHA=%q\nTABLES=%q\nMMLU_LIMIT=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$F" "$CE" "$CG" "$SHA" "$BYTES" "$CONF_EXP" "$CONF_GRP" "$CONF_EXP_SHA" "$CONF_GRP_SHA" "$TABLES" "$MMLU_LIMIT")

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
TILE='tile 64 (served: measured optimum for sm_89)'
PARITY='int4: tv_q4_hh equals tv_q4_h bit for bit'
provenance() {  # log, wanted attention label
  grep -E '^served config: ' "$1" | grep -qF "$2"
  grep -qF "$TILE" "$1"
  grep -qF "$PARITY" "$1"
}

echo '== STEP 2: the served door under each config, 256 tokens =='
for arm in expanded grouped; do
  case $arm in
    expanded) C=$CE; WANT='attention expanded on cat' ;;
    grouped)  C=$CG; WANT='attention grouped on grow(256)' ;;
  esac
  date -u
  env LLVQ_CONFIG="$C" fusedrun "$F" 256 2>&1 | tee "$O/served-$arm-256.txt" | tail -20
  date -u
  provenance "$O/served-$arm-256.txt" "$WANT"
  grep -F 'embedding: q4 g64 (LLVQ_CONFIG)' "$O/served-$arm-256.txt" | grep -qF "$TABLES"
  grep -qF 'No dense arm on this path' "$O/served-$arm-256.txt"
  test "$(grep -cE '^served : ' "$O/served-$arm-256.txt")" = 1
  awk '/^served : /{f=1;next} /No dense arm on this path/{f=0} f' \
    "$O/served-$arm-256.txt" > "$O/line-$arm.txt"
  test -s "$O/line-$arm.txt"
done
sha256sum "$O/line-expanded.txt" "$O/line-grouped.txt" | tee "$O/lines.sha256"
if cmp -s "$O/line-expanded.txt" "$O/line-grouped.txt"; then
  echo "SERVED LINES EQUAL"
else
  echo "SERVED LINES DIFFER (reported, not a gate)"
  diff "$O/line-expanded.txt" "$O/line-grouped.txt" || true
fi

echo "== STEP 3: MMLU, $MMLU_LIMIT questions a subject, under each config =="
for arm in expanded grouped; do
  case $arm in
    expanded) C=$CE; WANT='attention expanded on cat' ;;
    grouped)  C=$CG; WANT='attention grouped on grow(256)' ;;
  esac
  date -u
  LLVQ_CONFIG="$C" LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP="$O/mmlu-14b-$arm.csv" \
    mmlu "$F" cuda "$MMLU_LIMIT" 2>&1 | tee "$O/mmlu-14b-$arm.txt" | tail -12
  date -u
  provenance "$O/mmlu-14b-$arm.txt" "$WANT"
  grep -qF "# attention=$arm" "$O/mmlu-14b-$arm.csv"
  tail -1 "$O/mmlu-14b-$arm.csv" | grep -qE '^# end fingerprint=[0-9a-f]{16} questions=285$'
done
test "$(tail -1 "$O/mmlu-14b-expanded.csv")" = "$(tail -1 "$O/mmlu-14b-grouped.csv")"
echo 'gate: both MMLU dumps end on the same plan'
echo '== raw kept ==' ; ls -la "$O"
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
  --name "served-grouped-14b${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
