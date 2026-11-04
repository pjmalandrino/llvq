#!/usr/bin/env bash
# The cheap bit-exact gestures at the token boundary of the decode loop, Part A: the argmax on the f16
# logits (a1), a key argmax with candle's tie order (a2), the token kept on the card (a3). Preregistered:
# proofs/preregistration-gestures-2026-11-04.md, to be stamped before this runs.
#
#   DRY_RUN=1 bash ops/jobs/gesture-ab.sh                 prints and parses the job script, $0
#   IMAGE_SHA=<space sha> bash ops/jobs/gesture-ab.sh      launches, after the operator's go
#
# The sealed 4B only. Four steps in one job: oracle; the bench door replayed (decoded line checked by
# sha256 against paper-table run 1), which checks that the default path did not move; the served door
# under the grouped config, whose whole decoded text must hash to the text it served on 2026-11-02;
# then LLVQ_GESTURE_AB on the served flags.
#
# Cost, estimated (l40sx1 $1.80/h): about 12 min, $0.35. Ceiling 20m, $0.60, the operator's cap.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

OBJ_DIR=sealed-4b-2026-09-23
NAME=qwen3-4b-sealed.bin
LOCAL=$HOME/q4b-sealed-2026-09-23/$NAME
TIMEOUT=20m
TABLES='1 table (model.embed_tokens.weight, lm_head tied on it)'
PREREG=proofs/preregistration-gestures-2026-11-04.md
BUCKET=Pier-Jean/jobs-artifacts
CONF_GRP_LOCAL=configs/qwen3-4b-tetra-e4-grouped.json
# The decoded line of run 1 of the paper table, `grep -F '  fused F0:  '` of
# docs/mesures/paper-table-2026-09-25-brut/paper-served-4b-2026-09-25-r2/fused-q4-256.txt.
REF_LINE_SHA=8784ffc423bfbeeaeb9a92abd426573cc45f9a083ba56e61e52d9ff40e3b018f
# The whole decoded text the served door printed under both configs on 2026-11-02, extracted from the
# `served : ` line to `No dense arm on this path` (served-grouped.sh), not the bench door's line: the two
# doors print in different formats.
SERVED_TEXT_SHA=0597cee8b20befecdf92c2f3f2f1938d8216bd244542faab42d2637920a3c12a
# RETRY=2 names a relaunch after a failed attempt left its directory behind.
O=/out/gesture-ab-4b-2026-11-04${RETRY:+-r$RETRY}
F=/out/$OBJ_DIR/$NAME
CG=$O/qwen3-4b-tetra-e4-grouped.json
DRY=${DRY_RUN:-0}

CONF_GRP=$(cat "$CONF_GRP_LOCAL")
CONF_GRP_SHA=$(printf '%s\n' "$CONF_GRP" | shasum -a 256 | cut -d' ' -f1)
[ "$CONF_GRP_SHA" = "$(shasum -a 256 < "$CONF_GRP_LOCAL" | cut -d' ' -f1)" ] \
  || { echo "refused: $CONF_GRP_LOCAL does not end in exactly one newline" >&2; exit 1; }
# The A/B spells the served flags out on the bench door; the config is what they must equal.
python3 -c 'import json,sys
g=json.load(open(sys.argv[1]))
k=("layout","embed","rot_share","fuse","kv","attention")
assert tuple(g[x] for x in k)==("tetra48","q4","1","0","f16","grouped"), g' "$CONF_GRP_LOCAL"
REF=docs/mesures/paper-table-2026-09-25-brut/paper-served-4b-2026-09-25-r2/fused-q4-256.txt
[ "$(grep -F '  fused F0:  ' "$REF" | shasum -a 256 | cut -d' ' -f1)" = "$REF_LINE_SHA" ] \
  || { echo "refused: the reference line of $REF does not hash to $REF_LINE_SHA" >&2; exit 1; }
grep -q "^$SERVED_TEXT_SHA .*line-grouped.txt\$" docs/mesures/served-grouped-2026-11-02-brut/lines.sha256 \
  || { echo "refused: the served text of 2026-11-02 is not $SERVED_TEXT_SHA" >&2; exit 1; }

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
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of the rebuild that carries LLVQ_GESTURE_AB>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "4b sealed: $BYTES B, sha256 $SHA; config $CONF_GRP_SHA; image $NOW"
fi

PRE=$(printf 'O=%q\nF=%q\nCG=%q\nSHA=%q\nBYTES=%q\nCONF_GRP=%q\nCONF_GRP_SHA=%q\nTABLES=%q\nREF_LINE_SHA=%q\nSERVED_TEXT_SHA=%q' \
  "$O" "$F" "$CG" "$SHA" "$BYTES" "$CONF_GRP" "$CONF_GRP_SHA" "$TABLES" "$REF_LINE_SHA" "$SERVED_TEXT_SHA")

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
echo '== STEP 3: the served door, grouped, unchanged =='
date -u
env LLVQ_CONFIG="$CG" fusedrun "$F" 256 2>&1 | tee "$O/served-grouped-256.txt" | tail -20
date -u
grep -qF "$TILE" "$O/served-grouped-256.txt"
grep -qF "$PARITY" "$O/served-grouped-256.txt"
grep -E '^served config: ' "$O/served-grouped-256.txt" | grep -qF 'attention grouped on grow(256)'
grep -qF 'No dense arm on this path' "$O/served-grouped-256.txt"
test "$(grep -cE '^served : ' "$O/served-grouped-256.txt")" = 1
# The WHOLE decoded text, extracted as served-grouped.sh extracted the reference.
awk '/^served : /{f=1;next} /No dense arm on this path/{f=0} f' \
  "$O/served-grouped-256.txt" > "$O/line-grouped.txt"
test -s "$O/line-grouped.txt"
sha256sum "$O/line-grouped.txt" | tee "$O/lines.sha256"
test "$(sha256sum < "$O/line-grouped.txt" | cut -d' ' -f1)" = "$SERVED_TEXT_SHA"
echo 'gate 5: served text identical to the one of 2026-11-02'
echo '== STEP 4: LLVQ_GESTURE_AB=1, the served flags, 256 tokens =='
date -u
env $FLAGS LLVQ_GESTURE_AB=1 fusedrun "$F" 256 2>&1 | tee "$O/gesture-ab-256.txt" | tail -60
date -u
A=$O/gesture-ab-256.txt
grep -qF "$TILE" "$A"
grep -qF "$PARITY" "$A"
grep -qF 'cudarc event tracking: off (one stream, ordered by issue), before the load' "$A"
grep -qF 'attention grouped on grow(256), layout tetra48' "$A"
grep -qF "P1 tok_argmax: candle's fast_argmax_f16 index on 40 cases in sequence, NaN flag raised" "$A"
grep -qF 'P2 9 of 9 card mutants killed on the card, each edit found exactly once' "$A"
test "$(grep -cE '^  m[1-9] .*: [0-9] edit\(s\), each found once; killed, ' "$A")" = 9
grep -qE '^P3 kernel time, us a launch ' "$A"
grep -qE '^P4 host time behind a 5 ms spin, ms ' "$A"
grep -qF '5 arms × 5 interleaved rounds after 2 discarded, 256 identical tokens in every arm and round' "$A"
test "$(grep -cE '^ +[a-z-]+ / [a-z0-9]+ +[0-9.]+ \[' "$A")" = 5
grep -qE '^clock E, ms a decode step ' "$A"
grep -qE '^clock H/D: [0-9]+ of 315 spun steps valid' "$A"
echo 'gate 6: the A/B printed every line its prereg names'
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
  --name "gesture-ab-4b${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
