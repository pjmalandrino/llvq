#!/usr/bin/env bash
# The grouped attention on the sealed 8B: the 4B's protocol of 2026-11-01 replayed at 8B, speed against
# the context (LLVQ_GQA_AB) and the paired WikiText-2 perplexity of the tolerance gate (LLVQ_GQA_PPL), in
# one job. Preregistered: proofs/preregistration-gqa-8b-2026-11-04.md, stamped before this.
#
#   DRY_RUN=1 bash ops/jobs/gqa-8b.sh                                        prints and parses the job script
#   GSM8K_4B_VERDICT=G1|G3 IMAGE_SHA=<space sha> bash ops/jobs/gqa-8b.sh      launches, after the operator's go
#
# The sealed 8B only. Four steps in one job: oracle; the published arm replayed (paper-table run 2, flags
# spelled out, against the dense arm, decoded line checked by sha256 against that run), which checks that
# the default path did not move at 8B; LLVQ_GQA_AB=5,2048,4096,8192 on the same flags; LLVQ_GQA_PPL=4096,12
# on the same flags. The config is written by the job from this command line and checked by sha256.
#
# Why flags and not LLVQ_CONFIG: LLVQ_GQA_AB and LLVQ_GQA_PPL are measurement modes, refused by name beside
# LLVQ_CONFIG (bin/fusedrun.rs, the `served.is_some()` loop). The launcher derives the five served
# variables from configs/qwen3-8b-tetra-e4.json field by field and refuses to launch if they differ from
# FLAGS below. The config carries no `attention` field, so its form is `expanded` on the `Cat` store,
# which is the published arm `cat+expanded` of both modes.
#
# The 8B runs only after the 4B GSM8K under the grouped config landed in G1 or G3
# (proofs/preregistration-gsm8k-grouped-2026-11-04.md section 7). The launcher cannot read a verdict, so
# it asks for GSM8K_4B_VERDICT, copied by hand from that journal; the operator's go decides.
#
# Cost, estimated (l40sx1 $1.80/h): about 48 min, $1.44. Ceiling 85m, $2.55.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

OBJ_DIR=sealed-8b27-2026-09-24
NAME=qwen3-8b-sealed-B.bin
LOCAL=$HOME/q8b-14b-sealed-2026-09-23/$NAME
TIMEOUT=85m
TABLES='2 tables (model.embed_tokens.weight + lm_head.weight)'
PREREG=proofs/preregistration-gqa-8b-2026-11-04.md
BUCKET=Pier-Jean/jobs-artifacts
CONF_LOCAL=configs/qwen3-8b-tetra-e4.json
# Paper-table run 2 of 2026-09-25, the 8B: its decoded line and the object's sha256 as it read them on the
# mount. The bytes as measured on the bucket copy (sealed-8b-27-2026-09-24.txt:12) and again on the mount
# (ppl-scelles-2026-10-04.txt:23).
REF_DIR=docs/mesures/paper-table-2026-09-25-brut/paper-served-8b-2026-09-25-r2
REF=$REF_DIR/fused-q4-256.txt
REF_LINE_SHA=323cabeb8520cba9cede5ea6c1a611a1ceb5322c0aae473fac63ed911d61e498
DENSE_LINE='fused F0: 256 tokens identical to the dense arm'
SHA_REF=7bdb9a5503518081b1f652a93215250898caae5cc1b6a2f33ee9fb6094565c13
BYTES_REF=2815098745
# The five served fields of $CONF_LOCAL as the bench door's variables. Checked against the file below.
FLAGS='LLVQ_FUSED_LAYOUT=tetra48 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_KV=f16 LLVQ_EMBED=q4'
LENGTHS=5,2048,4096,8192
PPL_SPEC=4096,12
# The scored stream of ppl-scelles-2026-10-04 and gqa-ppl-2026-11-01: wikitext-2 test, 12 windows of 4,096.
# `ppl` printed it for all three sealed files, each read with the tokenizer it carries (ppl-scelles:27).
TOKENS=3f1baca9033bf251
# RETRY=2 names a relaunch after a failed attempt left its directory behind.
O=/out/gqa-8b-2026-11-04${RETRY:+-r$RETRY}
F=/out/$OBJ_DIR/$NAME
C=$O/qwen3-8b-tetra-e4.json
DRY=${DRY_RUN:-0}

CONF=$(cat "$CONF_LOCAL")
CONF_SHA=$(printf '%s\n' "$CONF" | shasum -a 256 | cut -d' ' -f1)
[ "$CONF_SHA" = "$(shasum -a 256 < "$CONF_LOCAL" | cut -d' ' -f1)" ] \
  || { echo "refused: $CONF_LOCAL does not end in exactly one newline" >&2; exit 1; }
# The config as paper-table run 2 wrote it on the mount: the object this job replays.
[ "$CONF_SHA" = "$(awk '$2 ~ /qwen3-8b-tetra-e4\.json$/ {print $1}' "$REF_DIR/files.sha256")" ] \
  || { echo "refused: $CONF_LOCAL is not the config of paper-table run 2" >&2; exit 1; }
# Field by field: each served field of the file, mapped to the variable the bench door reads, must give
# FLAGS exactly; no `attention` field (the published arm is expanded); no field this mapping does not know.
python3 -c 'import json,sys
d=json.load(open(sys.argv[1])); flags=sys.argv[2]
known={"layout","embed","rot_share","fuse","kv","attention","note"}
assert set(d) <= known, ("unknown field", set(d)-known)
assert d.get("attention") is None, ("the published arm is expanded", d.get("attention"))
var={"layout":"LLVQ_FUSED_LAYOUT","rot_share":"LLVQ_ROT_SHARE","fuse":"LLVQ_FUSE","kv":"LLVQ_KV","embed":"LLVQ_EMBED"}
want=" ".join(f"{var[k]}={d[k]}" for k in ("layout","rot_share","fuse","kv","embed"))
assert want==flags, (want, flags)
assert (d["layout"],d["embed"],d["rot_share"],d["fuse"],d["kv"])==("tetra48","q4","1","0","f16"), d' \
  "$CONF_LOCAL" "$FLAGS"
[ "$(grep -F '  fused F0:  ' "$REF" | shasum -a 256 | cut -d' ' -f1)" = "$REF_LINE_SHA" ] \
  || { echo "refused: the reference line of $REF does not hash to $REF_LINE_SHA" >&2; exit 1; }
grep -qF "$DENSE_LINE" "$REF" || { echo "refused: $REF does not carry the line gate 4 expects" >&2; exit 1; }
grep -F 'embedding: q4 g64 (LLVQ_EMBED)' "$REF" | grep -qF "$TABLES" \
  || { echo "refused: $REF does not carry the embedding line gate 4 expects" >&2; exit 1; }
[ "$SHA_REF" = "$(awk -v p="$F" '$2 == p {print $1}' "$REF_DIR/files.sha256")" ] \
  || { echo "refused: $REF_DIR/files.sha256 does not give $SHA_REF for $F" >&2; exit 1; }

if [ "$DRY" = 1 ]; then
  SHA=0000000000000000000000000000000000000000000000000000000000000000
  BYTES=0
else
  case "${GSM8K_4B_VERDICT:-}" in
    G1|G3) ;;
    *) echo "refused: the 8B runs after the 4B GSM8K grouped run landed in G1 or G3; GSM8K_4B_VERDICT=${GSM8K_4B_VERDICT:-unset}" >&2
       exit 1 ;;
  esac
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  test -f "$LOCAL" || { echo "refused: $LOCAL missing" >&2; exit 1; }
  SHA=$(shasum -a 256 "$LOCAL" | cut -d' ' -f1)
  BYTES=$(stat -f %z "$LOCAL")
  [ "$SHA" = "$SHA_REF" ] || { echo "refused: $LOCAL hashes to $SHA, paper-table run 2 read $SHA_REF" >&2; exit 1; }
  [ "$BYTES" = "$BYTES_REF" ] || { echo "refused: $LOCAL is $BYTES B, expected $BYTES_REF B" >&2; exit 1; }
  REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/$OBJ_DIR/" 2>/dev/null | awk -v n="/$NAME" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
  [ "$REMOTE" = "$BYTES" ] || { echo "refused: bucket copy is ${REMOTE:-absent} B, local $BYTES B" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of the image this job runs on, ef658736 at the prereg>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "8b sealed: $BYTES B, sha256 $SHA; config $CONF_LOCAL sha256 $CONF_SHA; image $NOW; 4B GSM8K $GSM8K_4B_VERDICT"
fi

PRE=$(printf 'O=%q\nF=%q\nC=%q\nSHA=%q\nBYTES=%q\nCONF=%q\nCONF_SHA=%q\nTABLES=%q\nREF_LINE_SHA=%q\nDENSE_LINE=%q\nFLAGS=%q\nLENGTHS=%q\nPPL_SPEC=%q\nTOKENS=%q' \
  "$O" "$F" "$C" "$SHA" "$BYTES" "$CONF" "$CONF_SHA" "$TABLES" "$REF_LINE_SHA" "$DENSE_LINE" "$FLAGS" "$LENGTHS" "$PPL_SPEC" "$TOKENS")

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
echo "served fields as bench variables: $FLAGS"
PARITY='int4: tv_q4_hh equals tv_q4_h bit for bit'
TILE='tile 64 (served: measured optimum for sm_89)'
echo '== STEP 2: the published arm replayed, 256 tokens against the dense arm, same process =='
date -u
env $FLAGS fusedrun "$F" 256 2>&1 | tee "$O/published-q4-256.txt" | tail -30
date -u
grep -qF "$TILE" "$O/published-q4-256.txt"
grep -qF "$PARITY" "$O/published-q4-256.txt"
grep -F 'embedding: q4 g64 (LLVQ_EMBED)' "$O/published-q4-256.txt" | grep -qF "$TABLES"
grep -qF "$DENSE_LINE" "$O/published-q4-256.txt"
test "$(grep -cF '  fused F0:  ' "$O/published-q4-256.txt")" = 1
test "$(grep -F '  fused F0:  ' "$O/published-q4-256.txt" | sha256sum | cut -d' ' -f1)" = "$REF_LINE_SHA"
echo 'gate 4: decoded line identical to paper-table run 2'
echo "== STEP 3: LLVQ_GQA_AB=$LENGTHS, three arms, one model, 256 tokens =="
date -u
env $FLAGS LLVQ_GQA_AB="$LENGTHS" fusedrun "$F" 256 2>&1 | tee "$O/gqa-ab-256.txt" | tail -40
date -u
grep -qF "$TILE" "$O/gqa-ab-256.txt"
grep -qF "$PARITY" "$O/gqa-ab-256.txt"
grep -F 'embedding: q4 g64 (LLVQ_EMBED)' "$O/gqa-ab-256.txt" | grep -qF "$TABLES"
grep -qF 'cudarc event tracking: off (one stream, ordered by issue)' "$O/gqa-ab-256.txt"
test "$(grep -cE '^prompt [0-9]+$' "$O/gqa-ab-256.txt")" = 4
test "$(grep -cE '^  cat\+expanded +[0-9.]+ \[' "$O/gqa-ab-256.txt")" = 4
for arm in shared grouped; do
  test "$(grep -cE "^  grow\(256\)\+$arm +logits against cat\+expanded: " "$O/gqa-ab-256.txt")" = 4
  test "$(grep -cE "^  grow\(256\)\+$arm +[0-9.]+ \[.*(tokens identical|differs from token)" "$O/gqa-ab-256.txt")" = 4
done
echo 'gate 5: four lengths, three arms, logit and speed lines complete'
echo "== STEP 4: LLVQ_GQA_PPL=$PPL_SPEC, two arms, prefill and decode paths =="
date -u
env $FLAGS LLVQ_GQA_PPL="$PPL_SPEC" fusedrun "$F" 2>&1 | tee "$O/gqa-ppl.txt" | tail -24
date -u
grep -qF "$TILE" "$O/gqa-ppl.txt"
grep -qF "$PARITY" "$O/gqa-ppl.txt"
grep -F 'embedding: q4 g64 (LLVQ_EMBED)' "$O/gqa-ppl.txt" | grep -qF "$TABLES"
grep -qF 'cudarc event tracking: off (one stream, ordered by issue)' "$O/gqa-ppl.txt"
grep -qF "(tokens $TOKENS)" "$O/gqa-ppl.txt"
test "$(grep -cE '^  window +[0-9]+/12  prefill nll ' "$O/gqa-ppl.txt")" = 12
test "$(grep -cE '^  (prefill|decode) +[0-9.]+ +[0-9.]+ +[-+][0-9.]+ % \[' "$O/gqa-ppl.txt")" = 2
# The harness: both paths score the same tokens, so on the published arm they agree to rounding.
awk '/^  harness: decode over prefill/ {r = $NF; ok = (r >= 0.99 && r <= 1.01)} END {exit !ok}' "$O/gqa-ppl.txt"
echo 'gate 6: decode and prefill agree on the published arm'
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
  --name "gqa-8b${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
