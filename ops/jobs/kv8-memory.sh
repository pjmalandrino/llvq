#!/usr/bin/env bash
# The int8 decode cache as a memory lead (gain 2 of dev/trois-gains-2026-11-15): on one L40S, the sealed 4B under the
# NOT SERVED configs/qwen3-4b-tetra-e4-4e-v9-kv8-NOT-SERVED.json (K8: arm v9k8, the binary16 caches released at the
# hand-over) against the served configs/qwen3-4b-tetra-e4-4e-v9.json (V9), each arm in its own process. One job:
#   (1) the load check (`LLVQ_PERS_4E=check:v9,v9k8`, no model): v9 loads, v9k8's module is the Mac's and its token
#       symbol line is printed, or the job STOPS (K8 is timed with whatever local memory the card gives it);
#   (2) V9 and (3) K8 under the eager prefill, (4) V9 and (5) K8 under the chain prefill (a12ipg32ft at 32 rows), each
#       a prefillbench process at 32 / 1,024 / 4,096 / 8,191 tokens with the 44 house prompts dumped, under both
#       samplers (the process peak, and the peak inside each `mark time-<l>` and `mark dec-<l>` window);
#   (6) GSM8K-200 under K8 (the measurement door) and under V9, then gsm8kpair;
#   then the house texts of int8 against binary16 under each prefill, compared token for token.
# Output kv8-memory-2026-11-15.
# Preregistered: proofs/preregistration-kv8-memory-2026-11-15-DRAFT.md, to sign and stamp before this (the stamped
# file drops -DRAFT from its name; PREREG below names the stamped one).
#
#   DRY_RUN=1 bash ops/jobs/kv8-memory.sh                     parses, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/kv8-memory.sh         launches, after the operator's go
#
# Cost (l40sx1 $1.80/h), estimated from measured times (prereg's cost section): about 3,310 s, 55 min, $1.66.
# Timeout 75m: $2.25 at worst.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-kv8-memory-2026-11-15.md
OUT_NAME=kv8-memory-2026-11-15
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/$OUT_NAME${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
P4E_V9_SHA=1b422b62328be6752df4add43280565fd2a49cfa20ee26953b301abe7a15c530
# the check's modules at the card's tile define (pers.rs::the_k8_module_keeps_its_bytes; ops/jobs/recherche-debit-kv8-local.sh)
SHAS="v9:d55a097b146e8b962829045dfea6975470b408b942c1e0c22c038ad4d9d00860 v9k8:8b53b1cdb8d4ff6264fea1685a355045231655f8d6ac1a368c89a8b605c224f9"
DATASET_REV=740312add88f781978c0658806c59bc2815b9866
ARM='a12ipg32ft@8.10.8.20.6'
LENGTHS=32,1024,4096,8191
TIMEOUT=${TIMEOUT:-75m}
NTOK=256
LIMIT=200
PROMPTS=/usr/local/share/llvq/spec-prompts-v1.jsonl
PROMPTS_SHA=$(shasum -a 256 < ops/data/spec-prompts-v1.jsonl | cut -d' ' -f1)
[ "$PROMPTS_SHA" = f749eb359c2cf5c8a648ca99fc0f856e620f8adb3f1d233f4c477ff322257f1a ] \
  || { echo "refused: ops/data/spec-prompts-v1.jsonl is not at its pinned sha256" >&2; exit 1; }
LONG=/usr/local/share/llvq/spec-long-8k-v1.jsonl
LONG_SHA=$(shasum -a 256 < ops/data/spec-long-8k-v1.jsonl | cut -d' ' -f1)
[ "$LONG_SHA" = da4edbd6f12184cf8ff938f057500f3e30e35d8c7a351d2e925316d225d62ebc ] \
  || { echo "refused: ops/data/spec-long-8k-v1.jsonl is not at its pinned sha256" >&2; exit 1; }

SEALED="sealed-4b-2026-09-23/qwen3-4b-sealed.bin:1418224685"
SEALED_SHA=886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8
V9=configs/qwen3-4b-tetra-e4-4e-v9.json
K8=configs/qwen3-4b-tetra-e4-4e-v9-kv8-NOT-SERVED.json
[ "$(shasum -a 256 < "$V9" | cut -d' ' -f1)" = 29b3ef32bbda757b6eeff251e10b5c8f103c4dfa3157bff23ca42fef298a3af0 ] \
  || { echo "refused: $V9 is not the served v9 config (29b3ef32)" >&2; exit 1; }
grep -q '^  "decode_cache": "int8",$' "$K8" || { echo "refused: $K8 does not name the int8 cache" >&2; exit 1; }
V9_SHA=$(shasum -a 256 < "$V9" | cut -d' ' -f1)
K8_SHA=$(shasum -a 256 < "$K8" | cut -d' ' -f1)
V9_B64=$(base64 < "$V9" | tr -d '\n')
K8_B64=$(base64 < "$K8" | tr -d '\n')

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
  echo "bucket copy verified by size; image $NOW"
fi

PRE=$(printf 'O=%q\nSEALED=%q\nSEALED_SHA=%q\nV9_SHA=%q\nK8_SHA=%q\nV9_B64=%q\nK8_B64=%q\nP4E_V9_SHA=%q\nSHAS=%q\nDATASET_REV=%q\nARM=%q\nLENGTHS=%q\nNTOK=%q\nLIMIT=%q\nPROMPTS=%q\nPROMPTS_SHA=%q\nLONG=%q\nLONG_SHA=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$SEALED" "$SEALED_SHA" "$V9_SHA" "$K8_SHA" "$V9_B64" "$K8_B64" "$P4E_V9_SHA" "$SHAS" "$DATASET_REV" "$ARM" "$LENGTHS" "$NTOK" "$LIMIT" "$PROMPTS" "$PROMPTS_SHA" "$LONG" "$LONG_SHA")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
echo '== control 1: oracle =='
timeout 4m oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle.txt" | tail -2
grep -q MATCH "$O/oracle.txt"
echo '== control 1: the object, the two configs, the prompt sets =='
S=/out/${SEALED%%:*}; SB=${SEALED##*:}
test "$(stat -c %s "$S")" = "$SB"
sha256sum "$S" | tee "$O/files.sha256"
test "$(sha256sum "$S" | cut -d' ' -f1)" = "$SEALED_SHA"
PV=$O/qwen3-4b-tetra-e4-4e-v9.json; PK=$O/qwen3-4b-tetra-e4-4e-v9-kv8-NOT-SERVED.json
printf '%s' "$V9_B64" | base64 -d > "$PV"; printf '%s' "$K8_B64" | base64 -d > "$PK"
test "$(sha256sum "$PV" | cut -d' ' -f1)" = "$V9_SHA"
test "$(sha256sum "$PK" | cut -d' ' -f1)" = "$K8_SHA"
for pair in "$PROMPTS:$PROMPTS_SHA" "$LONG:$LONG_SHA"; do
  f=${pair%:*}; h=${pair##*:}
  sha256sum "$f" | tee -a "$O/files.sha256"
  test "$(sha256sum "$f" | cut -d' ' -f1)" = "$h"
done
vram() {
  local tag=$1; shift
  local idle; idle=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 100 > "$O/vram-$tag.csv" &
  local sp=$!
  ( while :; do printf '%s %s\n' "$(date +%s.%N)" "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"; sleep 0.05; done ) > "$O/vram-$tag-t.txt" &
  local st=$!
  local rc=0
  "$@" || rc=$?
  kill "$sp" "$st"; wait "$sp" "$st" 2>/dev/null || true
  local peak; peak=$(sort -n "$O/vram-$tag.csv" | tail -1)
  echo "vram $tag idle $idle MiB peak $peak MiB" | tee -a "$O/vram.txt" >&2
  return $rc
}
phases() {
  awk 'NR == FNR { if ($1 == "mark") { t = $2; if (!(t in seen)) { seen[t] = 1; o[++n] = t; k[t] = 0 }
                     if ($3 == "begin") { k[t]++; b[t, k[t]] = $4 } else e[t, k[t]] = $4 } next }
       { for (i = 1; i <= n; i++) { t = o[i]; for (j = 1; j <= k[t]; j++)
           if (((t, j) in e) && $1 >= b[t, j] && $1 <= e[t, j] && $2 + 0 > p[t]) p[t] = $2 + 0 } }
       END { for (i = 1; i <= n; i++) { t = o[i]; printf "phase peak %s %s: %.3f GB (%d MiB) over %d window(s)\n", tg, t, p[t] * 1048576 / 1e9, p[t], k[t] } }' \
      tg="$1" "$O/$1.txt" "$O/vram-$1-t.txt"
}
echo '== (1) the load check: v9,v9k8, no model =='
date -u
rc=0
env LLVQ_FUSED_LAYOUT=tetra48 LLVQ_EMBED=q4 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_PERS_4E="check:v9,v9k8" timeout 9m fusedrun "$S" 2 > "$O/check.txt" 2>&1 || rc=$?
echo "fusedrun check exit $rc" | tee -a "$O/check.txt"
grep -aE '^4E check|^CHECK' "$O/check.txt" | tee "$O/check-summary.txt"
for pair in $SHAS; do
  arm=${pair%%:*}; sha=${pair#*:}
  grep -aq "^4E check $arm: module [0-9]* bytes, sha256 $sha\$" "$O/check.txt" \
    || { echo "STOP: control 2, $arm's module is not the Mac's $sha" | tee -a "$O/check-summary.txt"; exit 1; }
done
grep -q '^CHECK ARM v9 LOADS' "$O/check.txt" || { echo "STOP: control 2, v9 does not load" | tee -a "$O/check-summary.txt"; exit 1; }
grep -aqE '^CHECK v9k8 pers_4e_tokens_[^ ]*: [0-9]+ registers, [0-9]+ B local' "$O/check.txt" \
  || { echo "STOP: control 2, no token symbol line for v9k8" | tee -a "$O/check-summary.txt"; exit 1; }
echo "the load check passed" | tee -a "$O/check-summary.txt"
set +e
for arm in v9-eager k8-eager v9-chain k8-chain; do
  cfg=$PV; [ "${arm%%-*}" = k8 ] && cfg=$PK
  form=${arm#*-}
  extra=(); [ "$form" = chain ] && extra=(--arm "$ARM" --r 32)
  echo "== $arm =="
  date -u
  vram "$arm" env LLVQ_CONFIG="$cfg" timeout 16m prefillbench --form "$form" "${extra[@]}" --long "$LONG" --lengths "$LENGTHS" \
    --prompts "$PROMPTS" --new "$NTOK" --dump "$O/texts-$arm.jsonl" "$S" > "$O/$arm.txt" 2>&1
  echo "prefillbench $arm exit $?" >> "$O/$arm.txt"
done
echo '== (6) GSM8K-200 under K8, then V9, then the pairing =='
date -u
LLVQ_DATASET_REV="$DATASET_REV" LLVQ_CONFIG="$PK" LLVQ_GSM8K_DUMP="$O/gsm8k-k8.jsonl" timeout 12m gsm8k "$S" cuda "$LIMIT" > "$O/gsm8k-k8.txt" 2>&1
echo "gsm8k k8 exit $?" >> "$O/gsm8k-k8.txt"
LLVQ_DATASET_REV="$DATASET_REV" LLVQ_CONFIG="$PV" LLVQ_GSM8K_DUMP="$O/gsm8k-v9.jsonl" timeout 12m gsm8k "$S" cuda "$LIMIT" > "$O/gsm8k-v9.txt" 2>&1
echo "gsm8k v9 exit $?" >> "$O/gsm8k-v9.txt"
gsm8kpair "$O/gsm8k-v9.jsonl" "$O/gsm8k-k8.jsonl" > "$O/gsm8kpair.txt" 2>&1
echo "gsm8kpair exit $?" >> "$O/gsm8kpair.txt"
date -u
for t in v9-eager k8-eager v9-chain k8-chain; do phases "$t"; done > "$O/phase-peaks.txt" 2>&1
# the house texts, int8 against binary16, token for token (the tokens field of each dump line, in order)
toks() { sed -E 's/.*"tokens":(\[[^]]*\]).*/\1/' "$1"; }
{
  for form in eager chain; do
    a=$O/texts-v9-$form.jsonl; b=$O/texts-k8-$form.jsonl
    if [ -s "$a" ] && [ -s "$b" ]; then
      n=$(wc -l < "$a"); same=$(paste -d'|' <(toks "$a") <(toks "$b") | awk -F'|' '$1 == $2' | wc -l)
      echo "texts $form: int8 identical to binary16 on $same of $n"
      paste -d'|' <(sed -E 's/.*"id":"([^"]*)".*/\1/' "$a") <(toks "$a") <(toks "$b") | awk -F'|' '$2 != $3 {print "  differs: " $1}'
    else
      echo "texts $form: a dump is missing"
    fi
  done
} > "$O/texts.txt" 2>&1
set -e
{
  cat "$O/vram.txt"
  cat "$O/phase-peaks.txt"
  cat "$O/texts.txt"
  for f in v9-eager k8-eager v9-chain k8-chain; do
    echo "== $f =="
    grep -aE '^ +[0-9]+ +[0-9.]+ \[|card memory|4E v9k8|MEASUREMENT ONLY|^4E: persistent module|exit |^Error|^error' "$O/$f.txt" || true
  done
  for f in gsm8k-k8 gsm8k-v9 gsm8kpair; do echo "== $f =="; grep -avE '^  \[ *[0-9]+/' "$O/$f.txt" | tail -20; done
} > "$O/summary.txt" 2>&1
cat "$O/summary.txt"
for f in k8-eager k8-chain gsm8k-k8; do
  grep -aq 'MEASUREMENT ONLY' "$O/$f.txt" \
    && echo "control 3 ($f): MEASUREMENT ONLY printed" | tee -a "$O/summary.txt" \
    || echo "VOID: control 3 ($f), no MEASUREMENT ONLY line" | tee -a "$O/summary.txt"
done
for f in k8-eager k8-chain; do
  grep -aq "each released after its layer's fill" "$O/$f.txt" \
    && echo "control 3 ($f): the binary16 caches released" | tee -a "$O/summary.txt" \
    || echo "VOID: control 3 ($f), no release line in the kv8 report" | tee -a "$O/summary.txt"
done
for f in v9-eager v9-chain gsm8k-v9; do
  grep -qF "4E: persistent module 252757 bytes, sha256 $P4E_V9_SHA, row form 7," "$O/$f.txt" \
    && echo "control 4 ($f): the served 4E v9 module" | tee -a "$O/summary.txt" \
    || echo "VOID: control 4 ($f), the 4E module is not the served v9 module" | tee -a "$O/summary.txt"
done
ls -la "$O"
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
  --name "kv8-memory${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
