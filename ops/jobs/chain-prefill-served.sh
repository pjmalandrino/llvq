#!/usr/bin/env bash
# The chain prefill at the served door (gain 1 of dev/trois-gains-2026-11-15): on one L40S, the sealed 4B under
# configs/qwen3-4b-tetra-e4-4e-v9-chain.json (CH) against the served configs/qwen3-4b-tetra-e4-4e-v9.json (V9), each in
# its own processes. One job:
#   (1) the load check: CH at the served door with an empty module cache, 8 tokens. Two cache misses, the chain arm's
#       install line, the a12ip in-place check, a token ids line and exit 0, or the job STOPS;
#   (2) the same command again: two cache hits, the cached load time, the same token ids as (1);
#   (3) CH at the served door (fusedrun, 5-token prompt, 256 tokens): 4E after the chain against the eager decode after
#       the chain, bit for bit (the binary fails otherwise);
#   (4) CH and (5) V9 under LLVQ_CONTEXTS=5,1024,4096,8192, 256 tokens, under both samplers (process peak and the peak
#       inside each `mark ctx-<n>` window);
#   (6) GSM8K-200 under CH and (7) under V9, then gsm8kpair.
# Output chain-prefill-served-2026-11-15.
# Preregistered: proofs/preregistration-chain-prefill-served-2026-11-15-DRAFT.md, to sign and stamp before this (the
# stamped file drops -DRAFT from its name; PREREG below names the stamped one).
#
#   DRY_RUN=1 bash ops/jobs/chain-prefill-served.sh                     parses, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/chain-prefill-served.sh         launches, after the operator's go
#
# Cost (l40sx1 $1.80/h), estimated from measured times (prereg's last section): about 2,550 s, 43 min, $1.28.
# Timeout 60m: $1.80 at worst.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-chain-prefill-served-2026-11-15.md
OUT_NAME=chain-prefill-served-2026-11-15
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/$OUT_NAME${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
P4E_V9_SHA=1b422b62328be6752df4add43280565fd2a49cfa20ee26953b301abe7a15c530
TREE_SHA=ce1c354d2583eb178ae8622ff99e10f9b457543972bb5039ce89ede9eec31e44
FT_SHA=19b734a3fb6029848d87339c7d505e9bcfc612f182256793026ea964d09853e2
DATASET_REV=740312add88f781978c0658806c59bc2815b9866
TIMEOUT=${TIMEOUT:-60m}
NTOK=256
LIMIT=200

SEALED="sealed-4b-2026-09-23/qwen3-4b-sealed.bin:1418224685"
SEALED_SHA=886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8
V9=configs/qwen3-4b-tetra-e4-4e-v9.json
CH=configs/qwen3-4b-tetra-e4-4e-v9-chain.json
grep -q '^  "rows": "v9",$' "$V9" || { echo "refused: $V9 does not name rows v9" >&2; exit 1; }
grep -q '^  "prefill": "chain",$' "$CH" || { echo "refused: $CH does not name the chain prefill" >&2; exit 1; }
[ "$(shasum -a 256 < "$V9" | cut -d' ' -f1)" = 29b3ef32bbda757b6eeff251e10b5c8f103c4dfa3157bff23ca42fef298a3af0 ] \
  || { echo "refused: $V9 is not the served v9 config (29b3ef32)" >&2; exit 1; }
V9_SHA=$(shasum -a 256 < "$V9" | cut -d' ' -f1)
CH_SHA=$(shasum -a 256 < "$CH" | cut -d' ' -f1)
V9_B64=$(base64 < "$V9" | tr -d '\n')
CH_B64=$(base64 < "$CH" | tr -d '\n')

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

PRE=$(printf 'O=%q\nSEALED=%q\nSEALED_SHA=%q\nV9_SHA=%q\nCH_SHA=%q\nV9_B64=%q\nCH_B64=%q\nP4E_V9_SHA=%q\nTREE_SHA=%q\nFT_SHA=%q\nDATASET_REV=%q\nNTOK=%q\nLIMIT=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$SEALED" "$SEALED_SHA" "$V9_SHA" "$CH_SHA" "$V9_B64" "$CH_B64" "$P4E_V9_SHA" "$TREE_SHA" "$FT_SHA" "$DATASET_REV" "$NTOK" "$LIMIT")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
echo '== control 1: oracle =='
timeout 4m oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle.txt" | tail -2
grep -q MATCH "$O/oracle.txt"
echo '== control 1: the object and the two configs =='
S=/out/${SEALED%%:*}; SB=${SEALED##*:}
test "$(stat -c %s "$S")" = "$SB"
sha256sum "$S" | tee "$O/files.sha256"
test "$(sha256sum "$S" | cut -d' ' -f1)" = "$SEALED_SHA"
PV=$O/qwen3-4b-tetra-e4-4e-v9.json; PC=$O/qwen3-4b-tetra-e4-4e-v9-chain.json
printf '%s' "$V9_B64" | base64 -d > "$PV"; printf '%s' "$CH_B64" | base64 -d > "$PC"
test "$(sha256sum "$PV" | cut -d' ' -f1)" = "$V9_SHA"
test "$(sha256sum "$PC" | cut -d' ' -f1)" = "$CH_SHA"
sha256sum "$PV" "$PC" | tee -a "$O/files.sha256"
MC=$O/modcache
# Two samplers around one command (ops/jobs/large-prefill-ft.sh): the process peak every 100 ms, and an epoch-stamped
# one for the peak inside each phase the binary marks (`mark <tag> begin|end <unix s>`).
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
echo '== (1) the load check: CH, empty module cache =='
date -u
rc=0
LLVQ_CONFIG="$PC" LLVQ_MODULE_CACHE="$MC" timeout 16m fusedrun "$S" 8 > "$O/load-cold.txt" 2>&1 || rc=$?
echo "fusedrun load-cold exit $rc" | tee -a "$O/load-cold.txt"
grep -aE '^4E tree|^chain prefill|^4E: persistent module|loaded in|exit ' "$O/load-cold.txt" | tee "$O/check-summary.txt"
[ "$rc" = 0 ] || { echo "STOP: control 3, the cold load exited $rc" | tee -a "$O/check-summary.txt"; exit 1; }
[ "$(grep -ac '^4E tree: module cache: miss' "$O/load-cold.txt")" = 2 ] \
  || { echo "STOP: control 3, not two cache misses" | tee -a "$O/check-summary.txt"; exit 1; }
grep -aq '^chain prefill: arm a12ipg32ft@8.10.8.20.6' "$O/load-cold.txt" \
  || { echo "STOP: control 3, no chain arm line" | tee -a "$O/check-summary.txt"; exit 1; }
grep -aq 'projections read in place' "$O/load-cold.txt" \
  || { echo "STOP: control 3, no in-place check line" | tee -a "$O/check-summary.txt"; exit 1; }
grep -aqE '^  token ids: [0-9a-f]{16} [0-9]' "$O/load-cold.txt" \
  || { echo "STOP: control 3, no token ids line" | tee -a "$O/check-summary.txt"; exit 1; }
for pair in "4E tree: module 306060 bytes, sha256 $TREE_SHA" "4E tree a12ipg32ft: module 457150 bytes, sha256 $FT_SHA" \
            "4E: persistent module 252757 bytes, sha256 $P4E_V9_SHA"; do
  grep -aqF "$pair" "$O/load-cold.txt" || { echo "STOP: control 2 or 6, not found: $pair" | tee -a "$O/check-summary.txt"; exit 1; }
done
ls -la "$MC" | tee -a "$O/check-summary.txt"
echo "the load check passed" | tee -a "$O/check-summary.txt"
set +e
echo '== (2) the same, the cache filled =='
date -u
LLVQ_CONFIG="$PC" LLVQ_MODULE_CACHE="$MC" timeout 6m fusedrun "$S" 8 > "$O/load-warm.txt" 2>&1
echo "fusedrun load-warm exit $?" >> "$O/load-warm.txt"
echo '== (3) CH at the served door, 5-token prompt =='
date -u
vram door-ch env LLVQ_CONFIG="$PC" LLVQ_MODULE_CACHE="$MC" timeout 8m fusedrun "$S" "$NTOK" > "$O/door-ch.txt" 2>&1
echo "fusedrun door-ch exit $?" >> "$O/door-ch.txt"
echo '== (4) CH, contexts 5, 1,024, 4,096, 8,192 =='
date -u
vram ctx-ch env LLVQ_CONFIG="$PC" LLVQ_MODULE_CACHE="$MC" LLVQ_CONTEXTS=5,1024,4096,8192 timeout 12m fusedrun "$S" "$NTOK" > "$O/ctx-ch.txt" 2>&1
echo "fusedrun ctx-ch exit $?" >> "$O/ctx-ch.txt"
echo '== (5) V9, contexts 5, 1,024, 4,096, 8,192 =='
date -u
vram ctx-v9 env LLVQ_CONFIG="$PV" LLVQ_CONTEXTS=5,1024,4096,8192 timeout 14m fusedrun "$S" "$NTOK" > "$O/ctx-v9.txt" 2>&1
echo "fusedrun ctx-v9 exit $?" >> "$O/ctx-v9.txt"
echo '== (6) GSM8K-200 under CH =='
date -u
LLVQ_DATASET_REV="$DATASET_REV" LLVQ_CONFIG="$PC" LLVQ_MODULE_CACHE="$MC" LLVQ_GSM8K_DUMP="$O/gsm8k-ch.jsonl" \
  timeout 12m gsm8k "$S" cuda "$LIMIT" > "$O/gsm8k-ch.txt" 2>&1
echo "gsm8k ch exit $?" >> "$O/gsm8k-ch.txt"
echo '== (7) GSM8K-200 under V9, then the pairing =='
date -u
LLVQ_DATASET_REV="$DATASET_REV" LLVQ_CONFIG="$PV" LLVQ_GSM8K_DUMP="$O/gsm8k-v9.jsonl" \
  timeout 12m gsm8k "$S" cuda "$LIMIT" > "$O/gsm8k-v9.txt" 2>&1
echo "gsm8k v9 exit $?" >> "$O/gsm8k-v9.txt"
gsm8kpair "$O/gsm8k-v9.jsonl" "$O/gsm8k-ch.jsonl" > "$O/gsm8kpair.txt" 2>&1
echo "gsm8kpair exit $?" >> "$O/gsm8kpair.txt"
date -u
for t in ctx-ch ctx-v9 door-ch; do phases "$t"; done > "$O/phase-peaks.txt" 2>&1
set -e
{
  echo "== load =="
  grep -aE 'module cache|loaded in|exit ' "$O/load-cold.txt" "$O/load-warm.txt" || true
  echo "== token ids, cold and warm =="
  grep -a 'token ids' "$O/load-cold.txt" "$O/load-warm.txt" || true
  cat "$O/vram.txt"
  cat "$O/phase-peaks.txt"
  for f in door-ch ctx-ch ctx-v9; do
    echo "== $f =="
    grep -aE '^served|^  4E|^  eager|tokens equal|first divergence|^ +[0-9]+ +[0-9.]+ |prompt [0-9]+:|exit |^Error|^error' "$O/$f.txt" || true
  done
  for f in gsm8k-ch gsm8k-v9 gsm8kpair; do echo "== $f =="; grep -avE '^  \[ *[0-9]+/' "$O/$f.txt" | tail -20; done
} > "$O/summary.txt" 2>&1
cat "$O/summary.txt"
[ "$(grep -ac '^4E tree: module cache: hit' "$O/load-warm.txt")" = 2 ] \
  && echo "control 4: two cache hits on the second start" | tee -a "$O/summary.txt" \
  || echo "VOID: control 4, not two cache hits on the second start" | tee -a "$O/summary.txt"
TC=$(grep -aE '^  token ids: [0-9a-f]{16} ' "$O/load-cold.txt" | cut -d' ' -f5-); TW=$(grep -aE '^  token ids: [0-9a-f]{16} ' "$O/load-warm.txt" | cut -d' ' -f5-)
[ -n "$TC" ] && [ "$TC" = "$TW" ] \
  && echo "control 4: the same token ids cold and warm" | tee -a "$O/summary.txt" \
  || echo "VOID: control 4, the token ids differ cold and warm (or are missing)" | tee -a "$O/summary.txt"
for f in door-ch ctx-ch ctx-v9 gsm8k-ch gsm8k-v9; do
  grep -qF "4E: persistent module 252757 bytes, sha256 $P4E_V9_SHA, row form 7," "$O/$f.txt" \
    && echo "control 2 ($f): the served 4E v9 module" | tee -a "$O/summary.txt" \
    || echo "VOID: control 2 ($f), the 4E module is not the served v9 module" | tee -a "$O/summary.txt"
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
  --name "chain-prefill-served${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
