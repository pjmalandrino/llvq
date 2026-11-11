#!/usr/bin/env bash
# D1 of docs/mesures/large-prefill-contexte-2026-11-11.txt section 3 (prereg
# proofs/preregistration-large-prefill-d1-2026-11-11.md): on one L40S, the sealed 4B under its served v9 config, the
# prompt prefilled as chains of 16 known tokens through a12ipg (a12ip whose attention reads each split that ends at
# or below pos0 once for a group of 4 rows, pers_4et_a12g.cu) and through a12ip, then 4E. One prefillbench process
# with both arms installed (the same words read in place, no copy), under two samplers (nvidia-smi -lms 100 for the
# process peak, and an epoch-stamped one for the peak inside each phase the binary marks, one tag an arm and a
# length): timing at 1,024 / 4,096 / 8,191 tokens round by round (a12ip first in each round), the identity of the two
# arms at the three lengths (argmax, last row's logits, every layer's K and V), each arm's 8k phase split, and the 47
# texts through both arms, paired. No eager arm and no perplexity: a12ipg is held bit for bit to a12ip, which the
# first job (large-prefill-2026-11-11) gated against the eager prefill and scored on WikiText-2.
# l40sx1 only: no twin.
#
#   DRY_RUN=1 bash ops/jobs/large-prefill-d1.sh             parses the job script, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/large-prefill-d1.sh  launches
#
# Cost, estimated (l40sx1 $1.80/h): about 19 min, $0.57; timeout 29m, $0.87.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-large-prefill-d1-2026-11-11.md
OUT_NAME=large-prefill-d1-2026-11-11
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/$OUT_NAME${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
P4E_V9_SHA=1b422b62328be6752df4add43280565fd2a49cfa20ee26953b301abe7a15c530
TIMEOUT=${TIMEOUT:-29m}
ARM='a12ipg@8.10.8.20.6'
AGAINST='a12ip@8.10.8.20.6'
NTOK=256
PROMPTS=/usr/local/share/llvq/spec-prompts-v1.jsonl
PROMPTS_SHA=$(shasum -a 256 < ops/data/spec-prompts-v1.jsonl | cut -d' ' -f1)
[ "$PROMPTS_SHA" = f749eb359c2cf5c8a648ca99fc0f856e620f8adb3f1d233f4c477ff322257f1a ] \
  || { echo "refused: ops/data/spec-prompts-v1.jsonl is not at its pinned sha256" >&2; exit 1; }
LONG=/usr/local/share/llvq/spec-long-8k-v1.jsonl
LONG_SHA=$(shasum -a 256 < ops/data/spec-long-8k-v1.jsonl | cut -d' ' -f1)
[ "$LONG_SHA" = da4edbd6f12184cf8ff938f057500f3e30e35d8c7a351d2e925316d225d62ebc ] \
  || { echo "refused: ops/data/spec-long-8k-v1.jsonl is not at its pinned sha256" >&2; exit 1; }
FLAVOR=${FLAVOR:-l40sx1}
case "$FLAVOR" in
  l40sx1) ;;
  *) echo "refused: FLAVOR=$FLAVOR (this job runs on l40sx1 only, no twin)" >&2; exit 1 ;;
esac
SEALED="sealed-4b-2026-09-23/qwen3-4b-sealed.bin:1418224685"
SEALED_SHA=886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8
C=configs/qwen3-4b-tetra-e4-4e-v9.json
grep -q '^  "rows": "v9",$' "$C" || { echo "refused: $C does not name rows v9" >&2; exit 1; }
[ "$(printf '%s\n' "$(cat "$C")" | shasum -a 256 | cut -d' ' -f1)" = "$(shasum -a 256 < "$C" | cut -d' ' -f1)" ] \
  || { echo "refused: $C does not end in exactly one newline" >&2; exit 1; }
CFG_SHA=$(shasum -a 256 < "$C" | cut -d' ' -f1)
CFG_B64=$(base64 < "$C" | tr -d '\n')

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

PRE=$(printf 'O=%q\nSEALED=%q\nSEALED_SHA=%q\nCFG_SHA=%q\nCFG_B64=%q\nP4E_V9_SHA=%q\nARM=%q\nAGAINST=%q\nNTOK=%q\nPROMPTS=%q\nPROMPTS_SHA=%q\nLONG=%q\nLONG_SHA=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$SEALED" "$SEALED_SHA" "$CFG_SHA" "$CFG_B64" "$P4E_V9_SHA" "$ARM" "$AGAINST" "$NTOK" "$PROMPTS" "$PROMPTS_SHA" "$LONG" "$LONG_SHA")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
echo '== control 1: oracle =='
timeout 5m oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle.txt" | tail -2
grep -q MATCH "$O/oracle.txt"
echo '== control 1: the object, its config, the prompt sets =='
S=/out/${SEALED%%:*}; SB=${SEALED##*:}
test "$(stat -c %s "$S")" = "$SB"
sha256sum "$S" | tee "$O/files.sha256"
test "$(sha256sum "$S" | cut -d' ' -f1)" = "$SEALED_SHA"
PT=$O/qwen3-4b-tetra-e4-4e-v9.json
printf '%s' "$CFG_B64" | base64 -d > "$PT"
test "$(sha256sum "$PT" | cut -d' ' -f1)" = "$CFG_SHA"
for pair in "$PROMPTS:$PROMPTS_SHA" "$LONG:$LONG_SHA"; do
  f=${pair%:*}; h=${pair##*:}
  sha256sum "$f" | tee -a "$O/files.sha256"
  test "$(sha256sum "$f" | cut -d' ' -f1)" = "$h"
done
# Two samplers around one command: memory.used every 100 ms for the process peak (the duel's, ops/jobs/duel-iq2.sh),
# and one stamped with the epoch for the peak inside each phase the binary marks (`mark <tag> begin|end <unix s>`).
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
# The peak inside each marked phase of a run, over every window of its tag (a timing tag opens once a round),
# GB = MiB x 1,048,576 / 1e9.
phases() {
  awk 'NR == FNR { if ($1 == "mark") { t = $2; if (!(t in seen)) { seen[t] = 1; o[++n] = t; k[t] = 0 }
                     if ($3 == "begin") { k[t]++; b[t, k[t]] = $4 } else e[t, k[t]] = $4 } next }
       { for (i = 1; i <= n; i++) { t = o[i]; for (j = 1; j <= k[t]; j++)
           if (((t, j) in e) && $1 >= b[t, j] && $1 <= e[t, j] && $2 + 0 > p[t]) p[t] = $2 + 0 } }
       END { for (i = 1; i <= n; i++) { t = o[i]; printf "phase peak %s %s: %.3f GB (%d MiB) over %d window(s)\n", tg, t, p[t] * 1048576 / 1e9, p[t], k[t] } }' \
      tg="$1" "$O/$1.txt" "$O/vram-$1-t.txt"
}
set +e
echo "== chain prefill, $ARM against $AGAINST, one process =="
date -u
vram d1 env LLVQ_CONFIG="$PT" timeout 24m prefillbench --form chain --arm "$ARM" --against "$AGAINST" --r 16 --long "$LONG" \
  --prompts "$PROMPTS" --new "$NTOK" --dump "$O/texts-a12ipg.jsonl" --dump-against "$O/texts-a12ip.jsonl" "$S" > "$O/d1.txt" 2>&1
echo "prefillbench d1 exit $?" >> "$O/d1.txt"
date -u
phases d1 > "$O/phase-peaks.txt" 2>&1
set -e
{
  cat "$O/vram.txt"
  cat "$O/phase-peaks.txt"
  grep -vE '^(mark |NVRTC|cudarc|tv_|attn_|batch|int4|embedding|shared rotation|projection fusion|fused layout|total expected)' "$O/d1.txt" | sed "s/^/d1: /" || true
} > "$O/summary.txt"
tail -150 "$O/summary.txt"
for f in d1; do
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
  --flavor "$FLAVOR" --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "large-prefill" \
  "$PRE" "$BODY"
