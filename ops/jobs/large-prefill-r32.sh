#!/usr/bin/env bash
# Design A of docs/mesures/large-prefill-rangees-etude-2026-11-11.txt (arbre-nuit, 2881ea3; prereg to come,
# proofs/preregistration-large-prefill-r32-2026-11-11.md): on one L40S, the sealed 4B under its served v9 config, the
# prompt prefilled as chains of known tokens through three arms in one prefillbench process, the same words read in
# place (no copy): a12ip at 16 rows a pass (the reference of record, the first job's arm), a12ipg at 16 (D1's arm) and
# a12ipg32 at 32 (a12ipg's pass at RC = 4, pers_4et_a12g32.cu), then 4E. Under two samplers (nvidia-smi -lms 100 for
# the process peak, and an epoch-stamped one for the peak inside each phase the binary marks, one tag an arm and a
# length): timing at 1,024 / 4,096 / 8,191 tokens round by round (a12ip, a12ipg, a12ipg32 in each round), the identity
# of a12ipg and of a12ipg32 against a12ip at 16 at the three lengths (argmax, last row's logits, every layer's K and
# V), each arm's 8k phase split (the last three passes, stamped), and the 47 texts through the three arms, each paired
# with a12ip's. No eager arm and no perplexity: both arms are held bit for bit to a12ip, which the first job
# (large-prefill-2026-11-11) gated against the eager prefill and scored on WikiText-2.
# l40sx1 only: no twin.
#
#   DRY_RUN=1 bash ops/jobs/large-prefill-r32.sh               parses the job script, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/large-prefill-r32.sh   launches
#   WITH_G16=1 ...                                             adds a12ipg at 16 (D1's arm) as a third arm
#
# Tree modules through NVRTC at load: 4E tree (ce1c354d), a12ip (63155df9), a12ipg32 (a35f2b83); a12ipg (246e99b1)
# too with WITH_G16=1. Load estimated from the first job (419.5 s with the first two) plus a12ipg32 at about 370 s
# (its Mac nvcc compile 177 s against 138 for a12ip and 158 for a12ipg, whose card loads took about 254 and 339 s).
# Cost, estimated (l40sx1 $1.80/h): about 21 min, $0.63, two arms; about 29 min with WITH_G16=1, which then needs
# TIMEOUT=33m and a cap of $0.99, over the $0.90 this job is held to. Timeout 30m, $0.90.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-large-prefill-r32-2026-11-11.md
OUT_NAME=large-prefill-r32-2026-11-11
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/$OUT_NAME${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
P4E_V9_SHA=1b422b62328be6752df4add43280565fd2a49cfa20ee26953b301abe7a15c530
TIMEOUT=${TIMEOUT:-30m}
# the new arm at --r 32; the reference (the first --against) and a12ipg at 16 by their /16 suffix
ARM='a12ipg32@8.10.8.20.6'
AGAINST='a12ip@8.10.8.20.6/16'
WITH_G16=${WITH_G16:-0}
case "$WITH_G16" in 1) G16='a12ipg@8.10.8.20.6/16' ;; 0) G16='' ;; *) echo "refused: WITH_G16=$WITH_G16 (0 or 1)" >&2; exit 1 ;; esac
A12IP_SHA=63155df95ef832aabbff328bd92062b6981763ed57358e32dc118d150df7c6ad
A12IPG_SHA=246e99b18895a7f9452411372b8ff4bbb686276408318e547656fb809b722da8
TREE_SHA=ce1c354d2583eb178ae8622ff99e10f9b457543972bb5039ce89ede9eec31e44
# the a12ipg32 module at the card's tile define, as pers.rs builds it on the Mac (dump_tree_modules)
A12IPG32_SHA=a35f2b83061dc8a22ee65e6805c9e73454a93ec773ec919a14afc80fcf60aa8f
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

PRE=$(printf 'O=%q\nSEALED=%q\nSEALED_SHA=%q\nCFG_SHA=%q\nCFG_B64=%q\nP4E_V9_SHA=%q\nARM=%q\nAGAINST=%q\nG16=%q\nA12IP_SHA=%q\nA12IPG_SHA=%q\nTREE_SHA=%q\nA12IPG32_SHA=%q\nNTOK=%q\nPROMPTS=%q\nPROMPTS_SHA=%q\nLONG=%q\nLONG_SHA=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$SEALED" "$SEALED_SHA" "$CFG_SHA" "$CFG_B64" "$P4E_V9_SHA" "$ARM" "$AGAINST" "$G16" "$A12IP_SHA" "$A12IPG_SHA" "$TREE_SHA" "$A12IPG32_SHA" "$NTOK" "$PROMPTS" "$PROMPTS_SHA" "$LONG" "$LONG_SHA")

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
echo "== chain prefill, $ARM at 32 against $AGAINST${G16:+ and $G16}, one process =="
date -u
EXTRA=()
[ -n "$G16" ] && EXTRA=(--against "$G16" --dump-against "$O/texts-a12ipg.jsonl")
vram r32 env LLVQ_CONFIG="$PT" timeout 27m prefillbench --form chain --arm "$ARM" --r 32 --against "$AGAINST" \
  --dump-against "$O/texts-a12ip.jsonl" "${EXTRA[@]}" --long "$LONG" --prompts "$PROMPTS" --new "$NTOK" \
  --dump "$O/texts-a12ipg32.jsonl" "$S" > "$O/r32.txt" 2>&1
echo "prefillbench r32 exit $?" >> "$O/r32.txt"
date -u
phases r32 > "$O/phase-peaks.txt" 2>&1
set -e
{
  cat "$O/vram.txt"
  cat "$O/phase-peaks.txt"
  grep -vE '^(mark |NVRTC|cudarc|tv_|attn_|batch|int4|embedding|shared rotation|projection fusion|fused layout|total expected)' "$O/r32.txt" | sed "s/^/r32: /" || true
} > "$O/summary.txt"
tail -150 "$O/summary.txt"
for f in r32; do
  grep -qF "4E: persistent module 252757 bytes, sha256 $P4E_V9_SHA, row form 7," "$O/$f.txt" \
    && echo "control 2 ($f): the served 4E v9 module" | tee -a "$O/summary.txt" \
    || echo "VOID: control 2 ($f), the 4E module is not the served v9 module" | tee -a "$O/summary.txt"
  # control 3: the tree modules of record keep D1's bytes
  for pair in "4E tree: module 306060 bytes, sha256 $TREE_SHA" "4E tree a12ip: module 422018 bytes, sha256 $A12IP_SHA" \
              "4E tree a12ipg32: module 451917 bytes, sha256 $A12IPG32_SHA" \
              ${G16:+"4E tree a12ipg: module 450134 bytes, sha256 $A12IPG_SHA"}; do
    grep -qF "$pair" "$O/$f.txt" \
      && echo "control 3 ($f): $pair" | tee -a "$O/summary.txt" \
      || echo "VOID: control 3 ($f), not found: $pair" | tee -a "$O/summary.txt"
  done
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
