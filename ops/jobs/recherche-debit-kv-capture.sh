#!/usr/bin/env bash
# Decode-speed research phase at long context (CERVEAU O13), step R2: the falsifier's data. Three captures of what a
# decode row of the sealed 4B reads from the KV cache after 8,192 tokens (`LLVQ_KV_CAPTURE`, llvq-llm/src/kvcapture.rs):
# window 0 of wikitext2-test, c4 and dclm-edu, 8,192 tokens fed one at a time on LLVQ_GQA_PPL's kernel0 attention
# (grow(256), the kernel form on every decode row), then 256 tokens whose queries are kept; the final K, the V row
# norms and one self-check pair are written. Scored offline at $0 by ops/kv_mass_score.py.
# Each capture finishes and is listed before the next starts, so a timeout leaves what is done.
# Preregistered: proofs/preregistration-recherche-debit-kv-capture-2026-11-14.md, to stamp before this.
#
#   DRY_RUN=1 bash ops/jobs/recherche-debit-kv-capture.sh                 parses, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/recherche-debit-kv-capture.sh     launches, after the operator's go
#
# Cost (l40sx1 $1.80/h), estimated from measured times:
# - the feed: LLVQ_GQA_PPL 8192,6,6,kernel (attn-t8-r2-2026-11-04-brut/ppl-8192.txt) fed 6 windows of 8,192 tokens
#   one at a time through TWO arms in lockstep in 940 s: 156.7 s a window, 78 s an arm, 9.6 ms a token with the
#   head's log-softmax read on the host every step. One arm over 8,448 tokens: about 81 s. The capture reads the
#   head on 255 tokens only; that is not counted as a saving.
# - the load: 4.3 s, the kernel check 1.1 s (same log); the sha256 of the 1.42 GB file, a few seconds.
# - the corpora: wikitext2-test and the C4 evaluation shard are tens of MB; the dclm-edu shard is 2.91 GB downloaded
#   whatever is read (corpus.rs), 30 to 60 s assumed (not measured here).
# - the writes: about 725 MB a capture to the bucket mount (*computed*, below); 15 to 60 s assumed (not measured).
# - the oracle: under 4 min (its timeout), about 1 min in earlier jobs.
# A capture: 81 + 10 + 60 + 60 = about 3.5 min at worst; three, plus the oracle: about 12 min, $0.36. Each capture
# has its own 8 min timeout. Job timeout 30m: $0.90 at worst.
# Bytes to the bucket a capture (*computed*, 4B, L = 8,448): q 256 x 36 x 32 x 128 x 2 = 75.5 MB; K 36 x 8 x 8,448 x
# 128 x 2 = 622.9 MB; V norms 36 x 8 x 8,448 x 4 = 9.7 MB; V of the check layer 17.3 MB; about 725 MB. Three: 2.18 GB.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-recherche-debit-kv-capture-2026-11-14.md
OUT_NAME=recherche-debit-kv-capture-2026-11-14
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/$OUT_NAME${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
TIMEOUT=${TIMEOUT:-30m}
CTX=8192
NTOK=256
WINDOW=0
CORPORA="wikitext2-test c4 dclm-edu"
# LLVQ_GQA_PPL's WikiText-2 revision (ops/jobs/attn-t8-r2.sh). LLVQ_DATASET_REV names one revision for every dataset
# a process reads, so it is set for the wikitext capture only; c4 and dclm-edu read main, and the window's token ids
# and fingerprint go into meta.json and the log.
WIKI_REV=740312add88f781978c0658806c59bc2815b9866

SEALED="sealed-4b-2026-09-23/qwen3-4b-sealed.bin:1418224685"
SEALED_SHA=886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  grep -q '^TO WRITE$' "$PREREG" && { echo "refused: $PREREG still holds TO WRITE" >&2; exit 1; }
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

PRE=$(printf 'O=%q\nSEALED=%q\nSEALED_SHA=%q\nCTX=%q\nNTOK=%q\nWINDOW=%q\nCORPORA=%q\nWIKI_REV=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$SEALED" "$SEALED_SHA" "$CTX" "$NTOK" "$WINDOW" "$CORPORA" "$WIKI_REV")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
df -h /tmp | tee "$O/disk.txt"
echo '== control 1: oracle =='
timeout 4m oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle.txt" | tail -2
grep -q MATCH "$O/oracle.txt"
echo '== control 1: the object =='
S=/out/${SEALED%%:*}; SB=${SEALED##*:}
test "$(stat -c %s "$S")" = "$SB"
sha256sum "$S" | tee "$O/files.sha256"
test "$(sha256sum "$S" | cut -d' ' -f1)" = "$SEALED_SHA"
FLAGS=(LLVQ_FUSED_LAYOUT=tetra48 LLVQ_EMBED=q4 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_KV=f16)
set +e
for corpus in $CORPORA; do
  echo "== capture: $corpus window $WINDOW, $CTX + $NTOK tokens =="
  date -u
  REV=()
  [ "$corpus" = wikitext2-test ] && REV=(LLVQ_DATASET_REV="$WIKI_REV")
  env "${FLAGS[@]}" "${REV[@]}" LLVQ_KV_CAPTURE="$CTX,$NTOK,$corpus,$WINDOW" timeout 8m \
    fusedrun "$S" 0 "$O/$corpus" > "$O/capture-$corpus.txt" 2>&1
  rc=$?
  echo "fusedrun capture $corpus exit $rc" | tee -a "$O/capture-$corpus.txt"
  date -u
  grep -aE '^sealed file|^loaded in|^attn_dec|^corpus |^attention |^feeding |^  fed |^captured |^KV CAPTURE WRITTEN|^Error|^error' \
    "$O/capture-$corpus.txt"
  if [ "$rc" = 0 ]; then
    grep -aqF "sha256 $SEALED_SHA" "$O/capture-$corpus.txt" || echo "VOID: $corpus, the capture did not read the sealed file's sha256"
    grep -aqF 'grow(256)+kernel, every decode row on the kernel from position 0' "$O/capture-$corpus.txt" \
      || echo "VOID: $corpus, the capture did not run the kernel0 attention"
    n=$(ls "$O/$corpus" | grep -cE '^(k|vnorm)-[0-9]+\.bin$')
    [ "$n" = 72 ] && echo "$corpus: 72 K and V-norm files" || echo "VOID: $corpus holds $n K and V-norm files, expected 72"
    for f in q.bin v-check.bin out-check.bin meta.json; do test -s "$O/$corpus/$f" || echo "VOID: $corpus/$f missing"; done
    du -sb "$O/$corpus" | tee -a "$O/bytes.txt"
  fi
  ls -la "$O/$corpus" 2>/dev/null | head -5
done
set -e
date -u
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
  --name "recherche-debit-kv-capture${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
