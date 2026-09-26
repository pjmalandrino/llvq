#!/usr/bin/env bash
# The MMLU census of the row-scale-trained 14B, one arm, full split, dense
# reconstruction: the arm that pairs against arm A of census-14b-base.sh. Then the
# f16 perplexity of the trained file and of the base, both on this card.
# ops/jobs/dclm-8b-ft-mmlu.sh at 14B, plus the perplexity the 8B read on the Mac.
#
# Preregistered under the training prereg, as at 4B and 8B:
# proofs/preregistration-dclm-14b-rowscales-2026-09-22.md, stamped before the
# training job. Its signed prediction is the PAIRED delta against the base census.
#
# Why the perplexity moved to the card: at 8B it ran on Metal (ppl-ft-f16.txt). At
# 14B the Mac is not to be loaded, and the sealed f16 load is >= ~52 GB of Metal
# buffers against a 55.7 GB working set (*computed*, the costing's check). On the
# card it is ~2 min a file (the August 14B arms read 12 windows in 21-28 s). The
# base is read again here, beside the trained file, so that the pair shares a card,
# an image and a process order; the encode job's reading of the base was on
# another card (rtx-pro-6000) and does not pair with this one. The f16 reference
# stays the August L40S reading, 7.9820 (campagne-14b-qualite-2026-08-10.txt:143),
# as the 8B's 8.9899 came from its August campaign. LLVQ_DTYPE=f16 is required:
# ppl defaults to f32, and a 14B dense reconstruction in f32 is 59 GB on a 46 GB card.
#
# Same image as arm A, or the pair crosses images: the launcher refuses when the
# Space sha differs from IMAGE_SHA, which defaults to the sha census-14b-base.sh
# recorded in $LJ/image.sha. Nobody runs `ops/run.py publish --cuda` between the
# two jobs.
#
# Inputs, from fold-14b.sh (Mac): $LJ/qwen3-14b-dclm-ft.bin, uploaded to
# dclm-14b-ft-<OBJ_DATE>/, and $LJ/ft.sha256, which holds its sha256 and the base's
# (the base checked there against the encode job's line). $LJ = ~/q14b-dclm-<OBJ_DATE>.
#
# Cost, *estimated* (l40sx1 $1.80/h):
#   pull 2.5 + oracle 0.5 + sha256 of the FT file over the mount 2.3        =  5 min
#   FT census: the 8B's 1,786 s of scoring x 1.58 (the 14B/8B sealed-arm
#     sample ratio) = 2,822 s, + load 1.5                                    = 49 min
#   sha256 of the base 2.3 + two perplexities ~2 each                         =  6 min
#   ~60 min, $1.80, range [54, 70] min = [$1.62, $2.10]; timeout 80m, ceiling $2.40.
#   The census runs first: a timeout can only cut the perplexities.
#
# DRY_RUN=1 bash ops/jobs/dclm-14b-ft-mmlu.sh   prints the job script, parses it, launches nothing.
set -euo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
cd "$REPO"

OBJ_DATE=${OBJ_DATE:-2026-09-22}
D=${CENSUS_DATE:-$(date -u +%F)}                     # launch date, names the output dir
PREREG=${PREREG:-proofs/preregistration-dclm-14b-rowscales-2026-09-22.md}
LJ=${LJ:-$HOME/q14b-dclm-$OBJ_DATE}
IMAGE_SHA=${IMAGE_SHA:-$(cat "$LJ/image.sha" 2>/dev/null || true)}
BUCKET=Pier-Jean/jobs-artifacts
FT_DIR=${FT_DIR:-dclm-14b-ft-$OBJ_DATE}              # fold-14b.sh upload
BASE_DIR=${BASE_DIR:-dclm-14b-$OBJ_DATE}             # the encode job's
FT_NAME=qwen3-14b-dclm-ft.bin
BASE_NAME=qwen3-14b-dclm.bin
LOCAL=${FT_LOCAL:-$LJ/$FT_NAME}
BASE_LOCAL=${BASE_LOCAL:-$LJ/$BASE_NAME}
SUMS=${SUMS:-$LJ/ft.sha256}                          # fold-14b.sh writes both lines
O=/out/census-14b-ft-$D
F=/out/$FT_DIR/$FT_NAME
BF=/out/$BASE_DIR/$BASE_NAME
FP=a74a6d6213602979
DRY=${DRY_RUN:-0}

sum_of() {  # $1 file name: the one sha256 $SUMS gives it, or nothing
  awk -v n="/$1" '{p = "/" $2} substr(p, length(p) - length(n) + 1) == n {print $1}' "$SUMS" | tail -1
}

if [ "$DRY" = 1 ]; then
  SHA=0000000000000000000000000000000000000000000000000000000000000000
  BASE_SHA=1111111111111111111111111111111111111111111111111111111111111111
  BYTES=6563782117
else
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  test -s "$SUMS" || { echo "refused: $SUMS missing, the fold did not finish" >&2; exit 1; }
  SHA=$(sum_of "$FT_NAME"); BASE_SHA=$(sum_of "$BASE_NAME")
  [ ${#SHA} -eq 64 ] && [ ${#BASE_SHA} -eq 64 ] || { echo "refused: $SUMS lacks a sha256 for $FT_NAME or $BASE_NAME" >&2; exit 1; }
  [ "$SHA" = "$(shasum -a 256 "$LOCAL" | cut -d' ' -f1)" ] || { echo "refused: $LOCAL differs from $SUMS" >&2; exit 1; }
  BYTES=$(stat -f %z "$LOCAL")
  [ "$BYTES" = "$(stat -f %z "$BASE_LOCAL")" ] || { echo "refused: FT and base sizes differ" >&2; exit 1; }
  REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/$FT_DIR/" 2>/dev/null | awk -v n="/$FT_NAME" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
  [ "$REMOTE" = "$BYTES" ] || { echo "refused: bucket FT copy is ${REMOTE:-absent} B, local $BYTES B" >&2; exit 1; }
  RB=$(hf buckets ls "hf://buckets/$BUCKET/$BASE_DIR/" 2>/dev/null | awk -v n="/$BASE_NAME" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
  [ "$RB" = "$BYTES" ] || { echo "refused: bucket base copy is ${RB:-absent} B, want $BYTES B" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/census-14b-ft-$D/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: census-14b-ft-$D/ already exists" >&2; exit 1
  fi
  [ ${#IMAGE_SHA} -eq 40 ] || { echo "refused: no IMAGE_SHA, and $LJ/image.sha is absent (census-14b-base.sh writes it)" >&2; exit 1; }
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space moved to $NOW; the base census ran on $IMAGE_SHA" >&2; exit 1; }
  echo "FT object: $BYTES B, sha256 $SHA; base sha256 $BASE_SHA; image $NOW"
fi

PRE=$(printf 'O=%q\nF=%q\nBF=%q\nSHA=%q\nBASE_SHA=%q\nBYTES=%q\nFP=%q' "$O" "$F" "$BF" "$SHA" "$BASE_SHA" "$BYTES" "$FP")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
check() {
  grep -qxF '# dtype=f16' "$1"
  grep -qxF '# limit=census' "$1"
  grep -qxF '# alloc=flat, 100..1534 per subject, 14042 questions' "$1"
  grep -qxF '# config=none' "$1"
  grep -qxF '# arithmetic=dense reconstruction' "$1"
  grep -qxF '# kv=f16' "$1"
  test "$(tail -n 1 "$1")" = "# end fingerprint=$FP questions=14042"
  echo "dump ok: $1"
}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt" || true
echo '== oracle (hard rule 10) =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle-cuda.txt" | tail -2
grep -q MATCH "$O/oracle-cuda.txt"
echo '== the FT file on the mount: bytes and sha256 against the Mac =='
test "$(stat -c %s "$F")" = "$BYTES"
sha256sum "$F" | tee "$O/files.sha256"
test "$(cut -d' ' -f1 "$O/files.sha256")" = "$SHA"
echo '== arm FT: DCLM base + trained row scales, sealed, dense reconstruction, FULL split ==' ; date -u
LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP="$O/mmlu-14b-dclm-ft-FULL.csv" mmlu "$F" cuda 2>&1 | tee "$O/out-dclm-ft.txt" | tail -10
date -u
check "$O/mmlu-14b-dclm-ft-FULL.csv"
echo '== the base on the mount, for the perplexity pair: bytes and sha256 =='
test "$(stat -c %s "$BF")" = "$BYTES"
sha256sum "$BF" | tee -a "$O/files.sha256"
test "$(awk 'NR==2 {print $1}' "$O/files.sha256")" = "$BASE_SHA"
for arm in ft base; do
  if [ "$arm" = ft ]; then M=$F; else M=$BF; fi
  echo "== perplexity f16, $arm: wikitext2, ctx 4096, 12 windows ==" ; date -u
  LLVQ_DTYPE=f16 ppl 4096 12 cuda "$M" 2>&1 | tee "$O/ppl-$arm-f16.txt" | tail -3
  grep -qF 'dtype f16, kv f16, tokens 3f1baca9033bf251' "$O/ppl-$arm-f16.txt"
  grep '^ppl = ' "$O/ppl-$arm-f16.txt"
done
date -u
echo '== raw kept ==' ; wc -l "$O"/mmlu-14b-*-FULL.csv ; ls -la "$O"
JOB

if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; nothing launched" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout 80m \
  --bucket "$BUCKET" --out-mount /out \
  --name dclm-14b-ft-mmlu \
  "$PRE" "$BODY"
