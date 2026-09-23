#!/usr/bin/env bash
# The embedding pays for int4 tables, 4B. Preregistered:
#   proofs/preregistration-embed-q4-swap-2026-09-23.md, stamped (ots stamp) before this.
# Constant codes, 0.43 pp bar, paired with the 61.11 census dump.
# Signed prediction: +1.2 pp [+0.2, +2.2].
#
# No image republication. The Space holds af907416 (built from b68b590, the 14B
# chain), whose `sealed.rs`, `bin/mmlu.rs`, `llvq-artifact` and `embedquant.rs`
# are identical to this branch's; it only lacks `embedq`. Republishing from this
# branch would also drop the 14B chain's `export` and `rowscale` from the image
# and cross the 14B base/FT pair over two images. So `embedq` runs on the Mac
# (6 s, `upload` below) and the job scores its output, checking the sha256.
#
# The pair crosses images all the same: 61.11 was scored on the 2026-09-19 image,
# before the 2026-09-20 model.rs refactors. Hence the harness stage: the 61.11
# file itself, unchanged, scored at limit=40 here, joined against the census dump
# on subject, index and qhash (census-14b-base-2026-09-22.txt control 4, same form).
#
# Cost, estimated: the arm is 6aae8c53's shape (restore int4, census), 24 min;
# the harness ~5 min (4B dense limit=40 ran 4 min 22 s); ~30 min, ~$0.90 on
# l40sx1; timeout 60m, ceiling $1.80.
#
#   bash ops/jobs/embed-q4-swap.sh upload    copy the Mac-side output into the bucket
#   DRY_RUN=1 bash ops/jobs/embed-q4-swap.sh prints the job script, parses it, launches nothing
#   bash ops/jobs/embed-q4-swap.sh           launches
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-embed-q4-swap-2026-09-23.md
IMAGE_SHA=${IMAGE_SHA:-af90741654a5b28c51f6993c24c94993813220bc}
BUCKET=Pier-Jean/jobs-artifacts
OBJ_DIR=embed-q4-2026-09-23                          # the input, written from the Mac
LOCAL=${LOCAL:-$HOME/embed-q4-swap-2026-09-23/qwen3-4b-dclm-ft-e4.bin}
SHA=9679cae4d781dcabf6cd2b7e4aebd161ed91c9d76fe4ea9b60e0cefcdb31dafa
BYTES=1235440301
O=/out/embed-q4-swap-2026-09-23                      # the outputs
E=/out/$OBJ_DIR/qwen3-4b-dclm-ft-e4.bin
F=/out/dclm-ft-2026-09-19/qwen3-4b-dclm-ft.bin       # the 61.11 object, embedq's input
F_SHA=f8c1c903b753fe3427f395f53b79e1e4ab59d212a6bc1c4c5ceedf8aea971300
F_BYTES=1794564765
FP=a74a6d6213602979
FP40=65dcd53655e8bfa5
DRY=${DRY_RUN:-0}

local_ok() {
  [ "$(stat -f %z "$LOCAL")" = "$BYTES" ] || { echo "refused: $LOCAL is not $BYTES B" >&2; exit 1; }
  [ "$(shasum -a 256 "$LOCAL" | cut -d' ' -f1)" = "$SHA" ] || { echo "refused: $LOCAL sha256 differs" >&2; exit 1; }
}

if [ "${1:-}" = upload ]; then
  local_ok
  hf buckets cp "$LOCAL" "hf://buckets/$BUCKET/$OBJ_DIR/qwen3-4b-dclm-ft-e4.bin"
  hf buckets ls "hf://buckets/$BUCKET/$OBJ_DIR/"
  exit 0
fi

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  local_ok
  REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/$OBJ_DIR/" 2>/dev/null | awk '$NF ~ /qwen3-4b-dclm-ft-e4\.bin$/ {print $1}' || true)
  [ "$REMOTE" = "$BYTES" ] || { echo "refused: bucket copy is ${REMOTE:-absent} B, local $BYTES B" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space moved to $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "q4 object: $BYTES B, sha256 $SHA; image $NOW"
fi

PRE=$(printf 'O=%q\nE=%q\nSHA=%q\nBYTES=%q\nF=%q\nF_SHA=%q\nF_BYTES=%q\nFP=%q\nFP40=%q' \
  "$O" "$E" "$SHA" "$BYTES" "$F" "$F_SHA" "$F_BYTES" "$FP" "$FP40")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt" || true
echo '== oracle (hard rule 10) =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle-cuda.txt" | tail -2
grep -q MATCH "$O/oracle-cuda.txt"
echo '== both files on the mount: bytes and sha256 against the Mac =='
test "$(stat -c %s "$E")" = "$BYTES"
test "$(stat -c %s "$F")" = "$F_BYTES"
sha256sum "$E" "$F" | tee "$O/files.sha256"
test "$(sed -n 1p "$O/files.sha256" | cut -d' ' -f1)" = "$SHA"
test "$(sed -n 2p "$O/files.sha256" | cut -d' ' -f1)" = "$F_SHA"
echo '== arm: q4 embedding + o_proj + down_proj@12-23 at int4, dense reconstruction, FULL split ==' ; date -u
LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_RESTORE_Q4=o_proj,down_proj@12-23 \
LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP="$O/mmlu-4b-embed-q4-swap-FULL.csv" \
  mmlu "$E" cuda 2>&1 | tee "$O/out.txt" | tail -10
date -u
D="$O/mmlu-4b-embed-q4-swap-FULL.csv"
grep -qxF '# dtype=f16' "$D"
grep -qxF '# limit=census' "$D"
grep -qxF '# alloc=flat, 100..1534 per subject, 14042 questions' "$D"
grep -qxF '# config=none' "$D"
grep -qxF '# arithmetic=dense reconstruction' "$D"
grep -qxF '# kv=f16' "$D"
test "$(tail -n 1 "$D")" = "# end fingerprint=$FP questions=14042"
grep -qF '(48 matrices, 676331520 weights)' "$O/out.txt"
echo "dump ok: $D"
echo '== harness control: the 61.11 file, unchanged, at limit=40 on this image ==' ; date -u
LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP="$O/mmlu-4b-dclm-ft-l40-harness.csv" \
  mmlu "$F" cuda 40 2>&1 | tee "$O/out-harness.txt" | tail -6
grep -qxF '# arithmetic=dense reconstruction' "$O/mmlu-4b-dclm-ft-l40-harness.csv"
test "$(tail -n 1 "$O/mmlu-4b-dclm-ft-l40-harness.csv")" = "# end fingerprint=$FP40 questions=2280"
date -u
echo '== dumps ==' ; wc -l "$O"/*.csv ; ls -la "$O"
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
  --flavor l40sx1 --timeout 60m \
  --bucket "$BUCKET" --out-mount /out \
  --name embed-q4-swap \
  "$PRE" "$BODY"
