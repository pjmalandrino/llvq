#!/usr/bin/env bash
# Job N2 of the tree verify (docs/plan-verify-tree.md, prereg proofs/preregistration-verify-tree-n2-2026-11-10.md):
# on one L40S, the sealed 4B under its served v9 config, one `vtreebench <sealed> $ARMS` process:
#   the 4E v9 install and the tree pass beside it with two arms on the same tree-tiled copies and the same plan
#   (pers::p4t_splits_stream): a2 the tiled kernel (N1's), a3 the stream unit (pers_4et_stream.cu), a2 a3's guard
#   (a3's 32-row pass equal to a2's bit for bit, else a3's numbers are void); each module's bytes, sha256,
#   registers, local and shared bytes;
#   chat context for every arm: vt1 against 4E's greedy tokens on 64 positions; identity of every row across
#   R = 1, 2, 8, 16 and two shuffled slot orders; c_R = t(vt R) / t(4e), 2 discarded and 5 timed rounds, arms
#   interleaved; the phases of vt1 and vt16; then the rag context for the arm with the lowest chat c_16.
# Stamped before this: the prereg above and its .ots.
#
#   DRY_RUN=1 bash ops/jobs/verify-tree-n2.sh             parses the job script, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/verify-tree-n2.sh  launches
#   STAGE=n2a ARMS=a2,a3 ...                               job N2a: its own prereg and output directory
#     (PREREG and OUT_NAME override the paths STAGE derives)
#   FLAVOR=l40sx4 RETRY=1 TIMEOUT=18m ...                  the twin on l40sx4, one device visible
#   QUEUE_BEFORE_PREREG=1 / QUEUE_BEFORE_IMAGE=1           queue the job before the prereg and the image exist
#     (the orchestrator's N2 submission only: the image that runs must carry arm a4, since an image without it
#     refuses the arm list a2,a3,a4 before any timed pass; the prereg's ECARTS or its own paragraph says so)
#
# Cost, estimated (l40sx1 $1.80/h): two arms where N1 ran three, about 10 min, $0.30; timeout 25m, $0.75.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

STAGE=${STAGE:-n2}
case "$STAGE" in n2|n2a|n2b) ;; *) echo "refused: STAGE=$STAGE (n2, n2a or n2b)" >&2; exit 1;; esac
PREREG=${PREREG:-proofs/preregistration-verify-tree-$STAGE-2026-11-10.md}
OUT_NAME=${OUT_NAME:-verify-tree-$STAGE-2026-11-10}
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/$OUT_NAME${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
P4E_V9_SHA=1b422b62328be6752df4add43280565fd2a49cfa20ee26953b301abe7a15c530
TIMEOUT=${TIMEOUT:-25m}
ARMS=${ARMS:-a2,a3,a4}
case "$ARMS" in a2,a3,a4|a2,a3) ;; *) echo "refused: ARMS=$ARMS (a2,a3,a4 or a2,a3: a2 is a3's guard, a3 is a4's)" >&2; exit 1;; esac
FLAVOR=${FLAVOR:-l40sx1}
case "$FLAVOR" in
  l40sx1) FLAVOR_ARGS=(); DEV_PRE='' ;;
  l40sx4) FLAVOR_ARGS=(--any-flavor); DEV_PRE=$'\nexport CUDA_VISIBLE_DEVICES=0' ;;
  *) echo "refused: FLAVOR=$FLAVOR (l40sx1 or l40sx4)" >&2; exit 1 ;;
esac
SKIP_PREREG=${QUEUE_BEFORE_PREREG:-0}
SKIP_IMAGE=${QUEUE_BEFORE_IMAGE:-0}
SEALED="sealed-4b-2026-09-23/qwen3-4b-sealed.bin:1418224685"
SEALED_SHA=886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8
C=configs/qwen3-4b-tetra-e4-4e-v9.json
grep -q '^  "rows": "v9",$' "$C" || { echo "refused: $C does not name rows v9" >&2; exit 1; }
[ "$(printf '%s\n' "$(cat "$C")" | shasum -a 256 | cut -d' ' -f1)" = "$(shasum -a 256 < "$C" | cut -d' ' -f1)" ] \
  || { echo "refused: $C does not end in exactly one newline" >&2; exit 1; }
CFG_SHA=$(shasum -a 256 < "$C" | cut -d' ' -f1)
CFG_B64=$(base64 < "$C" | tr -d '\n')

if [ "$DRY" != 1 ]; then
  if [ "$SKIP_PREREG" = 1 ]; then
    echo "!!!!! QUEUE_BEFORE_PREREG=1: the prereg/.ots refusal is SKIPPED for this submission ($(date)) !!!!!" >&2
  else
    { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  fi
  path=${SEALED%%:*}; bytes=${SEALED##*:}
  REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/${path%/*}/" 2>/dev/null \
    | awk -v n="/${path##*/}" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
  [ "$REMOTE" = "$bytes" ] || { echo "refused: $path is ${REMOTE:-absent} B in the bucket, expected $bytes" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  if [ "$SKIP_IMAGE" = 1 ]; then
    echo "!!!!! QUEUE_BEFORE_IMAGE=1: the Space sha refusal is SKIPPED; the Space is at $NOW now ($(date)) !!!!!" >&2
  else
    IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of the rebuild that carries this branch>}
    [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  fi
  echo "bucket copy verified by size; image $NOW"
fi

PRE=$(printf 'O=%q\nSEALED=%q\nSEALED_SHA=%q\nCFG_SHA=%q\nCFG_B64=%q\nP4E_V9_SHA=%q\nARMS=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$SEALED" "$SEALED_SHA" "$CFG_SHA" "$CFG_B64" "$P4E_V9_SHA" "$ARMS")$DEV_PRE

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
echo '== control 1: oracle =='
timeout 5m oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle.txt" | tail -2
grep -q MATCH "$O/oracle.txt"
echo '== control 1: the object and its config =='
S=/out/${SEALED%%:*}; SB=${SEALED##*:}
test "$(stat -c %s "$S")" = "$SB"
sha256sum "$S" | tee "$O/files.sha256"
test "$(sha256sum "$S" | cut -d' ' -f1)" = "$SEALED_SHA"
PT=$O/qwen3-4b-tetra-e4-4e-v9.json
printf '%s' "$CFG_B64" | base64 -d > "$PT"
test "$(sha256sum "$PT" | cut -d' ' -f1)" = "$CFG_SHA"
echo '== vtreebench =='
date -u
set +e
LLVQ_CONFIG="$PT" timeout 20m vtreebench "$S" "$ARMS" > "$O/vtree-4b.txt" 2>&1
rc=$?
set -e
date -u
echo "vtreebench exit $rc" | tee "$O/summary.txt"
grep -E '^4E: persistent module|^4E tree|^arm a[2-4]:|^== tree|^  (a[2-4] (vt1 against|identity|an extra row|vt(1|16) phases|guard)|a[2-4] against|4E, one|timed passes|a[2-4] +[0-9]+ +[0-9.]+ \[)|^  (chat|rag) a[2-4]:' "$O/vtree-4b.txt" | tee -a "$O/summary.txt"
tail -3 "$O/vtree-4b.txt"
# control 2: 4E is the served v9 module (its one-token launches are checked against its greedy tokens inside)
grep -qF "4E: persistent module 252757 bytes, sha256 $P4E_V9_SHA, row form 7," "$O/vtree-4b.txt" \
  && echo "control 2: the served 4E v9 module" | tee -a "$O/summary.txt" \
  || echo "VOID: control 2, the 4E module is not the served v9 module" | tee -a "$O/summary.txt"
grep -E '^4E tree[a-z0-9 ]*: pers_4et_[a-z0-9]+_f[0-9]_t512: [0-9]+ registers, [0-9]+ B local' "$O/vtree-4b.txt" | tee -a "$O/summary.txt" || true
grep -q '^  a3 guard against a2 .*: PASSES' "$O/vtree-4b.txt" \
  && echo "guard: a3 equals a2 bit for bit on the chat 32-row pass" | tee -a "$O/summary.txt" \
  || echo "VOID: a3's guard against a2 did not pass; a3's numbers are void" | tee -a "$O/summary.txt"
case "$ARMS" in *a4*) grep -q '^  a4 guard against a3 .*: PASSES' "$O/vtree-4b.txt" \
  && echo "guard: a4 equals a3 bit for bit on the chat 32-row pass" | tee -a "$O/summary.txt" \
  || echo "VOID: a4's guard against a3 did not pass; a4's numbers are void" | tee -a "$O/summary.txt" ;; esac
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
  --flavor "$FLAVOR" ${FLAVOR_ARGS[@]+"${FLAVOR_ARGS[@]}"} --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "verify-tree-$STAGE${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
