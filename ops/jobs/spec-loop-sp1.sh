#!/usr/bin/env bash
# Job SP1 of the speculative loop (prereg proofs/preregistration-spec-loop-sp1-2026-11-10.md): on one L40S, the
# sealed 4B under its served v9 config:
#   `vtreebench --res a12` (compile only, its own 4m timeout, non-fatal), then one
#   `specbench <sealed> a12@8.10.8.20.6 null,oracle:1,oracle:3,oracle:7,oracle:14 64` process: on four prompts
#   (chat, rag, code, math), 64 tokens each, the loop's tokens against 4E's greedy tokens, accepted tokens a pass, the
#   walls of the verify launches, the drafter calls and the rest of the host, tok/s end to end against 4E's greedy
#   tok/s, and for oracle:K the check greedy x (K+1) / c_(K+2) with c_R timed in the same process.
# Stamped before this: the prereg above and its .ots. Controls as the verify-tree jobs: oracle, the object and its
# config at their sha256, LLVQ_* refused in the job env, the served 4E v9 module.
#
#   DRY_RUN=1 bash ops/jobs/spec-loop-sp1.sh             parses the job script, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/spec-loop-sp1.sh  launches
#   FLAVOR=l40sx4 RETRY=1 TIMEOUT=24m ...                 the twin on l40sx4, one device visible
#
# Cost, estimated (l40sx1 $1.80/h, prereg "Cost"): about 16 min, $0.48; timeout 28m, $0.84.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-spec-loop-sp1-2026-11-10.md
OUT_NAME=spec-loop-sp1-2026-11-10
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/$OUT_NAME${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
P4E_V9_SHA=1b422b62328be6752df4add43280565fd2a49cfa20ee26953b301abe7a15c530
TIMEOUT=${TIMEOUT:-28m}
INNER_TIMEOUT=22m
ARM='a12@8.10.8.20.6'
DRAFTERS='null,oracle:1,oracle:3,oracle:7,oracle:14'
NTOK=64
FLAVOR=${FLAVOR:-l40sx1}
case "$FLAVOR" in
  l40sx1) FLAVOR_ARGS=(); DEV_PRE='' ;;
  l40sx4) FLAVOR_ARGS=(--any-flavor); DEV_PRE=$'\nexport CUDA_VISIBLE_DEVICES=0' ;;
  *) echo "refused: FLAVOR=$FLAVOR (l40sx1 or l40sx4)" >&2; exit 1 ;;
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

PRE=$(printf 'O=%q\nSEALED=%q\nSEALED_SHA=%q\nCFG_SHA=%q\nCFG_B64=%q\nP4E_V9_SHA=%q\nARM=%q\nDRAFTERS=%q\nNTOK=%q\nINNER_TIMEOUT=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$SEALED" "$SEALED_SHA" "$CFG_SHA" "$CFG_B64" "$P4E_V9_SHA" "$ARM" "$DRAFTERS" "$NTOK" "$INNER_TIMEOUT")$DEV_PRE

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
echo '== vtreebench --res a12 (compile only, non-fatal) =='
set +e
timeout 4m vtreebench --res a12 > "$O/res.txt" 2>&1
echo "vtreebench --res exit $?" >> "$O/res.txt"
set -e
tail -8 "$O/res.txt"
echo '== specbench =='
date -u
set +e
LLVQ_CONFIG="$PT" timeout "$INNER_TIMEOUT" specbench "$S" "$ARM" "$DRAFTERS" "$NTOK" > "$O/spec-4b.txt" 2>&1
rc=$?
set -e
date -u
echo "specbench exit $rc" | tee "$O/summary.txt"
sed 's/^/res: /' "$O/res.txt" | tee -a "$O/summary.txt"
grep -E '^4E: persistent module|^4E tree|^arm |^loaded|^== |^  |^summary' "$O/spec-4b.txt" | tee -a "$O/summary.txt" || true
tail -3 "$O/spec-4b.txt"
# control 2: 4E is the served v9 module
grep -qF "4E: persistent module 252757 bytes, sha256 $P4E_V9_SHA, row form 7," "$O/spec-4b.txt" \
  && echo "control 2: the served 4E v9 module" | tee -a "$O/summary.txt" \
  || echo "VOID: control 2, the 4E module is not the served v9 module" | tee -a "$O/summary.txt"
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
  --name "spec-loop-sp1${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
