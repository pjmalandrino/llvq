#!/usr/bin/env bash
# Job SP7 of the speculative loop (prereg proofs/preregistration-spec-loop-sp7-2026-11-11.md): on one L40S, a profile of
# the DFlash drafter, measured code b8a6fc3 (merges f3eb2df, `dflashcheck ... profile`). Measurement only: it changes
# nothing and tests no repair. Each step writes its own file and is non-fatal for the next:
#   (a) `dflashcheck <fixture> cuda <sealed> <drafter> f16 profile` (5m) into profile-f16.txt;
#   (b) the same in f32 (5m) into profile-f32.txt.
# Inputs as SP6 for the drafter and the sealed file: the fixture from the jobs bucket, the drafter at revision b74e3a32,
# both at their sha256. Controls as SP6 without the 4E module check (no specbench). l40sx1 only: no twin.
#
#   DRY_RUN=1 bash ops/jobs/spec-loop-sp7.sh             parses the job script, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/spec-loop-sp7.sh  launches
#
# Cost, estimated (l40sx1 $1.80/h, prereg "Cost"): about 5 min, $0.15; timeout 14m, $0.42.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-spec-loop-sp7-2026-11-11.md
OUT_NAME=spec-loop-sp7-2026-11-11
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/$OUT_NAME${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
TIMEOUT=${TIMEOUT:-14m}
FIXTURE="dflash-fixture-2026-11-10/llvq-dflash-fixture-2026-11-10.safetensors:8562156"
FIXTURE_SHA=636f664d92ae9bb6fd17c9a01d55aecb342c72dfc5e1571d6ba219e77fed67a8
DREPO=z-lab/Qwen3-4B-DFlash-b16
DREV=b74e3a329c4d963783143b1e970d95b002be72bd
DCFG_SHA=2de97cdf48f429628074d229d024753f1ea02b575732983d6ed2bc882bbd63c8
DMODEL_SHA=a21d0ffe5970c84e9a0710ccb12087d1a4bcbedf979e623476c92c9d005a9cd4
FLAVOR=${FLAVOR:-l40sx1}
case "$FLAVOR" in
  l40sx1) FLAVOR_ARGS=(); DEV_PRE='' ;;
  *) echo "refused: FLAVOR=$FLAVOR (SP7 runs on l40sx1 only, no twin)" >&2; exit 1 ;;
esac
SEALED="sealed-4b-2026-09-23/qwen3-4b-sealed.bin:1418224685"
SEALED_SHA=886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8
if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  path=${SEALED%%:*}; bytes=${SEALED##*:}
  REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/${path%/*}/" 2>/dev/null \
    | awk -v n="/${path##*/}" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
  [ "$REMOTE" = "$bytes" ] || { echo "refused: $path is ${REMOTE:-absent} B in the bucket, expected $bytes" >&2; exit 1; }
  fpath=${FIXTURE%%:*}; fbytes=${FIXTURE##*:}
  FREMOTE=$(hf buckets ls "hf://buckets/$BUCKET/${fpath%/*}/" 2>/dev/null \
    | awk -v n="/${fpath##*/}" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
  [ "$FREMOTE" = "$fbytes" ] || { echo "refused: $fpath is ${FREMOTE:-absent} B in the bucket, expected $fbytes" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of the rebuild that carries this branch>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "bucket copy verified by size; image $NOW"
fi

PRE=$(printf 'O=%q\nSEALED=%q\nSEALED_SHA=%q\nFIXTURE=%q\nFIXTURE_SHA=%q\nDREPO=%q\nDREV=%q\nDCFG_SHA=%q\nDMODEL_SHA=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$SEALED" "$SEALED_SHA" "$FIXTURE" "$FIXTURE_SHA" "$DREPO" "$DREV" "$DCFG_SHA" "$DMODEL_SHA")$DEV_PRE

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
echo '== control 1: oracle =='
timeout 5m oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle.txt" | tail -2
grep -q MATCH "$O/oracle.txt"
echo '== control 1: the object =='
S=/out/${SEALED%%:*}; SB=${SEALED##*:}
test "$(stat -c %s "$S")" = "$SB"
sha256sum "$S" | tee "$O/files.sha256"
test "$(sha256sum "$S" | cut -d' ' -f1)" = "$SEALED_SHA"
echo '== inputs: the fixture and the drafter =='
FX=/out/${FIXTURE%%:*}
test "$(stat -c %s "$FX")" = "${FIXTURE##*:}"
sha256sum "$FX" | tee -a "$O/files.sha256"
test "$(sha256sum "$FX" | cut -d' ' -f1)" = "$FIXTURE_SHA"
DD=/tmp/dflash-$DREV
mkdir -p "$DD"
for f in config.json model.safetensors; do
  curl -fsSL --retry 3 -o "$DD/$f" "https://huggingface.co/$DREPO/resolve/$DREV/$f"
done
sha256sum "$DD/config.json" "$DD/model.safetensors" | tee "$O/drafter.sha256" | tee -a "$O/files.sha256"
test "$(sha256sum "$DD/config.json" | cut -d' ' -f1)" = "$DCFG_SHA"
test "$(sha256sum "$DD/model.safetensors" | cut -d' ' -f1)" = "$DMODEL_SHA"
echo '== (a) dflashcheck profile, f16 =='
set +e
date -u
timeout 5m dflashcheck "$FX" cuda "$S" "$DD" f16 profile > "$O/profile-f16.txt" 2>&1
echo "dflashcheck profile f16 exit $?" >> "$O/profile-f16.txt"
echo '== (b) dflashcheck profile, f32 =='
date -u
timeout 5m dflashcheck "$FX" cuda "$S" "$DD" f32 profile > "$O/profile-f32.txt" 2>&1
echo "dflashcheck profile f32 exit $?" >> "$O/profile-f32.txt"
date -u
set -e
tail -n 80 "$O/profile-f16.txt" "$O/profile-f32.txt"
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
  --name "spec-loop-sp7" \
  "$PRE" "$BODY"
