#!/usr/bin/env bash
# Job SP2 of the speculative loop (prereg proofs/preregistration-spec-loop-sp2-2026-11-10.md): on one L40S, the
# sealed 4B under its served v9 config, SP1's content first, then the real drafter. Each step writes its own file and is
# non-fatal for the next:
#   (a) `vtreebench --res a12` (compile only, 4m);
#   (b) `dflashcheck <fixture> cuda <sealed> <drafter> f32`, then the same with f16 (5m together);
#   (c) `specbench <sealed> a12@8.10.8.20.6 null,oracle:3,oracle:14 64` (12m);
#   (d) `specbench <sealed> a12@8.10.8.20.6 dflash:chain,dflash:tree 64 <drafter> f16` (12m), with f32 instead if (b)'s
#       f16 run failed (the summary says so).
# Inputs: the fixture from the jobs bucket at its sha256; the drafter z-lab/Qwen3-4B-DFlash-b16 at revision b74e3a32
# (MIT), fetched with curl at the pinned revision into a job-local directory, its files checked at their sha256.
# Controls as the verify-tree jobs: oracle, the object and its config at their sha256, LLVQ_* refused, the served 4E v9
# module.
#
#   DRY_RUN=1 bash ops/jobs/spec-loop-sp2.sh             parses the job script, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/spec-loop-sp2.sh  launches
#   FLAVOR=l40sx4 RETRY=1 TIMEOUT=40m ...                 the twin on l40sx4, one device visible
#
# Cost, estimated (l40sx1 $1.80/h, prereg "Cost"): about 30 min, $0.90; timeout 45m, $1.35.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-spec-loop-sp2-2026-11-10.md
OUT_NAME=spec-loop-sp2-2026-11-10
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/$OUT_NAME${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
P4E_V9_SHA=1b422b62328be6752df4add43280565fd2a49cfa20ee26953b301abe7a15c530
TIMEOUT=${TIMEOUT:-45m}
ARM='a12@8.10.8.20.6'
STANDINS='null,oracle:3,oracle:14'
DFLASH='dflash:chain,dflash:tree'
FIXTURE="dflash-fixture-2026-11-10/llvq-dflash-fixture-2026-11-10.safetensors:8562156"
FIXTURE_SHA=636f664d92ae9bb6fd17c9a01d55aecb342c72dfc5e1571d6ba219e77fed67a8
DREPO=z-lab/Qwen3-4B-DFlash-b16
DREV=b74e3a329c4d963783143b1e970d95b002be72bd
DCFG_SHA=2de97cdf48f429628074d229d024753f1ea02b575732983d6ed2bc882bbd63c8
DMODEL_SHA=a21d0ffe5970c84e9a0710ccb12087d1a4bcbedf979e623476c92c9d005a9cd4
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

PRE=$(printf 'O=%q\nSEALED=%q\nSEALED_SHA=%q\nCFG_SHA=%q\nCFG_B64=%q\nP4E_V9_SHA=%q\nARM=%q\nSTANDINS=%q\nDFLASH=%q\nNTOK=%q\nFIXTURE=%q\nFIXTURE_SHA=%q\nDREPO=%q\nDREV=%q\nDCFG_SHA=%q\nDMODEL_SHA=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$SEALED" "$SEALED_SHA" "$CFG_SHA" "$CFG_B64" "$P4E_V9_SHA" "$ARM" "$STANDINS" "$DFLASH" "$NTOK" "$FIXTURE" "$FIXTURE_SHA" "$DREPO" "$DREV" "$DCFG_SHA" "$DMODEL_SHA")$DEV_PRE

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
echo '== (a) vtreebench --res a12 (compile only, non-fatal) =='
set +e
timeout 4m vtreebench --res a12 > "$O/res.txt" 2>&1
echo "vtreebench --res exit $?" >> "$O/res.txt"
tail -8 "$O/res.txt"
echo '== (b) dflashcheck f32, then f16 =='
timeout 160s dflashcheck "$FX" cuda "$S" "$DD" f32 > "$O/dflashcheck-f32.txt" 2>&1
echo "dflashcheck f32 exit $?" >> "$O/dflashcheck-f32.txt"
timeout 140s dflashcheck "$FX" cuda "$S" "$DD" f16 > "$O/dflashcheck-f16.txt" 2>&1
rc16=$?
echo "dflashcheck f16 exit $rc16" >> "$O/dflashcheck-f16.txt"
tail -6 "$O/dflashcheck-f32.txt" "$O/dflashcheck-f16.txt"
DDT=f16
if [ "$rc16" != 0 ] || ! grep -q '^equivalence PASS' "$O/dflashcheck-f16.txt"; then DDT=f32; fi
echo '== (c) specbench, the stand-in drafters =='
date -u
LLVQ_CONFIG="$PT" timeout 12m specbench "$S" "$ARM" "$STANDINS" "$NTOK" > "$O/spec-standins.txt" 2>&1
echo "specbench stand-ins exit $?" >> "$O/spec-standins.txt"
echo "== (d) specbench, DFlash at $DDT =="
date -u
LLVQ_CONFIG="$PT" timeout 12m specbench "$S" "$ARM" "$DFLASH" "$NTOK" "$DD" "$DDT" > "$O/spec-dflash.txt" 2>&1
echo "specbench dflash exit $?" >> "$O/spec-dflash.txt"
date -u
set -e
{
  echo "drafter dtype for (d): $DDT$([ "$DDT" = f32 ] && echo ' (the f16 dflashcheck run failed or did not print PASS)')"
  sed 's/^/res: /' "$O/res.txt"
  sed 's/^/dflashcheck-f32: /' "$O/dflashcheck-f32.txt"
  sed 's/^/dflashcheck-f16: /' "$O/dflashcheck-f16.txt"
  for f in spec-standins spec-dflash; do
    grep -E '^4E: persistent module|^4E tree|^arm |^loaded|^== |^  |^summary|exit [0-9]+$' "$O/$f.txt" | sed "s/^/$f: /" || true
  done
} > "$O/summary.txt"
cat "$O/summary.txt" | tail -40
# control 2: 4E is the served v9 module, in both specbench runs
for f in spec-standins spec-dflash; do
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
  --flavor "$FLAVOR" ${FLAVOR_ARGS[@]+"${FLAVOR_ARGS[@]}"} --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "spec-loop-sp2${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
