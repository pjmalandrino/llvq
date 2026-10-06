#!/usr/bin/env bash
# A second architecture, the runner half. Preregistered:
#   proofs/preregistration-a100-2026-10-06.md
#
# Two parts, two images, one a100-large each, LLVQ_NVRTC_ARCH=compute_80 on
# every command:
#   PART=bench   the ten-arm bench and the tile sweep of banc-252.sh, on the
#                standard image. planesbench compiles every kernel through
#                NVRTC, which is how F4 and A4 ran it on this card.
#   PART=served  the oracle and the three sealed files as paper-served.sh ran
#                them on the L40S. oracle and fusedrun also run candle, whose
#                kernels the image compiles for one compute capability, so
#                this part needs the sm80 image (`ops/run.py publish --cuda
#                --compute-cap 80 Pier-Jean/llvq-runner-cuda-sm80`), and its
#                Space sha in IMAGE_SHA.
#
# Cost: bench about 30 min, $1.25, timeout 45 min; served about 30 min, $1.25,
# timeout 50 min (estimated, $2.50/h).
# DRY_RUN=1 prints the job script and checks that it parses; nothing launches.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-a100-2026-10-06.md
PART=${PART:?PART=bench or PART=served}
case "$PART" in
  bench)  SPACE=Pier-Jean/llvq-runner-cuda
          IMAGE_SHA=${IMAGE_SHA:-97a2b62a6d0c8911dcd8e4d26a40461f0196ccc9}; TIMEOUT=45m ;;
  served) SPACE=Pier-Jean/llvq-runner-cuda-sm80
          IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the sm80 Space sha after its rebuild>}; TIMEOUT=50m ;;
  *) echo "refused: PART=$PART" >&2; exit 1 ;;
esac
BUCKET=Pier-Jean/jobs-artifacts
O=/out/a100-$PART-2026-10-06
DRY=${DRY_RUN:-0}

# size : path in the bucket : bytes : published sha256
SEALED=(
  "4b:sealed-4b-2026-09-23/qwen3-4b-sealed.bin:1418224685:886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8"
  "8b:sealed-8b27-2026-09-24/qwen3-8b-sealed-B.bin:2815098745:7bdb9a5503518081b1f652a93215250898caae5cc1b6a2f33ee9fb6094565c13"
  "14b:sealed-14b-2026-09-23/qwen3-14b-sealed.bin:5087000541:61db37fe7ce8e6a57c5c59f257319c796574981ecb5e450a02c73dd3c17e2ca0"
)
BALL="ball-ref-2026-09-20/qwen3-4b-llvq.bin:1770527533"
TETRA="tetra-4b-2026-09-06/qwen3-4b-tetra.bin:1770529149"

for s in 4b 8b 14b; do
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1]));
assert (d["layout"],d["embed"],d["rot_share"],d["fuse"],d["kv"])==("tetra48","q4","1","0","f16"), d' \
    "configs/qwen3-$s-tetra-e4.json"
done
CONF4=$(cat configs/qwen3-4b-tetra-e4.json)
CONF8=$(cat configs/qwen3-8b-tetra-e4.json)
CONF14=$(cat configs/qwen3-14b-tetra-e4.json)

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  for row in "${SEALED[@]}" "x:$BALL" "x:$TETRA"; do
    rest=${row#*:}; path=${rest%%:*}; rest=${rest#*:}; bytes=${rest%%:*}
    DIR=${path%/*}; NAME=${path##*/}
    REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/$DIR/" 2>/dev/null \
      | awk -v n="/$NAME" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
    [ "$REMOTE" = "$bytes" ] || { echo "refused: $path is ${REMOTE:-absent} B in the bucket, expected $bytes" >&2; exit 1; }
  done
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    "from huggingface_hub import HfApi; print(HfApi().space_info('$SPACE').sha)")
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: $SPACE is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "five bucket copies verified by size; image $NOW"
fi

PRE=$(printf 'O=%q\nROWS=%q\nB=%q\nF=%q\nCONF4=%q\nCONF8=%q\nCONF14=%q' "$O" "${SEALED[*]}" \
  "/out/${BALL%%:*}" "/out/${TETRA%%:*}" "$CONF4" "$CONF8" "$CONF14")

IFS= read -r -d '' HEAD <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
A=LLVQ_NVRTC_ARCH=compute_80
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt"
# Gate 1.
grep -q 'A100' "$O/gpu.txt"
grep -qE ', 8\.0, ' "$O/gpu.txt"

TILE='tile 128 (served fallback: sm_80 has no measured row)'
JOB
IFS= read -r -d '' SERVED <<'JOB' || true
echo '== oracle (hard rule 10), gate 2 =='
env $A oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle-cuda.txt" | tail -2
grep -q MATCH "$O/oracle-cuda.txt"
FLAGS='LLVQ_FUSED_LAYOUT=tetra48 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_KV=f16'
for row in $ROWS; do
  SIZE=${row%%:*}; rest=${row#*:}
  P=${rest%%:*}; rest=${rest#*:}
  BYTES=${rest%%:*}; SHA=${rest##*:}
  F_S=/out/$P
  case "$SIZE" in 4b) CONF=$CONF4 ;; 8b) CONF=$CONF8 ;; 14b) CONF=$CONF14 ;; esac
  echo "== served $SIZE: $P =="
  # Gate 5.
  test "$(stat -c %s "$F_S")" = "$BYTES"
  sha256sum "$F_S" | tee -a "$O/files.sha256"
  test "$(sha256sum "$F_S" | cut -d' ' -f1)" = "$SHA"
  C=$O/qwen3-$SIZE-tetra-e4.json
  printf '%s\n' "$CONF" > "$C"
  date -u
  env $A LLVQ_CONFIG="$C" LLVQ_PREFILL_TOKENS=203 fusedrun "$F_S" 2>&1 \
    | tee "$O/prefill-203-$SIZE.txt" | tail -14
  grep -qF "$TILE" "$O/prefill-203-$SIZE.txt"
  # Gate 6: the fused argmax is the dense argmax.
  grep -qE 'argmax ([0-9]+) vs \1$' "$O/prefill-203-$SIZE.txt"
  echo "== $SIZE, q4 tables, 256 tokens against the dense arm =="
  env $A $FLAGS LLVQ_EMBED=q4 fusedrun "$F_S" 256 2>&1 | tee "$O/fused-q4-256-$SIZE.txt" | tail -30
  grep -qF "$TILE" "$O/fused-q4-256-$SIZE.txt"
  grep -E 'tokens identical to the dense arm|divergence at token' "$O/fused-q4-256-$SIZE.txt" || true
  echo "== $SIZE, f16 tables, the same-head arm =="
  env $A $FLAGS LLVQ_EMBED=f16 fusedrun "$F_S" 256 2>&1 | tee "$O/fused-f16-256-$SIZE.txt" | tail -30
  grep -E 'tokens identical to the dense arm|divergence at token' "$O/fused-f16-256-$SIZE.txt" || true
  date -u
done

JOB
IFS= read -r -d '' BENCH <<'JOB' || true
P1=slot32,planes14,planes12x,golay70v1,fp16,awq,golay70v2,cublasf16,nullk
echo '== the ten arms at the fallback tile, phase 2 adds tetra48 =='
date -u
env $A LLVQ_BENCH_ARMS="$P1;$P1,tetra48" planesbench "$B" "$F" 2>&1 | tee "$O/banc.txt" | tail -45
date -u
# Gates 3 and 4.
grep -q 'tetra48: 252 of 252 matrices matched by name' "$O/banc.txt"
grep -qF "$TILE" "$O/banc.txt"

for T in 128 64 32; do
  echo "== sweep, tile $T =="
  env $A LLVQ_TILE_BLOCKS=$T LLVQ_BENCH_ARMS=fp16,planes14,nullk,tetra48 planesbench "$B" "$F" 2>&1 \
    | tee "$O/sweep-$T.txt" | grep -aE 'tile |Tetra48|Planes14|floor \(nullk\)'
  grep -q 'tetra48: 252 of 252 matrices matched by name' "$O/sweep-$T.txt"
done
JOB
IFS= read -r -d '' TAIL <<'JOB' || true
date -u
echo '== raw kept =='
wc -l "$O"/*.txt
JOB
case "$PART" in
  bench)  BODY="$HEAD
$BENCH
$TAIL" ;;
  served) BODY="$HEAD
$SERVED
$TAIL" ;;
esac

if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; nothing launched" >&2
  printf '%s\n' "$JOBSCRIPT"
  exit 0
fi

uv run ops/run.py bench \
  --image "hf.co/spaces/$SPACE" \
  --flavor a100-large --any-flavor --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "a100-$PART" \
  "$PRE" "$BODY"
