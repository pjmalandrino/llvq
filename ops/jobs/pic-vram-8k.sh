#!/usr/bin/env bash
# The card gate of the prefill memory fixes (2026-11-08): peak card memory after an 8,192-token prompt under the
# served v9 configs (in-place softmax, f16 mask, syncs), the same with the whole-tensor softmax in a second process
# (LLVQ_SOFTMAX_WHOLE=1, 4B only) to check the logits bit for bit, and the NOT SERVED kprefill configs. Same 100 ms
# nvidia-smi sampler as the duel (ops/jobs/duel-tetra.sh), each arm in its own process.
# Preregistered: proofs/preregistration-pic-vram-8k-2026-11-08.md, to stamp before any launch.
#
#   DRY_RUN=1 bash ops/jobs/pic-vram-8k.sh                 parses the job script, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/pic-vram-8k.sh     launches (only after the operator's go)
#
# Cost, estimated: ~58 min on l40sx1, $1.75; timeout 80m ($2.40).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-pic-vram-8k-2026-11-08.md
BUCKET=Pier-Jean/jobs-artifacts
O=/out/pic-vram-8k-2026-11-08${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
IMAGE_SHA=${IMAGE_SHA:-}
SEALED="4b:sealed-4b-2026-09-23/qwen3-4b-sealed.bin:886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8
8b:sealed-8b27-2026-09-24/qwen3-8b-sealed-B.bin:7bdb9a5503518081b1f652a93215250898caae5cc1b6a2f33ee9fb6094565c13
14b:sealed-14b-2026-09-23/qwen3-14b-sealed.bin:61db37fe7ce8e6a57c5c59f257319c796574981ecb5e450a02c73dd3c17e2ca0"

CONFIGS=""
for s in 4b 8b 14b; do
  for c in configs/qwen3-$s-tetra-e4-4e-v9.json configs/qwen3-$s-tetra-e4-4e-v9-kprefill.json; do
    CONFIGS="$CONFIGS$(basename "$c"):$(base64 < "$c" | tr -d '\n'):$(shasum -a 256 < "$c" | cut -d' ' -f1)
"
  done
done

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha built from this branch>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
fi

PRE=$(printf 'O=%q\nSEALED=%q\nCONFIGS=%q\nexport HF_HOME=/tmp/hf' "$O" "$SEALED" "$CONFIGS")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
timeout 5m oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle.txt" | tail -2
grep -q MATCH "$O/oracle.txt"
vram() {
  local tag=$1; shift
  local idle; idle=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 100 > "$O/vram-$tag.csv" &
  local sp=$!
  local rc=0
  "$@" || rc=$?
  kill "$sp"; wait "$sp" 2>/dev/null || true
  local peak; peak=$(sort -n "$O/vram-$tag.csv" | tail -1)
  echo "vram $tag idle $idle MiB peak $peak MiB" | tee -a "$O/vram.txt"
  return $rc
}
config() {
  local name=$1 P=$O/$1
  local cl; cl=$(printf '%s\n' "$CONFIGS" | grep "^$name:")
  echo "$cl" | cut -d: -f2 | base64 -d > "$P"
  test "$(sha256sum "$P" | cut -d' ' -f1)" = "$(echo "$cl" | cut -d: -f3)"
  echo "$P"
}
sweep() {  # tag config file [env...]
  local tag=$1 P=$2 f=$3; shift 3
  date -u
  vram "$tag" env "$@" LLVQ_CONFIG="$P" LLVQ_CONTEXTS=8192 timeout 30m fusedrun "$f" 256 > "$O/sweep-$tag.txt" 2>&1
  grep -q "^  prompt 8192: 4E tokens equal the eager path.s under this config, 256 of 256" "$O/sweep-$tag.txt"
  grep -E "^ +8192 +[0-9.]+ \[|logits fingerprint|prefill attention kernel, last" "$O/sweep-$tag.txt" | sed "s/^/$tag /" | tee -a "$O/speed.txt"
}
for s in 4b 8b 14b; do
  echo "== $s =="
  line=$(printf '%s\n' "$SEALED" | grep "^$s:")
  f=/out/$(echo "$line" | cut -d: -f2); want=$(echo "$line" | cut -d: -f3)
  test "$(sha256sum "$f" | cut -d' ' -f1)" = "$want"
  V=$(config "qwen3-$s-tetra-e4-4e-v9.json")
  K=$(config "qwen3-$s-tetra-e4-4e-v9-kprefill.json")
  sweep "$s-served" "$V" "$f"
  if [ "$s" = 4b ]; then sweep "$s-whole" "$V" "$f" LLVQ_SOFTMAX_WHOLE=1; fi
  sweep "$s-kprefill" "$K" "$f"
done
cat "$O/vram.txt"
ls -la "$O"
JOB

if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; nothing launched; out $O" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout 80m \
  --bucket "$BUCKET" --out-mount /out \
  --name "pic-vram-8k${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
