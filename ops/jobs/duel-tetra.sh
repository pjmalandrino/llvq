#!/usr/bin/env bash
# Our side of the llama.cpp duel: the three sealed files at their v9 door, decode speed after a prompt of 5 and of
# 8,192 tokens, each length in its own process under the same 100 ms VRAM sampler the llama.cpp job uses. Our quality
# numbers are not re-measured (docs/ETAT.md section 2); only the speed is re-read, the same day, for the peak memory
# that served-v9 did not take (it printed cuMemGetInfo after the install, before any context).
# Preregistered: proofs/preregistration-duel-iq2-2026-11-08.md, to stamp before any launch.
#
#   DRY_RUN=1 bash ops/jobs/duel-tetra.sh                 parses the job script, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/duel-tetra.sh     launches (only after the operator's go)
#
# The sweep checks, at each length, that 4E decodes the eager path's tokens; the eager run lives in the same process,
# so the peak read here is the larger of the two engines. That can only overstate our memory.
#
# Cost, estimated: ~30 min on l40sx1, $0.90, timeout 60m ($1.80).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-duel-iq2-2026-11-08.md
BUCKET=Pier-Jean/jobs-artifacts
O=/out/duel-iq2-2026-11-08/tetra${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
IMAGE_SHA=${IMAGE_SHA:-}
SEALED="4b:sealed-4b-2026-09-23/qwen3-4b-sealed.bin:886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8
8b:sealed-8b27-2026-09-24/qwen3-8b-sealed-B.bin:7bdb9a5503518081b1f652a93215250898caae5cc1b6a2f33ee9fb6094565c13
14b:sealed-14b-2026-09-23/qwen3-14b-sealed.bin:61db37fe7ce8e6a57c5c59f257319c796574981ecb5e450a02c73dd3c17e2ca0"

CONFIGS=""
for s in 4b 8b 14b; do
  c=configs/qwen3-$s-tetra-e4-4e-v9.json
  grep -q '^  "rows": "v9",$' "$c" || { echo "refused: $c does not name rows v9" >&2; exit 1; }
  CONFIGS="$CONFIGS$s:$(base64 < "$c" | tr -d '\n'):$(shasum -a 256 < "$c" | cut -d' ' -f1)
"
done

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha that carries 4E v9 at three sizes>}
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
WANT='attention kernel from position 0 on grow(256), decode persistent (4E rows v9, one launch a token run)'
for s in 4b 8b 14b; do
  echo "== $s =="
  line=$(printf '%s\n' "$SEALED" | grep "^$s:")
  f=/out/$(echo "$line" | cut -d: -f2); want=$(echo "$line" | cut -d: -f3)
  test "$(sha256sum "$f" | cut -d' ' -f1)" = "$want"
  ls -l "$f" | tee -a "$O/sizes.txt"
  cl=$(printf '%s\n' "$CONFIGS" | grep "^$s:")
  P=$O/qwen3-$s-tetra-e4-4e-v9.json
  echo "$cl" | cut -d: -f2 | base64 -d > "$P"
  test "$(sha256sum "$P" | cut -d' ' -f1)" = "$(echo "$cl" | cut -d: -f3)"
  for n in 5 8192; do
    date -u
    vram "tetra-$s-p$n" env LLVQ_CONFIG="$P" LLVQ_CONTEXTS="$n" timeout 30m fusedrun "$f" 256 > "$O/sweep-$s-p$n.txt" 2>&1
    grep -E '^served config: ' "$O/sweep-$s-p$n.txt" | grep -qF "$WANT"
    grep -q "^  prompt $n: 4E tokens equal the eager path.s under this config, 256 of 256" "$O/sweep-$s-p$n.txt"
    grep -E "^ +$n +[0-9.]+ \[" "$O/sweep-$s-p$n.txt" | tee -a "$O/speed.txt" | sed "s/^/$s /"
    { grep -E '^card memory' "$O/sweep-$s-p$n.txt" || true; } | sed "s/^/$s p$n /" | tee -a "$O/card-memory.txt"
  done
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
  --flavor l40sx1 --timeout 60m \
  --bucket "$BUCKET" --out-mount /out \
  --name "duel-tetra${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
