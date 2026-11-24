#!/usr/bin/env bash
# Inkling 2 Mini on CUDA, the int4 output head (`kernels/ink_head.cu`), on one card. Two arms, the experts in four
# launches, the fused glue and the dense attention kernels in both:
#   attn : the checkpoint's BF16 head through candle (the last arm of the attention run);
#   head : the sealed file's int4 g64 output table through the head kernel, packed on the card.
# FLAVOR=l4x1 (default) or l40sx1: the same job on the card the target is stated on. The output directory carries
# the flavor. Preregistered: proofs/preregistration-inkling-cuda-head-2026-11-23.md.
# Cost, estimated: about 6 minutes, $0.08 on l4x1 ($0.80/h), $0.18 on l40sx1 ($1.80/h); 20-minute timeout.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-inkling-cuda-head-2026-11-23.md
FLAVOR=${FLAVOR:-l4x1}
UNEMBED_TABLE_SHA=cb26ce0c6145d35db52886b378185910b4adaf3de8024cb28e1e6f0062abcbb7
OUT_NAME=inkling-cuda-head-$FLAVOR-2026-11-23
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
IN=/out/inkling2-mini-inputs-2026-11-22
O=/out/$OUT_NAME${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
TIMEOUT=${TIMEOUT:-20m}
NTOK=32
LAYERS=24
# Pinned by the staging script, which prints these values; refused below while a pin is unset.
TEXTONLY_SHA=${TEXTONLY_SHA:-}
CONFIG_SHA=${CONFIG_SHA:-}
TOKENS_SHA=${TOKENS_SHA:-}
MANIFEST_SHA=${MANIFEST_SHA:-}
TEXTONLY_BYTES=3218968280

if [ "$DRY" != 1 ]; then
  for v in TEXTONLY_SHA CONFIG_SHA TOKENS_SHA MANIFEST_SHA; do
    [ -n "${!v}" ] || { echo "refused: $v unset; ops/inkling/stage-first-card.sh prints it" >&2; exit 1; }
  done
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  # every input in the bucket, by name and size, before a card is rented
  LS=$(hf buckets ls "hf://buckets/$BUCKET/${IN#/out/}/" 2>/dev/null || true)
  for f in config.json model.safetensors tokens.json codes.sha256; do
    echo "$LS" | grep -q "/$f\$" || { echo "refused: $f absent from ${IN#/out/}/" >&2; exit 1; }
  done
  echo "$LS" | awk -v n=/model.safetensors 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' \
    | grep -qx "$TEXTONLY_BYTES" || { echo "refused: model.safetensors is not $TEXTONLY_BYTES B in the bucket" >&2; exit 1; }
  NT=$(hf buckets ls "hf://buckets/$BUCKET/${IN#/out/}/layers/" 2>/dev/null | grep -c '/L[0-9][0-9]\.tar$' || true)
  [ "$NT" = "$LAYERS" ] || { echo "refused: $NT layer archives in the bucket, $LAYERS expected" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of the rebuild that carries inkrun>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "bucket inputs present; image $NOW"
fi

PRE=$(printf 'UNEMBED_TABLE_SHA=%q\nO=%q\nIN=%q\nNTOK=%q\nLAYERS=%q\nTEXTONLY_SHA=%q\nCONFIG_SHA=%q\nTOKENS_SHA=%q\nMANIFEST_SHA=%q\nTEXTONLY_BYTES=%q\nexport HF_HOME=/tmp/hf' \
  "$UNEMBED_TABLE_SHA" "$O" "$IN" "$NTOK" "$LAYERS" "$TEXTONLY_SHA" "$CONFIG_SHA" "$TOKENS_SHA" "$MANIFEST_SHA" "$TEXTONLY_BYTES")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
command -v inkrun > /dev/null || { echo 'STOP: no inkrun in the image (rebuild with ops/Dockerfile.cuda)' >&2; exit 1; }
nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.used,driver_version --format=csv,noheader | tee "$O/gpu-before.txt"
df -h "${SCRATCH:-/scratch}" /tmp | tee "$O/disk.txt"
echo '== control 1: oracle on this backend =='
timeout 5m oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle.txt" | tail -2
grep -q MATCH "$O/oracle.txt"
echo '== control 2: inputs to local disk, at their pins =='
L=${SCRATCH:-/scratch}/inkling
mkdir -p "$L/layers"
date -u
cp "$IN/config.json" "$IN/model.safetensors" "$IN/tokens.json" "$IN/codes.sha256" "$L/"
test "$(stat -c %s "$L/model.safetensors")" = "$TEXTONLY_BYTES"
for i in $(seq 0 $((LAYERS - 1))); do tar -xf "$IN/layers/L$(printf %02d "$i").tar" -C "$L/layers"; done
date -u
test "$(sha256sum "$L/config.json" | cut -d' ' -f1)" = "$CONFIG_SHA"
test "$(sha256sum "$L/model.safetensors" | cut -d' ' -f1)" = "$TEXTONLY_SHA"
test "$(sha256sum "$L/tokens.json" | cut -d' ' -f1)" = "$TOKENS_SHA"
test "$(sha256sum "$L/codes.sha256" | cut -d' ' -f1)" = "$MANIFEST_SHA"
(cd "$L/layers" && sha256sum --quiet -c ../codes.sha256) 2>&1 | tee "$O/codes-check.txt"
test "$(grep -c . "$L/codes.sha256")" = "$((LAYERS * 129))"
sha256sum "$L/config.json" "$L/model.safetensors" "$L/tokens.json" "$L/codes.sha256" | tee "$O/files.sha256"
date -u
cp "$IN/tables/unembed.q4g64.llvqraw" "$L/"
test "$(sha256sum "$L/unembed.q4g64.llvqraw" | cut -d' ' -f1)" = "$UNEMBED_TABLE_SHA"
RC=0
for ARM in attn head; do
  case $ARM in attn) F="";; head) F="--head int4 --head-table $L/unembed.q4g64.llvqraw";; esac
  echo "== inkrun $ARM =="
  set +e
  timeout 6m inkrun --config "$L/config.json" --checkpoint "$L/model.safetensors" --layers "$L/layers" \
    --tokens "$L/tokens.json" --device cuda --n "$NTOK" --require-layers "$LAYERS" --moe four --glue fused --attn fused $F \
    --out "$O/inkrun-$ARM.json" > "$O/inkrun-$ARM.txt" 2>&1
  R=$?
  set -e
  tail -4 "$O/inkrun-$ARM.txt"
  grep -n 'NVRTC \(moe\|glue\|attn\|head\) unit\|keeps the one-expert path' "$O/inkrun-$ARM.txt" || true
  echo "inkrun $ARM exit $R" | tee -a "$O/summary.txt"
  [ "$R" = 0 ] || RC=$R
done
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader | tee "$O/gpu-after.txt"
python3 - "$O" <<'PY' | tee -a "$O/summary.txt"
import json, os, sys
o = sys.argv[1]
for arm in ("attn", "head"):
    p = os.path.join(o, f"inkrun-{arm}.json")
    if not os.path.exists(p):
        print(f"{arm}: no report"); continue
    r = json.load(open(p)); st = r["step_seconds"]
    print(f"{arm}: median step {r['step_seconds_median']:.4f} s [{min(st):.4f}, {max(st):.4f}], "
          f"{r['tokens_per_second_median_step']:.2f} tok/s; prefill of {r['prompt_tokens']} tokens {r['seconds_prefill']:.2f} s; "
          f"GB after the experts {r['memory']['after_resident'].get('used', 0) / 1e9:.3f}, at the end {r['memory']['end'].get('used', 0) / 1e9:.3f}; "
          f"tokens equal to the Mac's: {r['all_equal']} (prefix {r['equal_prefix']} of {r['new_tokens']}); "
          f"head {r['provenance'].get('head')}, min top-2 margin {r['min_margin']:.4g}")
    print(f"{arm} tokens", r["tokens"])
PY
ls -la "$O"
exit $RC
JOB

if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  if [ "${PRINT_JOB:-0}" = 1 ]; then printf '%s\n' "$JOBSCRIPT"; exit 0; fi
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; nothing launched" >&2
  grep -q -- '--bin inkrun' ops/Dockerfile.cuda && grep -q '/src/target/release/inkrun' ops/Dockerfile.cuda \
    || { echo "DRY_RUN: inkrun is not in both lists of ops/Dockerfile.cuda" >&2; exit 1; }
  python3 -I ops/check_image_lists.py
  exit 0
fi

# l4x1 is not in BENCH_FLAVORS (a bench ratio there is not comparable to the Metal one); this job is a first run, not
# a kernel ratio, and the prereg says so, hence --any-flavor.
uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor "$FLAVOR" --any-flavor --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "inkling-cuda-head-$FLAVOR${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
