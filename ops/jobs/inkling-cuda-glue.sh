#!/usr/bin/env bash
# Inkling 2 Mini on CUDA, the fused glue of a decoded row (`llvq-llm/kernels/ink_glue.cu`, the twin of the Metal
# kernels of `inkling/glue.rs`) against candle's operations, on one L4. Both arms take the experts in four launches.
#   (1) `inkrun --glue candle`: the path of the moe4 run, 32 greedy tokens, its logits written;
#   (2) `inkrun --glue fused`: the same run with the glue kernels and the routing kept on the card;
#   (3) the tokens of both against the Mac's, the largest logit difference between the arms, the median steps.
# Output inkling-cuda-glue-2026-11-23. Preregistered: proofs/preregistration-inkling-cuda-glue-2026-11-23.md.
# Cost (l4x1 $0.80/h), estimated: about 5 minutes, $0.07; $0.40 at the 30-minute timeout.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-inkling-cuda-glue-2026-11-23.md
OUT_NAME=inkling-cuda-glue-2026-11-23
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
IN=/out/inkling2-mini-inputs-2026-11-22
O=/out/$OUT_NAME${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
TIMEOUT=${TIMEOUT:-30m}
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

PRE=$(printf 'O=%q\nIN=%q\nNTOK=%q\nLAYERS=%q\nTEXTONLY_SHA=%q\nCONFIG_SHA=%q\nTOKENS_SHA=%q\nMANIFEST_SHA=%q\nTEXTONLY_BYTES=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$IN" "$NTOK" "$LAYERS" "$TEXTONLY_SHA" "$CONFIG_SHA" "$TOKENS_SHA" "$MANIFEST_SHA" "$TEXTONLY_BYTES")

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
RC=0
for ARM in candle fused; do
  echo "== inkrun --glue $ARM: 24 layers resident, $NTOK greedy tokens =="
  set +e
  timeout 10m inkrun --config "$L/config.json" --checkpoint "$L/model.safetensors" --layers "$L/layers" \
    --tokens "$L/tokens.json" --device cuda --n "$NTOK" --require-layers "$LAYERS" --moe four --glue "$ARM" \
    --logits-out "$L/logits-$ARM.f32" --out "$O/inkrun-$ARM.json" > "$O/inkrun-$ARM.txt" 2>&1
  R=$?
  set -e
  tail -12 "$O/inkrun-$ARM.txt"
  grep -n 'NVRTC moe unit\|NVRTC glue unit\|keeps the one-expert path' "$O/inkrun-$ARM.txt" || true
  echo "inkrun --glue $ARM exit $R" | tee -a "$O/summary.txt"
  [ "$R" = 0 ] || RC=$R
done
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | tee "$O/gpu-after.txt"
if [ -f "$L/logits-candle.f32" ] && [ -f "$L/logits-fused.f32" ]; then
  sha256sum "$L/logits-candle.f32" "$L/logits-fused.f32" | tee "$O/logits.sha256"
  python3 - "$L/logits-candle.f32" "$L/logits-fused.f32" <<'PY' | tee -a "$O/summary.txt"
import array, sys
a = array.array("f"); a.frombytes(open(sys.argv[1], "rb").read())
b = array.array("f"); b.frombytes(open(sys.argv[2], "rb").read())
assert len(a) == len(b) and len(a) % 200058 == 0, (len(a), len(b))
rows = len(a) // 200058
worst = 0.0; top = 0.0
for r in range(rows):
    ra = a[r * 200058:(r + 1) * 200058]; rb = b[r * 200058:(r + 1) * 200058]
    worst = max(worst, max(abs(x - y) for x, y in zip(ra, rb)))
    top = max(top, max(abs(x) for x in ra))
print(f"logits: {rows} rows, largest |candle - fused| {worst:.6g}, largest |logit| {top:.6g}, identical {a == b}")
PY
fi
python3 - "$O" <<'PY' | tee -a "$O/summary.txt"
import json, os, sys
o = sys.argv[1]
for arm in ("candle", "fused"):
    p = os.path.join(o, f"inkrun-{arm}.json")
    if not os.path.exists(p):
        print(f"{arm}: no report"); continue
    r = json.load(open(p)); st = r["step_seconds"]
    print(f"{arm}: median step {r['step_seconds_median']:.4f} s [{min(st):.4f}, {max(st):.4f}], "
          f"{r['tokens_per_second_median_step']:.2f} tok/s; lattice launches a step {sorted(set(r['lattice_launches_per_step']))}; "
          f"GB at the end {r['memory']['end'].get('used', 0) / 1e9:.3f}; "
          f"tokens equal to the Mac's: {r['all_equal']} (prefix {r['equal_prefix']} of {r['new_tokens']}); "
          f"provenance glue {r['provenance'].get('glue')}, routing on device {r['provenance'].get('routing_on_device')}")
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
  --flavor l4x1 --any-flavor --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "inkling-cuda-glue${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
