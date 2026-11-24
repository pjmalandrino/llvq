#!/usr/bin/env bash
# Inkling 2 Mini on CUDA, three arms on one L4, the experts in four launches in each:
#   candle : glue and attention through candle (the path of the moe4 run);
#   glue   : the fused glue (`kernels/ink_glue.cu`);
#   attn   : the fused glue and the dense attention products (`kernels/ink_attn.cu`).
# Each arm runs twice: once plain (the timed run, its logits written), once with the routing of every decode step
# recorded (`--routing-steps-out`, untimed). The summary gives, per arm, the median step, the tokens against the
# Mac's, and against the candle arm the largest logit difference row by row and the visits whose routed experts
# differ, with the margin of the candle arm there: a class B drift is small unless a routing flips at a near-tie.
# Output inkling-cuda-attn-2026-11-23. Preregistered: proofs/preregistration-inkling-cuda-attn-2026-11-23.md.
# Cost (l4x1 $0.80/h), estimated: about 9 minutes, $0.12; $0.40 at the 30-minute timeout.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-inkling-cuda-attn-2026-11-23.md
OUT_NAME=inkling-cuda-attn-2026-11-23
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
for ARM in candle glue attn; do
  case $ARM in candle) F="--glue candle --attn candle";; glue) F="--glue fused --attn candle";; attn) F="--glue fused --attn fused";; esac
  echo "== inkrun $ARM ($F) =="
  set +e
  timeout 6m inkrun --config "$L/config.json" --checkpoint "$L/model.safetensors" --layers "$L/layers" \
    --tokens "$L/tokens.json" --device cuda --n "$NTOK" --require-layers "$LAYERS" --moe four $F \
    --logits-out "$L/logits-$ARM.f32" --out "$O/inkrun-$ARM.json" > "$O/inkrun-$ARM.txt" 2>&1
  R=$?
  timeout 6m inkrun --config "$L/config.json" --checkpoint "$L/model.safetensors" --layers "$L/layers" \
    --tokens "$L/tokens.json" --device cuda --n "$NTOK" --require-layers "$LAYERS" --moe four $F \
    --routing-steps-out "$O/routing-$ARM.json" --out "$O/inkrun-$ARM-routing.json" > "$O/inkrun-$ARM-routing.txt" 2>&1
  R2=$?
  set -e
  tail -4 "$O/inkrun-$ARM.txt"
  grep -n 'NVRTC \(moe\|glue\|attn\) unit\|keeps the one-expert path' "$O/inkrun-$ARM.txt" || true
  echo "inkrun $ARM exit $R, with routing recorded exit $R2" | tee -a "$O/summary.txt"
  [ "$R" = 0 ] || RC=$R
done
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | tee "$O/gpu-after.txt"
sha256sum "$L"/logits-*.f32 | tee "$O/logits.sha256"
python3 - "$O" "$L" <<'PY' | tee -a "$O/summary.txt"
import array, json, os, sys
o, l = sys.argv[1], sys.argv[2]
V = 200058
def logits(arm):
    p = os.path.join(l, f"logits-{arm}.f32")
    if not os.path.exists(p): return None
    a = array.array("f"); a.frombytes(open(p, "rb").read()); return a
def routing(arm):
    p = os.path.join(o, f"routing-{arm}.json")
    return json.load(open(p)) if os.path.exists(p) else None
ref, rref = logits("candle"), routing("candle")
for arm in ("candle", "glue", "attn"):
    p = os.path.join(o, f"inkrun-{arm}.json")
    if not os.path.exists(p):
        print(f"{arm}: no report"); continue
    r = json.load(open(p)); st = r["step_seconds"]
    print(f"{arm}: median step {r['step_seconds_median']:.4f} s [{min(st):.4f}, {max(st):.4f}], "
          f"{r['tokens_per_second_median_step']:.2f} tok/s; GB at the end {r['memory']['end'].get('used', 0) / 1e9:.3f}; "
          f"tokens equal to the Mac's: {r['all_equal']} (prefix {r['equal_prefix']} of {r['new_tokens']}); "
          f"glue {r['provenance'].get('glue')}, attn {r['provenance'].get('attn')}")
    a = logits(arm)
    if arm != "candle" and a is not None and ref is not None and len(a) == len(ref):
        rows = len(a) // V
        d = [max(abs(x - y) for x, y in zip(a[i * V:(i + 1) * V], ref[i * V:(i + 1) * V])) for i in range(rows)]
        print(f"{arm}: largest |logit - candle| row by row: " + " ".join(f"{x:.3g}" for x in d))
    ro = routing(arm)
    if arm != "candle" and ro is not None and rref is not None:
        flips = []
        for s, (sa, sb) in enumerate(zip(ro, rref)):
            for v, (va, vb) in enumerate(zip(sa, sb)):
                if sorted(va["experts"]) != sorted(vb["experts"]):
                    flips.append((s, v, vb["margin"]))
        print(f"{arm}: {len(flips)} visits of {sum(len(x) for x in rref)} route other experts than candle; "
              f"first (step, visit, candle margin): {flips[:8]}")
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
  --name "inkling-cuda-attn${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
