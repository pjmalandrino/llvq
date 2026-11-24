#!/usr/bin/env bash
# Inkling 2 Mini on CUDA, the perplexity of a prompt read in pieces against the same prompt read in one call, on one
# card. Every position of a text is scored through the prompt path (`inkrun --score-prompt on`), paired by position.
# Texts: three WikiText-2 test windows of 2,048 tokens, and the first 4,096 tokens of windows 0 and 1 joined.
#   ref : one call, every key kept;
#   c   : pieces of 512 tokens (`--prefill-chunk 512`);
#   all : pieces of 512, sliding slots bounded to their window, scores in chunks of 128 query rows.
# Every arm: the batched experts at 8 rows, the bias on the card, the int4 head.
# FLAVOR=l40sx1 (default) or l4x1. Preregistered: proofs/preregistration-inkling-cuda-pieces-ppl-2026-11-24.md.
# Cost, estimated: about 14 minutes, $0.42 on l40sx1; 35-minute timeout.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=${PREREG:-proofs/preregistration-inkling-cuda-pieces-ppl-2026-11-24.md}
FLAVOR=${FLAVOR:-l40sx1}
UNEMBED_TABLE_SHA=cb26ce0c6145d35db52886b378185910b4adaf3de8024cb28e1e6f0062abcbb7
OUT_NAME=inkling-cuda-pieces-ppl-$FLAVOR-2026-11-24
W0_SHA=89ea14b9f41b5c2b4b9aa9bb88929b4a3c0dfde763fe4f30ece25f2d7e400dfb
W1_SHA=a35f1f08071c37fe0a09124afb349f5af6d1ba53d00a2dd5bd2c01d91024af43
W2_SHA=1fcd44bc6d6a5a78bb78e4bdcc4e43cd06db0db9032248ae09364b7f4e476bc1
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
IN=/out/inkling2-mini-inputs-2026-11-22
O=/out/$OUT_NAME${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
TIMEOUT=${TIMEOUT:-35m}
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

PRE=$(printf 'W0_SHA=%q\nW1_SHA=%q\nW2_SHA=%q\nUNEMBED_TABLE_SHA=%q\nO=%q\nIN=%q\nNTOK=%q\nLAYERS=%q\nTEXTONLY_SHA=%q\nCONFIG_SHA=%q\nTOKENS_SHA=%q\nMANIFEST_SHA=%q\nTEXTONLY_BYTES=%q\nexport HF_HOME=/tmp/hf' \
  "$W0_SHA" "$W1_SHA" "$W2_SHA" "$UNEMBED_TABLE_SHA" "$O" "$IN" "$NTOK" "$LAYERS" "$TEXTONLY_SHA" "$CONFIG_SHA" "$TOKENS_SHA" "$MANIFEST_SHA" "$TEXTONLY_BYTES")

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
for w in 0 1 2; do
  cp "$IN/windows/w$w.json" "$L/"
  S=W${w}_SHA
  test "$(sha256sum "$L/w$w.json" | cut -d' ' -f1)" = "${!S}"
done
python3 - "$L" <<'PY'
import json, sys
l = sys.argv[1]
ids = json.load(open(f"{l}/w0.json"))["ids"] + json.load(open(f"{l}/w1.json"))["ids"]
open(f"{l}/w01.json", "w").write(json.dumps({"ids": ids}, separators=(",", ":")))
PY
RC=0
for W in w0 w1 w2 w01; do
  for ARM in ref c all; do
    case $ARM in
      ref) F="";;
      c) F="--prefill-chunk 512";;
      all) F="--prefill-chunk 512 --kv-window bounded --attn-chunk 128";;
    esac
    echo "== inkrun $W $ARM =="
    date -u
    set +e
    env LLVQ_PREFILL=8:64 timeout 8m inkrun --config "$L/config.json" --checkpoint "$L/model.safetensors" --layers "$L/layers" \
      --tokens "$L/$W.json" --prompt-key ids --device cuda --require-layers "$LAYERS" \
      --moe four --glue fused --attn fused+scores --head int4 --head-table $L/unembed.q4g64.llvqraw \
      --moe-rows batch --attn-bias device $F \
      --score-prompt on --nll-out "$O/$W-$ARM.nll.json" \
      --out "$O/$W-$ARM.json" > "$O/$W-$ARM.txt" 2>&1
    R=$?
    set -e
    tail -1 "$O/$W-$ARM.txt"
    echo "inkrun $W $ARM exit $R" | tee -a "$O/summary.txt"
    [ "$R" = 0 ] || RC=$R
  done
done
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader | tee "$O/gpu-after.txt"
python3 - "$O" <<'PY' | tee -a "$O/summary.txt"
import json, math, os, sys
o = sys.argv[1]
def load(n):
    p = os.path.join(o, n + ".nll.json")
    if not os.path.exists(p): return None
    j = json.load(open(p)); return dict(zip(j["positions"], j["nll"]))
def stats(d):
    n = len(d); md = sum(d) / n
    se = math.sqrt(sum((x - md) ** 2 for x in d) / (n - 1)) / math.sqrt(n)
    return n, md, se
def line(tag, d, extra=""):
    n, md, se = stats(d)
    lo, hi = 100 * (math.exp(md - 1.96 * se) - 1), 100 * (math.exp(md + 1.96 * se) - 1)
    print(f"{tag}: {n} positions{extra}, ratio {100 * (math.exp(md) - 1):+.4f} % [{lo:+.4f}, {hi:+.4f}]")
    return md, lo, hi
for arm in ("c", "all"):
    pooled = []
    for w in ("w0", "w1", "w2", "w01"):
        a, b = load(f"{w}-ref"), load(f"{w}-{arm}")
        if a is None or b is None:
            print(f"{arm} {w}: a report is missing"); continue
        pos = sorted(set(a) & set(b)); d = [b[p] - a[p] for p in pos]; pooled += d if w != 'w01' else []  # w01 repeats w0 and w1
        pa = math.exp(sum(a[p] for p in pos) / len(pos)); pb = math.exp(sum(b[p] for p in pos) / len(pos))
        line(f"{arm} {w}", d, f", ppl ref {pa:.5f}, {arm} {pb:.5f}")
    if pooled:
        md, lo, hi = line(f"{arm} pooled", pooled)
        print(f"{arm} band +-0.12 %: point {'inside' if abs(100 * (math.exp(md) - 1)) <= 0.12 else 'OUTSIDE'}, "
              f"interval {'inside' if lo >= -0.12 and hi <= 0.12 else 'not inside'}")
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
  --name "inkling-cuda-pieces-ppl-$FLAVOR${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
