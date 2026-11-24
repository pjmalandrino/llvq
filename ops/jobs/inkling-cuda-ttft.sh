#!/usr/bin/env bash
# Inkling 2 Mini on CUDA, the time to read a prompt (the time before the first token), under the whole stack, on one
# card. The prompt is 2,000 tokens of a WikiText-2 test window; 32 greedy tokens after it. Arms:
#   chunk   : the resident experts read the prompt one expert and one chunk of rows at a time (the path so far);
#   batch   : one launch a step for every (token, expert) pair of a layer (`inkrun --moe-rows batch`);
#   batch8, batch16, chunk16 : the same under LLVQ_PREFILL=8:64 and 16:32, rows a decode of the weight stream.
# Every arm writes its logits; each is compared byte for byte with `chunk`.
# FLAVOR=l40sx1 (default) or l4x1. Preregistered: proofs/preregistration-inkling-cuda-ttft-2026-11-24.md.
# Cost, estimated: about 13 minutes, $0.40 on l40sx1, $0.20 on l4x1; 25-minute timeout.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=${PREREG:-proofs/preregistration-inkling-cuda-ttft-2026-11-24.md}
FLAVOR=${FLAVOR:-l40sx1}
UNEMBED_TABLE_SHA=cb26ce0c6145d35db52886b378185910b4adaf3de8024cb28e1e6f0062abcbb7
OUT_NAME=inkling-cuda-ttft-$FLAVOR-2026-11-24
EVENTS=${EVENTS:-on}
ARMS=${ARMS:-chunk batch batch8 batch16 chunk16}
W0_SHA=89ea14b9f41b5c2b4b9aa9bb88929b4a3c0dfde763fe4f30ece25f2d7e400dfb
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
IN=/out/inkling2-mini-inputs-2026-11-22
O=/out/$OUT_NAME${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
TIMEOUT=${TIMEOUT:-25m}
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

PRE=$(printf 'export ARMS=%q\nEVENTS=%q\nW0_SHA=%q\nUNEMBED_TABLE_SHA=%q\nO=%q\nIN=%q\nNTOK=%q\nLAYERS=%q\nTEXTONLY_SHA=%q\nCONFIG_SHA=%q\nTOKENS_SHA=%q\nMANIFEST_SHA=%q\nTEXTONLY_BYTES=%q\nexport HF_HOME=/tmp/hf' \
  "$ARMS" "$EVENTS" "$W0_SHA" "$UNEMBED_TABLE_SHA" "$O" "$IN" "$NTOK" "$LAYERS" "$TEXTONLY_SHA" "$CONFIG_SHA" "$TOKENS_SHA" "$MANIFEST_SHA" "$TEXTONLY_BYTES")

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
cp "$IN/windows/w0.json" "$L/"
test "$(sha256sum "$L/w0.json" | cut -d' ' -f1)" = "$W0_SHA"
# The long prompt: windows 0, 1, 2, 0, 1 of the staged WikiText-2 test windows, 10,240 ids. A repeat: it is a
# prompt for time and memory, not for quality.
for w in 1 2; do cp "$IN/windows/w$w.json" "$L/"; done
python3 - "$L" <<'PY'
import json, sys
l = sys.argv[1]
ids = []
for w in ("w0", "w1", "w2", "w0", "w1"):
    ids += json.load(open(f"{l}/{w}.json"))["ids"]
open(f"{l}/long.json", "w").write(json.dumps({"ids": ids}, separators=(",", ":")))
PY
test "$(sha256sum "$L/long.json" | cut -d' ' -f1)" = 776675c5d277b49dc38c7fbab2a44d8286d5c5b2020a4de04ff666c2c6f36830
for ARM in $ARMS; do
  N=2000
  case $ARM in
    chunk) F="--moe-rows chunk"; P=;;
    batch) F="--moe-rows batch"; P=;;
    batch8) F="--moe-rows batch"; P=8:64;;
    batch16) F="--moe-rows batch"; P=16:32;;
    chunk16) F="--moe-rows chunk"; P=16:32;;
    phchunk) F="--moe-rows chunk --prefill-phases on"; P=;;
    phbatch8) F="--moe-rows batch --prefill-phases on"; P=8:64;;
    batch4x64) F="--moe-rows batch"; P=4:64;;
    batch8x32) F="--moe-rows batch"; P=8:32;;
    pk2000c) F="--moe-rows chunk --peak-probe on"; P=;;
    pk2000) F="--moe-rows batch --peak-probe on"; P=8:64;;
    t9000) F="--moe-rows batch"; P=8:64; N=9000;;
    pk9000) F="--moe-rows batch --peak-probe on"; P=8:64; N=9000;;
    d2000) F="--moe-rows batch --attn-bias device"; P=8:64;;
    phd2000) F="--moe-rows batch --attn-bias device --prefill-phases on"; P=8:64;;
    d9000) F="--moe-rows batch --attn-bias device"; P=8:64; N=9000;;
    phd9000) F="--moe-rows batch --attn-bias device --prefill-phases on"; P=8:64; N=9000;;
    r2000) F="--moe-rows batch --attn-bias device --peak-probe on"; P=8:64;;
    c2000) F="--moe-rows batch --attn-bias device --peak-probe on --prefill-chunk 512"; P=8:64;;
    k2000) F="--moe-rows batch --attn-bias device --peak-probe on --kv-window bounded"; P=8:64;;
    r9000) F="--moe-rows batch --attn-bias device --peak-probe on"; P=8:64; N=9000;;
    c9000) F="--moe-rows batch --attn-bias device --peak-probe on --prefill-chunk 512"; P=8:64; N=9000;;
    k9000) F="--moe-rows batch --attn-bias device --peak-probe on --kv-window bounded"; P=8:64; N=9000;;
    a9000) F="--moe-rows batch --attn-bias device --peak-probe on --prefill-chunk 512 --kv-window bounded --attn-chunk 128"; P=8:64; N=9000;;
    ta9000) F="--moe-rows batch --attn-bias device --prefill-chunk 512 --kv-window bounded --attn-chunk 128"; P=8:64; N=9000;;
    *) echo "unknown arm $ARM" >&2; exit 1;;
  esac
  TOK=$L/w0.json
  [ "$N" = 2000 ] || TOK=$L/long.json
  F="$F --tokens $TOK --prompt-key ids --prompt-len $N --attn fused+scores --events $EVENTS"
  echo "== inkrun $ARM ($F, LLVQ_PREFILL=${P:-unset}) =="
  set +e
  ${P:+env LLVQ_PREFILL=$P} timeout 6m inkrun --config "$L/config.json" --checkpoint "$L/model.safetensors" --layers "$L/layers" \
    --device cuda --n "$NTOK" --require-layers "$LAYERS" --moe four --glue fused \
    --head int4 --head-table "$L/unembed.q4g64.llvqraw" $F --logits-out "$O/logits-$ARM.f32" \
    --out "$O/inkrun-$ARM.json" > "$O/inkrun-$ARM.txt" 2>&1
  R=$?
  set -e
  tail -3 "$O/inkrun-$ARM.txt"
  grep -n 'keeps the one-expert path' "$O/inkrun-$ARM.txt" | head -3 || true
  echo "inkrun $ARM exit $R" | tee -a "$O/summary.txt"
  [ "$R" = 0 ] || RC=$R
done
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader | tee "$O/gpu-after.txt"
sha256sum "$O"/logits-*.f32 | tee -a "$O/summary.txt"
python3 - "$O" <<'PY' | tee -a "$O/summary.txt"
import json, os, sys
o = sys.argv[1]
ref = None
for arm in os.environ["ARMS"].split():
    p = os.path.join(o, f"inkrun-{arm}.json")
    if not os.path.exists(p):
        print(f"{arm}: no report"); continue
    r = json.load(open(p))
    lp = os.path.join(o, f"logits-{arm}.f32")
    lg = open(lp, "rb").read() if os.path.exists(lp) else None
    if ref is None or ref[2] != r["prompt_tokens"]:
        ref = (r["tokens"], lg, r["prompt_tokens"])
    same = (r["tokens"] == ref[0], lg is not None and lg == ref[1])
    if lg is not None and ref[1] is not None and len(lg) == len(ref[1]) and lg != ref[1]:
        import struct
        x = struct.unpack(f"<{len(lg) // 4}f", lg); y = struct.unpack(f"<{len(lg) // 4}f", ref[1])
        same = same + (f"largest logit gap {max(abs(u - v) for u, v in zip(x, y)):.3g}",)
    print(f"{arm}: prefill of {r['prompt_tokens']} tokens {r['seconds_prefill']:.2f} s, "
          f"{r['prompt_tokens'] / r['seconds_prefill']:.0f} tok/s; lattice launches in the prefill {r.get('lattice_launches_prefill')}; "
          f"median step {r['step_seconds_median']:.4f} s; GB at the end {r['memory']['end'].get('used', 0) / 1e9:.3f}; "
          f"moe_rows {r['provenance'].get('moe_rows')}; tokens and logits bytes equal to the first arm of this prompt length: {same}")
    m = r["memory"]
    gb = lambda k: (m.get(k) or {}).get("used", 0) / 1e9
    pk = r.get("peak_probe") or {}
    print(f"    GB after load {gb('after_resident'):.3f}, after the prompt {gb('after_prefill'):.3f}, at the end {gb('end'):.3f}; "
          f"peak probe: prompt {pk.get('prefill', 0) / 1e9:.3f}, whole run {pk.get('whole_run', 0) / 1e9:.3f}")
    pp = r.get("prefill_parts")
    if pp:
        tot = sum(pp.values())
        for k, v in sorted(pp.items(), key=lambda kv: -kv[1]):
            print(f"    {k:26s} {v:7.3f} s  {100 * v / tot:5.1f} %")
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
  --name "inkling-cuda-ttft-$FLAVOR${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
