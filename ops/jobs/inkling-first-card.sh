#!/usr/bin/env bash
# Inkling 2 Mini on an NVIDIA card for the first time: the text model in this engine, its routed experts resident in
# lattice codes through the served Tetra kernel (`llvq-llm/src/inkling/resident_cuda.rs`), one L4 (24 GB). One job:
#   (0) controls: the card, no LLVQ_* in the environment, `oracle` MATCH on Qwen3-0.6B (the house rule on a backend),
#       every input at its pinned sha256 after it is copied from the bucket to the container's disk;
#   (1) `inkrun`: dense weights from the text-only file, the experts of the 24 layers resident from their codes
#       (`--require-layers 24`: fewer is a refusal before any token), the 32-token prompt, 32 greedy tokens;
#       the report gives the tokens, the GB the driver reports before load, after the dense load, after the
#       experts and at the end, the prefill and per-step seconds, the lattice launches a token, and the tokens
#       compared with the Mac's resident run (`reference` of the tokens file);
#   (2) nvidia-smi after the run, the report and the logs to the bucket.
# Output inkling-first-card-2026-11-22.
# Preregistered: proofs/preregistration-inkling-first-card-2026-11-22-DRAFT.md, to fill, sign and stamp before this (the
# stamped file drops -DRAFT from its name; PREREG below names the stamped one).
# The image must carry the commit with `inkrun` and `inkling/` (a rebuild); ops/Dockerfile.cuda builds and copies it.
# Inputs: staged in the bucket by ops/inkling/stage-first-card.sh (operator's token), listed with sizes there.
#
#   DRY_RUN=1 bash ops/jobs/inkling-first-card.sh                     parses, checks the image lists, launches nothing
#   bash ops/inkling/first-card-sim.sh                                runs the job body on the Mac on fake inputs
#   IMAGE_SHA=<space sha> bash ops/jobs/inkling-first-card.sh         launches, after the operator's go
#
# Cost (l4x1 $0.80/h), estimated: image pull and start about 3 min, oracle about 1 min, inputs copied (9.45 GB) and
# checked about 4 min, load about 2 min, prefill and 31 steps under 1 min: about 11 min, $0.15. Timeout 30m: $0.40 at
# worst.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-inkling-first-card-2026-11-22.md
OUT_NAME=inkling-first-card-2026-11-22
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
echo '== (1) inkrun: 24 layers resident, 32 greedy tokens =='
set +e
timeout 15m inkrun --config "$L/config.json" --checkpoint "$L/model.safetensors" --layers "$L/layers" \
  --tokens "$L/tokens.json" --device cuda --n "$NTOK" --require-layers "$LAYERS" --out "$O/inkrun.json" \
  > "$O/inkrun.txt" 2>&1
RC=$?
set -e
tail -20 "$O/inkrun.txt"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | tee "$O/gpu-after.txt"
echo "inkrun exit $RC" | tee "$O/summary.txt"
if [ -f "$O/inkrun.json" ]; then
  python3 - "$O/inkrun.json" <<'PY' | tee -a "$O/summary.txt"
import json, sys
r = json.load(open(sys.argv[1]))
gb = lambda m: f"{m['used'] / 1e9:.3f}" if isinstance(m, dict) and "used" in m else "n/a"
mem = r["memory"]
print(f"resident layers {len(r['resident_layers'])}: {r['resident_layers']}")
print(f"GB used (driver): before {gb(mem['before_load'])}, after dense {gb(mem['after_dense'])}, "
      f"after experts {gb(mem['after_resident'])}, end {gb(mem['end'])}")
print(f"load: dense {r['seconds_dense_load']:.1f} s, experts {r['seconds_resident_load']:.1f} s; "
      f"prefill of {r['prompt_tokens']} tokens {r['seconds_prefill']:.2f} s")
st = r["step_seconds"]
print(f"steps: {len(st)}, median {r['step_seconds_median']:.4f} s [{min(st):.4f}, {max(st):.4f}], "
      f"{r['tokens_per_second_median_step']:.2f} tok/s at the median step")
print(f"lattice launches: prefill {r['lattice_launches_prefill']}, a step {sorted(set(r['lattice_launches_per_step']))}")
print(f"tokens equal to the Mac's resident run: {r['all_equal']} (equal prefix {r['equal_prefix']} of {r['new_tokens']}); "
      f"min top-2 margin {r['min_margin']:.4g}")
print("tokens", r["tokens"])
PY
fi
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
  --name "inkling-first-card${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
