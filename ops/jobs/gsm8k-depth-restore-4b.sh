#!/usr/bin/env bash
# GSM8K on three dense f16 versions of the Qwen3-4B, in vLLM: the f16 export of the sealed 4B as it is (control),
# with the seven projections of layers 25 to 35 restored to FP16 (late), with those of layers 0 to 24 restored
# (early). The question: do the blocks after the answer write matter for generation as little as for MMLU
# (docs/mesures/depth-restore-4b-2026-11-14.txt: +0.43 late, +6.12 early)?
# Preregistered: proofs/preregistration-gsm8k-depth-restore-4b-2026-11-14.md, which must be stamped, and must name
# the sha256 of every script below and of this one, before the job launches.
#
# One vLLM process an arm, in the image of the 2026-09-26 wave (ops/jobs/gsm8k-vllm.sh): zero-shot, reasoning block
# pre-filled empty, greedy, max_new 1,024, the 1,319 test problems at the pinned dataset revision. Each arm's
# prompts are checked by ops/gsm8k_vllm.py's own --check against the served dump of 2026-09-26 before it generates.
# Arms in the order control, late, early: a timeout cuts the least needed one. One arm on disk at a time:
# build (ops/hybrid_checkpoint.py, which verifies by reading the written file back), generate, delete. The disk holds
# the staged export, the FP16 snapshot and one arm, about 24 GB.
#
# Inputs travel through the bucket folder $IN (the `upload` subcommand), each checked by sha256 inside the job
# against the committed files.
#
# Cost, estimated on l40sx1 at $1.80/h, from the 2026-09-26 vLLM job (two arms in 319 s, downloads included) and the
# staging of the 2026-11-13 and 2026-11-14 jobs (export copy, shard download and hashes in about 104 s with the load):
#   setup (pip, selftest, export copy, FP16 download) 2 to 3 min; builds: control 1 min, late and early 1.5 to 2.5
#   min each; three vLLM arms 2.5 to 3.5 min each (process, load, compile, 70 s of generation). 13 to 19 min,
#   $0.39 to $0.57. Timeout 35m, ceiling $1.05.
#
#   bash ops/jobs/gsm8k-depth-restore-4b.sh upload      the committed inputs into the bucket
#   DRY_RUN=1 bash ops/jobs/gsm8k-depth-restore-4b.sh   prints the job script, parses it, launches nothing
#   bash ops/jobs/gsm8k-depth-restore-4b.sh             launches
#   bash ops/jobs/gsm8k-depth-restore-4b.sh pair        after the job, local: fetches the dumps, runs gsm8kpair
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-gsm8k-depth-restore-4b-2026-11-14.md
BUCKET=Pier-Jean/jobs-artifacts
HYB=ops/hybrid_checkpoint.py
LVO=ops/layer_vs_output.py
GEN=ops/gsm8k_vllm.py
LOC=ops/gsm8k_vllm_local.py
WAVE=docs/mesures/gsm8k-wave1-2026-09-26-brut
SERVED=$WAVE/run1-served/gsm8k-4b-served.jsonl
F16VLLM=$WAVE/run5-vllm/gsm8k-4b-f16-vllm.jsonl
SCRIPTS="$HYB $LVO $GEN $LOC"
INPUTS="$SCRIPTS $SERVED"
IN=gsm8k-depth-restore-4b-inputs-2026-11-14
OUT=gsm8k-depth-restore-4b-2026-11-14
EXPORT_DIR=sealed-4b-export-2026-09-23
IMAGE=vllm/vllm-openai:v0.26.0
DIGEST=sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52
DATASET_REV=740312add88f781978c0658806c59bc2815b9866
BASE=Qwen/Qwen3-4B
BASE_REV=1cfa9a7208912126459214e8b04321603b3df60c
FLAVOR=l40sx1
USD_H=1.80
TIMEOUT=${TIMEOUT:-35m}
EST="13 to 19 min, \$0.39 to \$0.57"
DRY=${DRY_RUN:-0}

refuse() { echo "refused: $*" >&2; exit 1; }
committed() {
  for f in "$@"; do
    git ls-files --error-unmatch "$f" >/dev/null 2>&1 || refuse "$f is not committed"
    git diff --quiet HEAD -- "$f" || refuse "$f differs from HEAD"
  done
}
remote_size() {  # size of $IN/<basename> in the bucket, empty if absent
  hf buckets ls "hf://buckets/$BUCKET/$IN/" 2>/dev/null \
    | awk -v n="/$1" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true
}
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }

if [ "${1:-}" = upload ]; then
  committed $INPUTS
  for f in $INPUTS; do hf buckets cp "$f" "hf://buckets/$BUCKET/$IN/$(basename "$f")"; done
  hf buckets ls "hf://buckets/$BUCKET/$IN/"
  exit 0
fi

if [ "${1:-}" = pair ]; then
  # `gsm8kpair <A> <B>` re-grades every row of both dumps with the Rust rules, refuses two dumps whose prompt
  # streams differ, and prints "A - B" with its 95 % interval and McNemar's exact p: the difference is taken from
  # the FIRST argument. So every line below reads "first minus second".
  #   late - control       what restoring layers 25 to 35 buys on GSM8K
  #   early - control      what restoring layers 0 to 24 buys
  #   control - served     the same sealed file, dense f16 in vLLM against the served kernel of 2026-09-26: a check
  #                        of the path, expected near zero (engines and arithmetic differ, so not zero)
  #   early - FP16 vLLM    what the late blocks still cost after the early restore, against FP16 in the same engine
  PAIR=${PAIR_DIR:-target/$OUT}
  BIN=target/release/gsm8kpair
  test -x "$BIN" || refuse "$BIN is missing: cargo build --release -p llvq-llm --features metal,fast-linalg --bin gsm8kpair"
  committed "$SERVED" "$F16VLLM"
  mkdir -p "$PAIR"
  for arm in control late early; do
    hf buckets cp "hf://buckets/$BUCKET/$OUT/gsm8k-4b-depth-$arm.jsonl" "$PAIR/gsm8k-4b-depth-$arm.jsonl"
  done
  "$BIN" "$PAIR/gsm8k-4b-depth-late.jsonl" "$PAIR/gsm8k-4b-depth-control.jsonl" 2>&1 | tee "$PAIR/pair-late-vs-control.txt"
  "$BIN" "$PAIR/gsm8k-4b-depth-early.jsonl" "$PAIR/gsm8k-4b-depth-control.jsonl" 2>&1 | tee "$PAIR/pair-early-vs-control.txt"
  "$BIN" "$PAIR/gsm8k-4b-depth-control.jsonl" "$SERVED" 2>&1 | tee "$PAIR/pair-control-vs-served.txt"
  "$BIN" "$PAIR/gsm8k-4b-depth-early.jsonl" "$F16VLLM" 2>&1 | tee "$PAIR/pair-early-vs-f16vllm.txt"
  ls -la "$PAIR"
  exit 0
fi

[ -z "${1:-}" ] || refuse "usage: $0 [upload | pair]"

HYB_SHA=$(sha "$HYB"); LVO_SHA=$(sha "$LVO"); GEN_SHA=$(sha "$GEN"); LOC_SHA=$(sha "$LOC"); SERVED_SHA=$(sha "$SERVED")
JOB_SHA=$(sha "$0")

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || refuse "$PREREG(.ots) missing"
  # The anchor is the proof: the banner a stamped prereg keeps is not read.
  ots info "$PREREG.ots" 2>/dev/null | grep -q "$(sha "$PREREG")" || refuse "$PREREG.ots does not anchor the sha256 of $PREREG"
  for f in $SCRIPTS; do
    grep -q "$(sha "$f")" "$PREREG" || refuse "$PREREG does not name the sha256 of $f ($(sha "$f"))"
  done
  grep -q "$JOB_SHA" "$PREREG" || refuse "$PREREG does not name this script's sha256 ($JOB_SHA)"
  committed "$0" $INPUTS
  for f in $INPUTS; do
    L=$(stat -f %z "$f"); R=$(remote_size "$(basename "$f")")
    [ "$R" = "$L" ] || refuse "$IN/$(basename "$f") is ${R:-absent} B in the bucket, $L B here; run upload"
  done
  if hf buckets ls "hf://buckets/$BUCKET/$OUT/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    refuse "$OUT/ already exists in the bucket"
  fi
fi

MIN=${TIMEOUT%m}
echo "cost: $FLAVOR at \$$USD_H/h, timeout $TIMEOUT, ceiling \$$(awk -v m="$MIN" -v r="$USD_H" 'BEGIN {printf "%.2f", m / 60 * r}');" \
     "estimated $EST (setup, two builds, three vLLM arms)"
echo "inputs $IN/: $HYB $HYB_SHA, $LVO $LVO_SHA, $GEN $GEN_SHA, $LOC $LOC_SHA, $(basename "$SERVED") $SERVED_SHA"
echo "out $OUT/; image $IMAGE@$DIGEST; job sha256 $JOB_SHA"

PRE=$(printf 'O=%q\nI=%q\nE=%q\nHYB_SHA=%q\nLVO_SHA=%q\nGEN_SHA=%q\nLOC_SHA=%q\nSERVED_SHA=%q\nBASE=%q\nBASE_REV=%q\nDATASET_REV=%q\nexport LLVQ_IMAGE_TAG=%q\nexport LLVQ_IMAGE_DIGEST=%q\nexport HF_HOME=/tmp/hf' \
  "/out/$OUT" "/out/$IN" "/out/$EXPORT_DIR" "$HYB_SHA" "$LVO_SHA" "$GEN_SHA" "$LOC_SHA" "$SERVED_SHA" \
  "$BASE" "$BASE_REV" "$DATASET_REV" "$IMAGE" "$DIGEST")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O" /tmp/w /tmp/stage
stamp() { echo "== $(date -u +%FT%TZ) $*" | tee -a "$O/timeline.txt"; }
stamp start
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt"
{ echo "nproc $(nproc)"; free -g 2>/dev/null | head -2 || true; df -h /tmp | tail -n +2; } | tee -a "$O/gpu.txt"
echo '== the inputs on the mount: sha256 =='
{ echo "$HYB_SHA  $I/hybrid_checkpoint.py"; echo "$LVO_SHA  $I/layer_vs_output.py"; echo "$GEN_SHA  $I/gsm8k_vllm.py";
  echo "$LOC_SHA  $I/gsm8k_vllm_local.py"; echo "$SERVED_SHA  $I/gsm8k-4b-served.jsonl"; } \
  | sha256sum -c - | tee "$O/inputs.sha256"
cp "$I/hybrid_checkpoint.py" "$I/layer_vs_output.py" "$I/gsm8k_vllm.py" "$I/gsm8k_vllm_local.py" /tmp/w/
python3 -c "import vllm; print('vllm', vllm.__version__)" | tee "$O/vllm-version.txt"
# The pinned vLLM image carries no pyarrow (attempt 1 of 2026-09-26 died on it). Same pin as ops/jobs/gsm8k-vllm.sh.
pip install --quiet --no-input pyarrow==25.0.1 2>&1 | tail -2
python3 -c "import pyarrow, torch, safetensors, numpy, huggingface_hub as h; print('pyarrow', pyarrow.__version__, 'torch', torch.__version__, 'safetensors', safetensors.__version__, 'numpy', numpy.__version__, 'huggingface_hub', h.__version__)" | tee "$O/versions.txt"
stamp selftest
python3 /tmp/w/hybrid_checkpoint.py --selftest 2>&1 | tee "$O/selftest.txt"
grep -q '^selftest: GREEN' "$O/selftest.txt" || { echo 'refused: selftest is not green' >&2; exit 1; }
# The export is copied to local disk first: a checkpoint memory-mapped from the mounted bucket has died of SIGBUS.
# Its sha256 is checked by every build against the pin of layer_vs_output.py.
stamp "export copy"
cp "$E/model.safetensors" "$E/config.json" /tmp/stage/
ls -la /tmp/stage
stamp "FP16 snapshot $BASE@$BASE_REV"
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$BASE', revision='$BASE_REV', local_dir='/tmp/fp16', allow_patterns=['model-0000?-of-00003.safetensors',
  'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'])
" 2>&1 | grep -vE 'it/s|%\|' | tail -3
ls -la /tmp/fp16
df -h /tmp | tee -a "$O/disk.txt"
for arm in control late early; do
  stamp "build $arm"
  python3 /tmp/w/hybrid_checkpoint.py --build --arm "$arm" --export /tmp/stage/model.safetensors --fp16-dir /tmp/fp16 \
    --out "/tmp/arm-$arm" --manifest "$O/manifest-$arm.json" 2>&1 | tee "$O/build-$arm.txt"
  grep -q "^arm $arm: GREEN" "$O/build-$arm.txt" || { echo "refused: build $arm is not green" >&2; exit 1; }
  ls -la "/tmp/arm-$arm" | tee -a "$O/build-$arm.txt"
  { echo "after build $arm"; df -h /tmp | tail -n +2; } | tee -a "$O/disk.txt"
  REV=$(python3 -c "import json; print(json.load(open('$O/manifest-$arm.json'))['written_sha256'][:16])")
  stamp "generate $arm (revision field $REV)"
  python3 /tmp/w/gsm8k_vllm_local.py --arm "$arm" --model "/tmp/arm-$arm" --revision "$REV" \
    --dataset-rev "$DATASET_REV" --check "$I/gsm8k-4b-served.jsonl" --out "$O/gsm8k-4b-depth-$arm.jsonl" 2>&1 \
    | tee "$O/gen-$arm.txt" | grep -vE 'it/s|%\|' | tail -12
  grep -q 'prompts checked *1319/1319' "$O/gen-$arm.txt" || { echo "refused: $arm prompts not checked" >&2; exit 1; }
  rm -rf "/tmp/arm-$arm"
  stamp "done $arm"
done
stamp end
echo '== raw kept ==' ; ls -la "$O"
JOB

if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; $FLAVOR, timeout $TIMEOUT; nothing launched" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image "$IMAGE" \
  --flavor "$FLAVOR" --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "$OUT" \
  "$PRE" "$BODY"
