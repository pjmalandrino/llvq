#!/usr/bin/env bash
# GSM8K-200 for Inkling 2 Mini through this engine on one NVIDIA card: the text model, its routed experts resident in
# lattice codes (`inkgsm run --device cuda --require-layers 24`), the protocol of `llvq-llm/src/bin/inkgsm.rs`: the
# first 200 test problems in parquet order (`/usr/local/share/llvq/spec-gsm8k-200.jsonl`, checked by sha256), the
# checkpoint's chat template at reasoning effort `none`, greedy, at most 1,536 new tokens, a completion cut by the cap
# counted wrong. One job:
#   (0) controls: the card, no LLVQ_* in the environment, `oracle` MATCH on Qwen3-0.6B, every input at its pin after
#       it is copied from the bucket to the container's disk (the first-card inputs plus tokenizer.json);
#   (1) `inkgsm run`: the dump is written on local disk, line by line, and copied to the bucket every COPY_EVERY
#       seconds and at the end; `--deadline-seconds` stops it cleanly before the job's timeout, every written line
#       whole. RESUME_FROM=<bucket dir of an earlier run> starts from that run's dump: only the missing problems run;
#   (2) `inkgsm score` on the dump, nvidia-smi, the logs to the bucket.
# Output inkling-gsm8k-card-<date>.
#
#   DRY_RUN=1 bash ops/jobs/inkling-gsm8k-card.sh          parses, checks the image lists and the pins it can, launches nothing
#   PRINT_JOB=1 DRY_RUN=1 bash ops/jobs/inkling-gsm8k-card.sh   prints the job body
#   TEXTONLY_SHA=.. CONFIG_SHA=.. MANIFEST_SHA=.. TOKENIZER_SHA=.. IMAGE_SHA=.. bash ops/jobs/inkling-gsm8k-card.sh
#
# Inputs: the bucket directory of ops/inkling/stage-first-card.sh (config.json, model.safetensors, codes.sha256,
# layers/Lxx.tar) and tokenizer.json, staged by ops/inkling/stage-gsm8k-card.sh, which prints TOKENIZER_SHA.
# Preregistered: PREREG below, to write, sign and stamp before this; it is not written by this script.
#
# Cost model (estimated until the first card run times a step): a problem costs P + n * t, with t the median step
# seconds of the first card run, n the generated tokens and P the prefill (the prompts are 53 to 171 tokens, 93 on average). The
# whole run is about 200 * (P + n_mean * t) + 10 min of setup; the cap bounds n at 1,536. Set DEADLINE from that
# estimate and TIMEOUT 20 min above it: a run that does not finish keeps its dump and is resumed by RESUME_FROM.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

DATE=${DATE:-$(date -u +%Y-%m-%d)}
PREREG=${PREREG:-proofs/preregistration-inkling-gsm8k-card-$DATE.md}
OUT_NAME=inkling-gsm8k-card-$DATE
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
IN=/out/inkling2-mini-inputs-2026-11-22
O=/out/$OUT_NAME${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
FLAVOR=${FLAVOR:-l4x1}
# Seconds from inkgsm's start after which no new problem begins and the one under way is dropped.
DEADLINE=${DEADLINE:-14400}
TIMEOUT=${TIMEOUT:-4h30m}
COPY_EVERY=${COPY_EVERY:-120}
LIMIT=${LIMIT:-200}
MAX_NEW=${MAX_NEW:-1536}
EFFORT=${EFFORT:-none}
LAYERS=24
RESUME_FROM=${RESUME_FROM:-}
TEXTONLY_SHA=${TEXTONLY_SHA:-}
CONFIG_SHA=${CONFIG_SHA:-}
MANIFEST_SHA=${MANIFEST_SHA:-}
TOKENIZER_SHA=${TOKENIZER_SHA:-}
TEXTONLY_BYTES=3218968280
# The problems file the image carries, pinned from this tree.
PROBLEMS_SHA=$( (sha256sum ops/data/spec-gsm8k-200.jsonl 2>/dev/null || shasum -a 256 ops/data/spec-gsm8k-200.jsonl) | cut -d' ' -f1)

# The protocol is the comparison's contract: another value is another arm, refused here unless named.
if [ "${PROTOCOL_CHANGE:-0}" != 1 ]; then
  [ "$LIMIT" = 200 ] && [ "$MAX_NEW" = 1536 ] && [ "$EFFORT" = none ] \
    || { echo "refused: LIMIT/MAX_NEW/EFFORT differ from 200/1536/none; PROTOCOL_CHANGE=1 to run another arm" >&2; exit 1; }
fi
[ "$DEADLINE" -gt 0 ] 2>/dev/null || { echo "refused: DEADLINE=$DEADLINE is not a positive count of seconds" >&2; exit 1; }

if [ "$DRY" != 1 ]; then
  for v in TEXTONLY_SHA CONFIG_SHA MANIFEST_SHA TOKENIZER_SHA; do
    [ -n "${!v}" ] || { echo "refused: $v unset; the staging scripts print it" >&2; exit 1; }
  done
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  LS=$(hf buckets ls "hf://buckets/$BUCKET/${IN#/out/}/" 2>/dev/null || true)
  for f in config.json model.safetensors codes.sha256 tokenizer.json; do
    echo "$LS" | grep -q "/$f\$" || { echo "refused: $f absent from ${IN#/out/}/" >&2; exit 1; }
  done
  echo "$LS" | awk -v n=/model.safetensors 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' \
    | grep -qx "$TEXTONLY_BYTES" || { echo "refused: model.safetensors is not $TEXTONLY_BYTES B in the bucket" >&2; exit 1; }
  NT=$(hf buckets ls "hf://buckets/$BUCKET/${IN#/out/}/layers/" 2>/dev/null | grep -c '/L[0-9][0-9]\.tar$' || true)
  [ "$NT" = "$LAYERS" ] || { echo "refused: $NT layer archives in the bucket, $LAYERS expected" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  if [ -n "$RESUME_FROM" ]; then
    hf buckets ls "hf://buckets/$BUCKET/$RESUME_FROM/" 2>/dev/null | grep -q '/inkgsm.jsonl$' \
      || { echo "refused: no inkgsm.jsonl under $RESUME_FROM/" >&2; exit 1; }
  fi
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of the rebuild that carries inkgsm>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "bucket inputs present; image $NOW"
fi

PRE=$(printf 'O=%q\nIN=%q\nLAYERS=%q\nLIMIT=%q\nMAX_NEW=%q\nEFFORT=%q\nDEADLINE=%q\nCOPY_EVERY=%q\nRESUME=%q\nTEXTONLY_SHA=%q\nCONFIG_SHA=%q\nMANIFEST_SHA=%q\nTOKENIZER_SHA=%q\nPROBLEMS_SHA=%q\nTEXTONLY_BYTES=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$IN" "$LAYERS" "$LIMIT" "$MAX_NEW" "$EFFORT" "$DEADLINE" "$COPY_EVERY" "${RESUME_FROM:+/out/$RESUME_FROM}" \
  "$TEXTONLY_SHA" "$CONFIG_SHA" "$MANIFEST_SHA" "$TOKENIZER_SHA" "$PROBLEMS_SHA" "$TEXTONLY_BYTES")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
command -v inkgsm > /dev/null || { echo 'STOP: no inkgsm in the image (rebuild with ops/Dockerfile.cuda)' >&2; exit 1; }
P=${PROBLEMS:-/usr/local/share/llvq/spec-gsm8k-200.jsonl}
test "$(sha256sum "$P" | cut -d' ' -f1)" = "$PROBLEMS_SHA" || { echo "STOP: $P is not the pinned problems file" >&2; exit 1; }
nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.used,driver_version --format=csv,noheader | tee "$O/gpu-before.txt"
df -h "${SCRATCH:-/scratch}" /tmp | tee "$O/disk.txt"
echo '== control 1: oracle on this backend =='
timeout 5m oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle.txt" | tail -2
grep -q MATCH "$O/oracle.txt"
echo '== control 2: inputs to local disk, at their pins =='
L=${SCRATCH:-/scratch}/inkling
mkdir -p "$L/layers"
date -u
cp "$IN/config.json" "$IN/model.safetensors" "$IN/codes.sha256" "$IN/tokenizer.json" "$L/"
test "$(stat -c %s "$L/model.safetensors")" = "$TEXTONLY_BYTES"
for i in $(seq 0 $((LAYERS - 1))); do tar -xf "$IN/layers/L$(printf %02d "$i").tar" -C "$L/layers"; done
test "$(sha256sum "$L/config.json" | cut -d' ' -f1)" = "$CONFIG_SHA"
test "$(sha256sum "$L/model.safetensors" | cut -d' ' -f1)" = "$TEXTONLY_SHA"
test "$(sha256sum "$L/codes.sha256" | cut -d' ' -f1)" = "$MANIFEST_SHA"
test "$(sha256sum "$L/tokenizer.json" | cut -d' ' -f1)" = "$TOKENIZER_SHA"
(cd "$L/layers" && sha256sum --quiet -c ../codes.sha256) 2>&1 | tee "$O/codes-check.txt"
test "$(grep -c . "$L/codes.sha256")" = "$((LAYERS * 129))"
sha256sum "$L/config.json" "$L/model.safetensors" "$L/codes.sha256" "$L/tokenizer.json" "$P" | tee "$O/files.sha256"
D="$L/inkgsm.jsonl"
if [ -n "$RESUME" ]; then cp "$RESUME/inkgsm.jsonl" "$D"; echo "resuming from $RESUME: $(($(grep -c . "$D") - 1)) lines"; fi
date -u
echo '== (1) inkgsm run: 24 layers resident, GSM8K =='
( while sleep "$COPY_EVERY"; do [ -f "$D" ] && cp "$D" "$O/inkgsm.jsonl.part" && mv "$O/inkgsm.jsonl.part" "$O/inkgsm.jsonl"; done ) &
COPIER=$!
set +e
timeout $((DEADLINE + 900)) inkgsm run --config "$L/config.json" --checkpoint "$L/model.safetensors" \
  --layers "$L/layers" --tokenizer "$L/tokenizer.json" --device cuda --require-layers "$LAYERS" \
  --problems "$P" --limit "$LIMIT" --max-new "$MAX_NEW" --effort "$EFFORT" --deadline-seconds "$DEADLINE" \
  --dump "$D" > "$O/inkgsm.txt" 2>&1
RC=$?
set -e
kill "$COPIER" 2>/dev/null || true
cp "$D" "$O/inkgsm.jsonl"
tail -25 "$O/inkgsm.txt"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | tee "$O/gpu-after.txt"
echo "inkgsm exit $RC (0 finished, 3 deadline: resume with RESUME_FROM=${O#/out/})" | tee "$O/summary.txt"
echo "rows in the dump: $(($(grep -c . "$O/inkgsm.jsonl") - 1)) lines after the header, the trailer included when finished" | tee -a "$O/summary.txt"
if [ "$RC" = 0 ]; then inkgsm score "$O/inkgsm.jsonl" 2>&1 | tee -a "$O/summary.txt"; fi
python3 - "$O/inkgsm.jsonl" <<'PY' | tee -a "$O/summary.txt"
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()][1:]
rows = [r for r in rows if "index" in r]
if rows:
    steps = [r["step_s_median"] for r in rows if r.get("step_s_median")]
    gen = [r["n_gen"] for r in rows]
    print(f"{len(rows)} problems: median step {statistics.median(steps):.4f} s, generated mean {statistics.mean(gen):.1f} "
          f"max {max(gen)}, cut by the cap {sum(r['stop'] == 'cap' for r in rows)}, "
          f"seconds a problem mean {statistics.mean(r['prefill_s'] + r['decode_s'] for r in rows):.1f}, "
          f"turns refused by the parser {sum(r.get('turn') != 'ok' for r in rows)}")
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
  grep -q -- '--bin inkgsm' ops/Dockerfile.cuda && grep -q '/src/target/release/inkgsm' ops/Dockerfile.cuda \
    || { echo "DRY_RUN: inkgsm is not in both lists of ops/Dockerfile.cuda" >&2; exit 1; }
  grep -q 'spec-gsm8k-200.jsonl /usr/local/share/llvq/spec-gsm8k-200.jsonl' ops/Dockerfile.cuda \
    || { echo "DRY_RUN: the image does not copy the problems file" >&2; exit 1; }
  python3 -I ops/check_image_lists.py
  echo "DRY_RUN: problems sha256 $PROBLEMS_SHA, deadline $DEADLINE s, timeout $TIMEOUT, flavor $FLAVOR" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor "$FLAVOR" --any-flavor --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "inkling-gsm8k-card${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
