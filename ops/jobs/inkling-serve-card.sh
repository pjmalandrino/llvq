#!/usr/bin/env bash
# The runtime server on the real Inkling 2 Mini, on one CUDA card, driven through its OpenAI route. It checks the
# server without loading the model on the operator's Mac. One job:
#   controls 1 and 2: `oracle` MATCH on Qwen3-0.6B, then every input the server reads copied to local disk and
#     checked at its pin (config, checkpoint, layer codes, tokenizer, the two int4 tables);
#   `llvq serve --local` in the background on 127.0.0.1, the model loaded once by an empty /api/chat;
#   (a) GET /v1/models; (b) a short chat, not streamed, and its time; (c) a streamed chat;
#   (d) a call of `get_weather(city)`, then the follow-up with the `role: tool` result, on /v1/chat/completions;
#       the same pair on /api/chat, whose answer carries `prompt_eval_cached_count`, to read the reuse;
#   (e) a first prompt of about 9,000 tokens, its time to first token, the same request again, then a follow-up
#       turn on /api/chat to read what the cache kept;
#   (f) nvidia-smi before the load, after it, after the long prompt and at the end.
# Every request and answer is saved raw under the output directory. summary.txt holds one line a check, PASS or
# FAIL, with its seconds. The job exits non-zero when a correctness check fails; reading checks change nothing.
# The requests and the checks are ops/inkling/serve_checks.py, copied into the job's command.
#
#   DRY_RUN=1 bash ops/jobs/inkling-serve-card.sh              parses, checks the image lists, launches nothing
#   PRINT_JOB=1 DRY_RUN=1 bash ops/jobs/inkling-serve-card.sh  prints the job body
#   TEXTONLY_SHA=.. CONFIG_SHA=.. MANIFEST_SHA=.. TOKENIZER_SHA=.. IMAGE_SHA=.. bash ops/jobs/inkling-serve-card.sh
#
# FLAVOR=l40sx1 (default) or l4x1. Preregistered: PREREG below, signed and stamped before this.
# Cost, estimated: about 15 minutes, $0.45 on l40sx1; 30-minute timeout.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=${PREREG:-proofs/preregistration-inkling-serve-card-2026-11-24.md}
FLAVOR=${FLAVOR:-l40sx1}
OUT_NAME=inkling-serve-card-$FLAVOR-2026-11-24
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
IN=/out/inkling2-mini-inputs-2026-11-22
O=/out/$OUT_NAME${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
TIMEOUT=${TIMEOUT:-30m}
LAYERS=24
# The two int4 g64 tables of the sealed file, as ops/inkling/stage-gsm8k-card.sh printed them (the GSM8K jobs ran on
# these values).
EMBED_TABLE_SHA=b7e9979f589a8a845215eeb8fecf5ed5a22e7685072b7bbdbddd45fee0bc027e
UNEMBED_TABLE_SHA=cb26ce0c6145d35db52886b378185910b4adaf3de8024cb28e1e6f0062abcbb7
# Pinned by the staging scripts, which print these values; refused below while a pin is unset.
TEXTONLY_SHA=${TEXTONLY_SHA:-}
CONFIG_SHA=${CONFIG_SHA:-}
MANIFEST_SHA=${MANIFEST_SHA:-}
TOKENIZER_SHA=${TOKENIZER_SHA:-}
TEXTONLY_BYTES=3218968280
CHECKS=ops/inkling/serve_checks.py

if [ "$DRY" != 1 ]; then
  for v in TEXTONLY_SHA CONFIG_SHA MANIFEST_SHA TOKENIZER_SHA; do
    [ -n "${!v}" ] || { echo "refused: $v unset; ops/inkling/stage-first-card.sh and stage-gsm8k-card.sh print it" >&2; exit 1; }
  done
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  # every input in the bucket, by name and size, before a card is rented
  LS=$(hf buckets ls "hf://buckets/$BUCKET/${IN#/out/}/" 2>/dev/null || true)
  for f in config.json model.safetensors codes.sha256 tokenizer.json; do
    echo "$LS" | grep -q "/$f\$" || { echo "refused: $f absent from ${IN#/out/}/" >&2; exit 1; }
  done
  echo "$LS" | awk -v n=/model.safetensors 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' \
    | grep -qx "$TEXTONLY_BYTES" || { echo "refused: model.safetensors is not $TEXTONLY_BYTES B in the bucket" >&2; exit 1; }
  NT=$(hf buckets ls "hf://buckets/$BUCKET/${IN#/out/}/layers/" 2>/dev/null | grep -c '/L[0-9][0-9]\.tar$' || true)
  [ "$NT" = "$LAYERS" ] || { echo "refused: $NT layer archives in the bucket, $LAYERS expected" >&2; exit 1; }
  NTB=$(hf buckets ls "hf://buckets/$BUCKET/${IN#/out/}/tables/" 2>/dev/null | grep -c 'embed\.q4g64\.llvqraw$' || true)
  [ "$NTB" = 2 ] || { echo "refused: $NTB of the 2 sealed tables in ${IN#/out/}/tables/" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of the rebuild that carries llvq serve --local>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "bucket inputs present; image $NOW"
fi

PRE=$(printf 'O=%q\nIN=%q\nLAYERS=%q\nTEXTONLY_SHA=%q\nCONFIG_SHA=%q\nMANIFEST_SHA=%q\nTOKENIZER_SHA=%q\nEMBED_TABLE_SHA=%q\nUNEMBED_TABLE_SHA=%q\nTEXTONLY_BYTES=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$IN" "$LAYERS" "$TEXTONLY_SHA" "$CONFIG_SHA" "$MANIFEST_SHA" "$TOKENIZER_SHA" "$EMBED_TABLE_SHA" \
  "$UNEMBED_TABLE_SHA" "$TEXTONLY_BYTES")
# The helper travels in the command: C is its path inside the container.
PRE="$PRE"$'\n'"C=/tmp/serve_checks.py"$'\n'"cat > \"\$C\" <<'SERVECHECKS'"$'\n'"$(cat "$CHECKS")"$'\n'"SERVECHECKS"

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
command -v llvq > /dev/null || { echo 'STOP: no llvq in the image (rebuild with ops/Dockerfile.cuda)' >&2; exit 1; }
command -v curl > /dev/null || { echo 'STOP: no curl in the image' >&2; exit 1; }
cp "$C" "$O/serve_checks.py"
export O MODEL=inkling2-mini:local
nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.used,driver_version --format=csv,noheader | tee "$O/gpu-before.txt"
df -h "${SCRATCH:-/scratch}" /tmp | tee "$O/disk.txt"
echo '== control 1: oracle on this backend =='
timeout 5m oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle.txt" | tail -2
grep -q MATCH "$O/oracle.txt"
echo '== control 2: inputs to local disk, at their pins =='
L=${SCRATCH:-/scratch}/inkling
mkdir -p "$L/layers" "$L/tables"
date -u
cp "$IN/config.json" "$IN/model.safetensors" "$IN/codes.sha256" "$IN/tokenizer.json" "$L/"
cp "$IN/tables/embed.q4g64.llvqraw" "$IN/tables/unembed.q4g64.llvqraw" "$L/tables/"
test "$(stat -c %s "$L/model.safetensors")" = "$TEXTONLY_BYTES"
for i in $(seq 0 $((LAYERS - 1))); do tar -xf "$IN/layers/L$(printf %02d "$i").tar" -C "$L/layers"; done
date -u
test "$(sha256sum "$L/config.json" | cut -d' ' -f1)" = "$CONFIG_SHA"
test "$(sha256sum "$L/model.safetensors" | cut -d' ' -f1)" = "$TEXTONLY_SHA"
test "$(sha256sum "$L/codes.sha256" | cut -d' ' -f1)" = "$MANIFEST_SHA"
test "$(sha256sum "$L/tokenizer.json" | cut -d' ' -f1)" = "$TOKENIZER_SHA"
test "$(sha256sum "$L/tables/embed.q4g64.llvqraw" | cut -d' ' -f1)" = "$EMBED_TABLE_SHA"
test "$(sha256sum "$L/tables/unembed.q4g64.llvqraw" | cut -d' ' -f1)" = "$UNEMBED_TABLE_SHA"
(cd "$L/layers" && sha256sum --quiet -c ../codes.sha256) 2>&1 | tee "$O/codes-check.txt"
test "$(grep -c . "$L/codes.sha256")" = "$((LAYERS * 129))"
sha256sum "$L/config.json" "$L/model.safetensors" "$L/codes.sha256" "$L/tokenizer.json" \
  "$L/tables/embed.q4g64.llvqraw" "$L/tables/unembed.q4g64.llvqraw" | tee "$O/files.sha256"
date -u

echo '== the server =='
cat > "$L/inkling.json" <<EOF
{
  "name": "$MODEL",
  "family": "inkling2",
  "config": "config.json",
  "checkpoint": "model.safetensors",
  "layers": "layers",
  "tokenizer": "tokenizer.json",
  "tables": "tables",
  "context": 32768
}
EOF
cp "$L/inkling.json" "$O/inkling.json"
llvq -v | tee "$O/llvq-version.txt"
H=http://127.0.0.1:11435
env LLVQ_HOME="$L/home" LLVQ_HOST=127.0.0.1:11435 LLVQ_KEEP_ALIVE=-1 \
  llvq serve --local "$L/inkling.json" > "$O/server.log" 2>&1 &
SRV=$!
trap 'kill "$SRV" 2>/dev/null || true' EXIT
UP=0
for _ in $(seq 1 60); do
  if curl -sS --max-time 2 "$H/api/version" > "$O/version.json" 2>/dev/null; then UP=1; break; fi
  kill -0 "$SRV" 2>/dev/null || break
  sleep 1
done
[ "$UP" = 1 ] || { echo 'STOP: the server did not answer /api/version within 60 s' >&2; tail -20 "$O/server.log" >&2; exit 1; }
cat "$O/version.json"; echo

# post <name> <route> <seconds>: the body $O/<name>.req.json, the answer $O/<name>.json, curl's times $O/<name>.time
post() {
  curl -sS --max-time "$3" -H 'Content-Type: application/json' --data-binary @"$O/$1.req.json" \
    -o "$O/$1.json" -w '%{http_code} %{time_starttransfer} %{time_total}' "$H$2" > "$O/$1.time" 2> "$O/$1.curl-err" || true
}
# stream <name> <seconds>: an OpenAI stream, every line timestamped from just before curl starts
stream() {
  local t0
  t0=$(date +%s.%N)
  curl -sS -N --max-time "$2" -H 'Content-Type: application/json' --data-binary @"$O/$1.req.json" \
    "$H/v1/chat/completions" 2> "$O/$1.curl-err" | python3 -I "$C" sse "$t0" "$O/$1.sse.txt" "$O/$1.sse.json" || true
}
req() { python3 -I "$C" req "$@" > "$O/$1.req.json"; }
check() { python3 -I "$C" check "$@" || true; }
smi() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$O/$1"; }

echo '== (a) GET /v1/models =='
curl -sS --max-time 10 "$H/v1/models" > "$O/a-models.json" 2> "$O/a-models.curl-err" || true
check a "$O/a-models.json"

echo '== load: an empty /api/chat =='
printf '{"model": "%s", "messages": [], "stream": false, "keep_alive": -1}' "$MODEL" > "$O/load.req.json"
post load /api/chat 1200
check load "$O/load.json" "$O/load.time"
curl -sS --max-time 10 "$H/api/ps" > "$O/ps-loaded.json" || true
check ps-loaded "$O/ps-loaded.json" "$O/server.log"
smi gpu-loaded.txt

echo '== (b) a short chat, not streamed =='
req b; post b /v1/chat/completions 120
check b "$O/b.json" "$O/b.time"

echo '== (c) a streamed chat =='
req c; stream c 120
check c "$O/c.sse.json"

echo '== (d) a tool call and its follow-up, OpenAI route =='
req d1; post d1 /v1/chat/completions 300
check d1 "$O/d1.json" "$O/d1.time"
if req d2 "$O/d1.json"; then stream d2 300; fi
check d2 "$O/d2.sse.json"
echo '== (d) the same pair on /api/chat, which reports what the cache held =='
req n1; post n1 /api/chat 300
check n1 "$O/n1.json" "$O/n1.time"
if req n2 "$O/n1.json"; then post n2 /api/chat 300; fi
check n2 "$O/n2.json" "$O/n2.time" "$O/n1.json"

echo '== (e) a first prompt of about 9,000 tokens, then the same again, then a follow-up =='
req e1; stream e1 900
check e1 "$O/e1.sse.json"
smi gpu-after-long.txt
cp "$O/e1.req.json" "$O/e2.req.json"; stream e2 900
check e2 "$O/e2.sse.json" /dev/null "$O/e1.sse.json"
if req e3 "$O/e1.sse.json"; then post e3 /api/chat 900; fi
check e3 "$O/e3.json" "$O/e3.time" "$O/e1.sse.json"

echo '== (f) the end =='
curl -sS --max-time 10 "$H/api/ps" > "$O/ps-end.json" || true
check ps-end "$O/ps-end.json" "$O/server.log"
smi gpu-after.txt
check gpu "$O/gpu-before.txt" "$O/gpu-loaded.txt" "$O/gpu-after-long.txt" "$O/gpu-after.txt"
kill "$SRV" 2>/dev/null || true
wait "$SRV" 2>/dev/null || true
tail -5 "$O/server.log"
ls -la "$O"
python3 -I "$C" final
JOB

if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  if [ "${PRINT_JOB:-0}" = 1 ]; then printf '%s\n' "$JOBSCRIPT"; exit 0; fi
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; nothing launched" >&2
  python3 -I -c 'import ast, sys; ast.parse(open(sys.argv[1]).read())' "$CHECKS" && echo "DRY_RUN: $CHECKS parses" >&2
  grep -q -- '-p llvq-runtime --features cuda --bin llvq' ops/Dockerfile.cuda \
    && grep -qE '/src/target/release/llvq( |$)' ops/Dockerfile.cuda \
    || { echo "DRY_RUN: llvq is not in both lists of ops/Dockerfile.cuda" >&2; exit 1; }
  python3 -I ops/check_image_lists.py
  exit 0
fi

uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor "$FLAVOR" --any-flavor --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "inkling-serve-card-$FLAVOR${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
