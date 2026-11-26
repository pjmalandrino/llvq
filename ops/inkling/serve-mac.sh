#!/usr/bin/env bash
# The runtime server on the real Inkling 2 Mini, on this Mac (Metal): the checks of ops/jobs/inkling-serve-card.sh,
# requests and readings from ops/inkling/serve_checks.py unchanged, the backend it expects set to metal. Run it under
# ~/.llvq-guard/heavy.sh: it loads the model and uses the GPU.
#
#   bash ops/inkling/serve-mac.sh <llvq binary> <inkling.json of the model> <output directory>
#
# The model file is read in place; LLVQ_HOME is a fresh directory under the output, so the store is not written.
# Memory: there is no nvidia-smi on a Mac. `footprint` of the server process (its physical footprint, Metal buffers
# included) and /api/ps `size_vram` stand in for it at the same four points.
set -euo pipefail
LLVQ=${1:?llvq binary}; MODEL_FILE=${2:?inkling.json}; O=${3:?output directory}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
C=$REPO/ops/inkling/serve_checks.py
mkdir -p "$O"
[ -z "$(ls -A "$O")" ] || { echo "refused: $O is not empty" >&2; exit 1; }
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the environment' >&2; exit 1; fi
cp "$C" "$O/serve_checks.py"
MODEL=$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["name"])' "$MODEL_FILE")
export O MODEL BACKEND=metal
cp "$MODEL_FILE" "$O/inkling.json"
"$LLVQ" -v | tee "$O/llvq-version.txt"
H=http://127.0.0.1:11436
env LLVQ_HOME="$O/home" LLVQ_HOST=127.0.0.1:11436 LLVQ_KEEP_ALIVE=-1 \
  "$LLVQ" serve --local "$MODEL_FILE" > "$O/server.log" 2>&1 &
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

post() {
  curl -sS --max-time "$3" -H 'Content-Type: application/json' --data-binary @"$O/$1.req.json" \
    -o "$O/$1.json" -w '%{http_code} %{time_starttransfer} %{time_total}' "$H$2" > "$O/$1.time" 2> "$O/$1.curl-err" || true
}
stream() {
  local t0
  t0=$(python3 -I -c 'import time; print(time.time())')
  curl -sS -N --max-time "$2" -H 'Content-Type: application/json' --data-binary @"$O/$1.req.json" \
    "$H/v1/chat/completions" 2> "$O/$1.curl-err" | python3 -I "$C" sse "$t0" "$O/$1.sse.txt" "$O/$1.sse.json" || true
}
req() { python3 -I "$C" req "$@" > "$O/$1.req.json"; }
check() { python3 -I "$C" check "$@" || true; }
mem() {
  { date '+%F %T'; footprint -p "$SRV" 2>&1 | grep -iE 'phys_footprint|Footprint|IOAccelerator|TOTAL' | head -8
    curl -sS --max-time 10 "$H/api/ps" 2>/dev/null; echo; } > "$O/$1"
}

mem mem-before.txt
echo '== (a) GET /v1/models =='
curl -sS --max-time 10 "$H/v1/models" > "$O/a-models.json" 2> "$O/a-models.curl-err" || true
check a "$O/a-models.json"

echo '== load: an empty /api/chat =='
printf '{"model": "%s", "messages": [], "stream": false, "keep_alive": -1}' "$MODEL" > "$O/load.req.json"
post load /api/chat 1200
check load "$O/load.json" "$O/load.time"
curl -sS --max-time 10 "$H/api/ps" > "$O/ps-loaded.json" || true
check ps-loaded "$O/ps-loaded.json" "$O/server.log"
mem mem-loaded.txt

echo '== (b) a short chat, not streamed =='
req b; post b /v1/chat/completions 300
check b "$O/b.json" "$O/b.time"

echo '== (c) a streamed chat =='
req c; stream c 300
check c "$O/c.sse.json"

echo '== (d) a tool call and its follow-up, OpenAI route =='
req d1; post d1 /v1/chat/completions 600
check d1 "$O/d1.json" "$O/d1.time"
if req d2 "$O/d1.json"; then stream d2 600; fi
check d2 "$O/d2.sse.json"
echo '== (d) the same pair on /api/chat, which reports what the cache held =='
req n1; post n1 /api/chat 600
check n1 "$O/n1.json" "$O/n1.time"
if req n2 "$O/n1.json"; then post n2 /api/chat 600; fi
check n2 "$O/n2.json" "$O/n2.time" "$O/n1.json"

echo '== (e) a first prompt of about 9,000 tokens, then the same again, then a follow-up =='
req e1; stream e1 1800
check e1 "$O/e1.sse.json"
mem mem-after-long.txt
cp "$O/e1.req.json" "$O/e2.req.json"; stream e2 1800
check e2 "$O/e2.sse.json" /dev/null "$O/e1.sse.json"
if req e3 "$O/e1.sse.json"; then post e3 /api/chat 1800; fi
check e3 "$O/e3.json" "$O/e3.time" "$O/e1.sse.json"

echo '== (f) the end =='
curl -sS --max-time 10 "$H/api/ps" > "$O/ps-end.json" || true
check ps-end "$O/ps-end.json" "$O/server.log"
mem mem-after.txt
check gpu "$O/mem-before.txt" "$O/mem-loaded.txt" "$O/mem-after-long.txt" "$O/mem-after.txt"
kill "$SRV" 2>/dev/null || true
wait "$SRV" 2>/dev/null || true
tail -5 "$O/server.log"
python3 -I "$C" final
