#!/usr/bin/env bash
# A request sent again to `llvq serve`, on the real Inkling 2 Mini on this Mac (Metal): gate C-ii of
# proofs/preregistration-servi-au-bit-2026-11-26.md. One server, one arm (a binary before or after the change):
#   a0       the agent-shaped request of prefix_checks.py (static block of about 8,770 tokens + u1), read cold;
#   a1..aN   the same request sent again, the first token timed;
#   c0       a short chat (`chat`, about 40 tokens), then c1..cN the same sent again.
# With CALIB=1, first one request of a shorter static block (PREFIX_SECTIONS, PREFIX_TOOLS, user u4): its prompt
# count and the save point the server logs fix the block of ops/inkling/serve-prefix-cpu.sh.
# Before every timed request the 1-minute load must be at most 6 (waits up to 10 minutes, then says so).
# Run it under ~/.llvq-guard/heavy.sh: it loads the model and uses the GPU.
#
#   bash ops/inkling/held-mac.sh <llvq binary> <inkling.json of the model> <output directory> [N, default 4]
set -euo pipefail
LLVQ=${1:?llvq binary}; MODEL_FILE=${2:?inkling.json}; O=${3:?output directory}; N=${4:-4}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
C=$REPO/ops/inkling/prefix_checks.py
mkdir -p "$O"
[ -z "$(ls -A "$O")" ] || { echo "refused: $O is not empty" >&2; exit 1; }
if env | grep '^LLVQ_' | grep -qv '^LLVQ_THREADS='; then echo 'refused: LLVQ_* set in the environment' >&2; exit 1; fi
cp "$C" "$REPO/ops/inkling/serve_checks.py" "$0" "$O/"
MODEL=$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["name"])' "$MODEL_FILE")
export O MODEL
"$LLVQ" -v | tee "$O/llvq-version.txt"
shasum -a 256 "$LLVQ" | tee "$O/llvq-sha256.txt"
H=http://127.0.0.1:11439
SRV=""
trap '[ -n "$SRV" ] && kill "$SRV" 2>/dev/null || true' EXIT
machine() { "$REPO/docs/mesures/inkling-mac-prefix-2026-11-26-brut/machine.sh"; }
quiet() {
  local l
  for _ in $(seq 1 40); do
    l=$(sysctl -n vm.loadavg | awk '{print $2}')
    awk -v l="$l" 'BEGIN{exit !(l <= 6)}' && { echo "$(date '+%F %T') $1 $(machine)" >> "$O/machine.txt"; return; }
    sleep 15
  done
  echo "$(date '+%F %T') $1 LOAD ABOVE 6 AFTER 10 MIN $(machine)" | tee -a "$O/machine.txt"
}
mem() {
  { date '+%F %T'; footprint -p "$SRV" 2>&1 | grep -iE 'phys_footprint|Footprint|IOAccelerator|TOTAL' | head -8
    curl -sS --max-time 10 "$H/api/ps" 2>/dev/null; echo; } > "$O/$1"
}
ask() {
  local name=$1 r=$2 t0
  quiet "$name"
  python3 -I "$C" req "$r" > "$O/$name.req.json"
  t0=$(python3 -I -c 'import time; print(time.time())')
  curl -sS -N --max-time 1800 -H 'Content-Type: application/json' --data-binary @"$O/$name.req.json" \
    "$H/api/chat" 2> "$O/$name.curl-err" | python3 -I "$C" ndjson "$t0" "$O/$name.raw.txt" "$O/$name.json" || true
  python3 -I "$C" show "$name" "$O/$name.json" | tee -a "$O/driver.txt"
}

echo "$(date '+%F %T') start $(machine)" >> "$O/machine.txt"
env LLVQ_HOME="$O/home" LLVQ_HOST=127.0.0.1:11439 LLVQ_KEEP_ALIVE=-1 \
  "$LLVQ" serve --local "$MODEL_FILE" > "$O/server.log" 2>&1 &
SRV=$!
up=0
for _ in $(seq 1 60); do
  if curl -sS --max-time 2 "$H/api/version" > /dev/null 2>&1; then up=1; break; fi
  kill -0 "$SRV" 2>/dev/null || break
  sleep 1
done
[ "$up" = 1 ] || { echo "STOP: the server did not answer within 60 s" >&2; tail -20 "$O/server.log" >&2; exit 1; }
printf '{"model": "%s", "messages": [], "stream": false, "keep_alive": -1}' "$MODEL" > "$O/load.req.json"
curl -sS --max-time 1200 -H 'Content-Type: application/json' --data-binary @"$O/load.req.json" "$H/api/chat" > "$O/load.json"
mem mem-loaded.txt

if [ "${CALIB:-0}" = 1 ]; then
  PREFIX_SECTIONS=${CALIB_SECTIONS:-10} PREFIX_TOOLS=${CALIB_TOOLS:-3} ask calib u4
fi
ask a0 u1
for i in $(seq 1 "$N"); do ask "a$i" u1; done
ask c0 chat
for i in $(seq 1 "$N"); do ask "c$i" chat; done
mem mem-end.txt
kill "$SRV" 2>/dev/null || true; wait "$SRV" 2>/dev/null || true; SRV=""
grep -E 'prefix cache|cache could not' "$O/server.log" > "$O/server-prefix-lines.txt" || true
echo "done $(date '+%F %T')" | tee -a "$O/driver.txt"
