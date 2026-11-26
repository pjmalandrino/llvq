#!/usr/bin/env bash
# The prefix cache on disk of `llvq serve`, on the real Inkling 2 Mini, on this Mac (Metal): checks P-a to P-f of
# proofs/preregistration-inkling-mac-prefix-2026-11-26.md, requests and readings from ops/inkling/prefix_checks.py.
# Run it under ~/.llvq-guard/heavy.sh: it loads the model eight times and uses the GPU.
#
#   bash ops/inkling/serve-prefix-mac.sh <llvq binary> <inkling.json of the model> <output directory>
#
# Every server listens on 127.0.0.1:11437 and stores in a directory under the output (LLVQ_HOME), never ~/.llvq.
# Servers in order, each one stopped before the next starts:
#   s1  cache on, a cold store: P-a (a first request: full read, the file written), then P-c (a new conversation).
#   s2-s4  cache on, the same store, restarted: P-b three times (a third user message), the first token timed.
#   s5  cache on: one byte of the file damaged first; P-d (refused by name, full read, same answer).
#   s6  cache on: P-e (the system prompt changed by one word: no load).
#   s7, s8  --prefix-cache off, a store of their own: P-f, and the reference answers of the three user messages.
# The cold reads alternate CANDLE_METAL_COMPUTE_PER_BUFFER between the lot 3 default (30, unset) and 50.
set -euo pipefail
LLVQ=${1:?llvq binary}; MODEL_FILE=${2:?inkling.json}; O=${3:?output directory}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
C=$REPO/ops/inkling/prefix_checks.py
mkdir -p "$O"
[ -z "$(ls -A "$O")" ] || { echo "refused: $O is not empty" >&2; exit 1; }
if env | grep '^LLVQ_' | grep -qv '^LLVQ_THREADS='; then echo 'refused: LLVQ_* set in the environment' >&2; exit 1; fi
if env | grep -q '^CANDLE_METAL_COMPUTE_PER_BUFFER='; then echo 'refused: CANDLE_METAL_COMPUTE_PER_BUFFER set' >&2; exit 1; fi
cp "$C" "$REPO/ops/inkling/serve_checks.py" "$O/"
MODEL=$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["name"])' "$MODEL_FILE")
export O MODEL
"$LLVQ" -v | tee "$O/llvq-version.txt"
H=http://127.0.0.1:11437
SRV=""
trap '[ -n "$SRV" ] && kill "$SRV" 2>/dev/null || true' EXIT

machine() { "$REPO/docs/mesures/inkling-mac-prefix-2026-11-26-brut/machine.sh"; }
mem() {
  { date '+%F %T'; footprint -p "$SRV" 2>&1 | grep -iE 'phys_footprint|Footprint|IOAccelerator|TOTAL' | head -8
    curl -sS --max-time 10 "$H/api/ps" 2>/dev/null; echo; } > "$O/$1"
}
# start <label> <home> <cpb: 30 or 50> [serve flags...]
start() {
  local label=$1 home=$2 cpb=$3; shift 3
  echo "== $label: home $(basename "$home"), compute per buffer $cpb, flags [$*] =="
  echo "$(date '+%F %T') $label cpb $cpb $(machine)" | tee -a "$O/machine.txt"
  local cenv=()
  [ "$cpb" = 50 ] && cenv=(CANDLE_METAL_COMPUTE_PER_BUFFER=50)
  env LLVQ_HOME="$home" LLVQ_HOST=127.0.0.1:11437 LLVQ_KEEP_ALIVE=-1 ${cenv[@]+"${cenv[@]}"} \
    "$LLVQ" serve --local "$MODEL_FILE" "$@" > "$O/$label.server.log" 2>&1 &
  SRV=$!
  local up=0
  for _ in $(seq 1 60); do
    if curl -sS --max-time 2 "$H/api/version" > /dev/null 2>&1; then up=1; break; fi
    kill -0 "$SRV" 2>/dev/null || break
    sleep 1
  done
  [ "$up" = 1 ] || { echo "STOP: $label did not answer within 60 s" >&2; tail -20 "$O/$label.server.log" >&2; exit 1; }
  printf '{"model": "%s", "messages": [], "stream": false, "keep_alive": -1}' "$MODEL" > "$O/$label.load.req.json"
  curl -sS --max-time 1200 -H 'Content-Type: application/json' --data-binary @"$O/$label.load.req.json" \
    "$H/api/chat" > "$O/$label.load.json"
  mem "$label.mem-loaded.txt"
}
stop() {
  mem "$1.mem-end.txt"
  kill "$SRV" 2>/dev/null || true
  wait "$SRV" 2>/dev/null || true
  SRV=""
}
# footprint of the server every second while a request runs: what the save or the load adds
sample() {
  while kill -0 "$SRV" 2>/dev/null; do
    printf '%s %s\n' "$(python3 -I -c 'import time; print(f"{time.time():.1f}")')" \
      "$(footprint -p "$SRV" 2>/dev/null | grep -m1 -i 'phys_footprint:' | tr -s ' ')" >> "$O/$1"
    sleep 1
  done
}
# ask <name> <request>: one streamed /api/chat, timed from the request to each line
ask() {
  local name=$1 r=$2 t0 smp
  python3 -I "$C" req "$r" > "$O/$name.req.json"
  sample "$name.footprint.txt" & smp=$!
  t0=$(python3 -I -c 'import time; print(time.time())')
  curl -sS -N --max-time 1800 -H 'Content-Type: application/json' --data-binary @"$O/$name.req.json" \
    "$H/api/chat" 2> "$O/$name.curl-err" | python3 -I "$C" ndjson "$t0" "$O/$name.raw.txt" "$O/$name.json" || true
  kill "$smp" 2>/dev/null || true; wait "$smp" 2>/dev/null || true
  python3 -I "$C" show "$name" "$O/$name.json"
}
check() { python3 -I "$C" check "$@" || true; }
loaded_lines() { grep -c 'prefix cache: loaded' "$O/$1.server.log" || true; }

ON=$O/home; OFF=$O/home-off

start s1 "$ON" 30
ask pa u1
start_files=$(ls "$ON/prefix" 2>/dev/null || true)
echo "files after P-a: $start_files"
ls -l "$ON/prefix" > "$O/pa.prefix-ls.txt" 2>&1 || true
ask pc u3
stop s1

for i in 1 2 3; do
  start "s$((i + 1))" "$ON" 30
  ask "pb$i" u2
  stop "s$((i + 1))"
done

F=$(ls "$ON/prefix" | grep '\.llvqprefix$' | head -1)
python3 -I - "$ON/prefix/$F" <<'PY'
import sys
p = sys.argv[1]
b = bytearray(open(p, "rb").read())
at = len(b) // 2
b[at] ^= 0x01
open(p, "wb").write(b)
print(f"damaged one byte at offset {at} of {len(b)}")
PY
start s5 "$ON" 50
ask pd u2
stop s5

start s6 "$ON" 30
ask pe u2-changed
stop s6

start s7 "$OFF" 50 --prefix-cache off
ask ref1 u1
ask ref2 u2
stop s7
start s8 "$OFF" 30 --prefix-cache off
ask ref3 u3
stop s8

check cold-save "$O/pa.json" "$O/pa.prefix-ls.txt" "$O/s1.server.log"
check same P-a "$O/pa.json" "$O/ref1.json"
check loaded P-c "$O/pc.json" "$O/ref3.json" "$O/s1.server.log" 0
for i in 1 2 3; do check loaded "P-b$i" "$O/pb$i.json" "$O/ref2.json" "$O/s$((i + 1)).server.log" 0; done
check damaged "$O/pd.json" "$O/ref2.json" "$O/s5.server.log" "$F"
check changed "$O/pe.json" "$O/s6.server.log"
check off "$OFF" "$O/s7.server.log" "$O/ref1.json" "$O/ref2.json"
check off "$OFF" "$O/s8.server.log" "$O/ref3.json"
python3 -I "$C" final
