#!/usr/bin/env bash
# The prefix cache on disk of `llvq serve` on the CPU, on the real Inkling 2 Mini, on this Mac: gate B-iii of
# proofs/preregistration-servi-au-bit-2026-11-26.md, requests and readings from ops/inkling/prefix_checks.py with a
# static block just over 2,048 tokens (PREFIX_SECTIONS, PREFIX_TOOLS; the save point PREFIX_B, 2,048).
# Run it under ~/.llvq-guard/heavy.sh with a CPU build of llvq: it loads the model six times and reads three prompts
# cold on the CPU. LLVQ_THREADS is the guard's, the same for every server.
#
#   bash ops/inkling/serve-prefix-cpu.sh <llvq CPU binary> <inkling.json of the model> <output directory>
#
# Every server listens on 127.0.0.1:11438 and stores under the output (LLVQ_HOME), never ~/.llvq. In order:
#   s1      --prefix-cache on, a cold store: P-a (static + u4: full read, the file written at 2,048).
#   s2-s4   --prefix-cache on, the same store, restarted: P-b three times (static + u5), the first token timed.
#   s5      --prefix-cache off, a store of its own: the reference answer of static + u5.
#   s6      --prefix-cache on: one byte of the file damaged first; P-d (refused by name, full read, same answer).
# Before every request the 1-minute load must be at most 6 (waits up to 10 minutes, then says so).
set -euo pipefail
LLVQ=${1:?llvq binary}; MODEL_FILE=${2:?inkling.json}; O=${3:?output directory}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
C=$REPO/ops/inkling/prefix_checks.py
mkdir -p "$O"
[ -z "$(ls -A "$O")" ] || { echo "refused: $O is not empty" >&2; exit 1; }
if env | grep '^LLVQ_' | grep -qv '^LLVQ_THREADS='; then echo 'refused: LLVQ_* set in the environment' >&2; exit 1; fi
cp "$C" "$REPO/ops/inkling/serve_checks.py" "$0" "$O/"
MODEL=$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["name"])' "$MODEL_FILE")
export O MODEL PREFIX_SECTIONS=${PREFIX_SECTIONS:-12} PREFIX_TOOLS=${PREFIX_TOOLS:-3} PREFIX_B=${PREFIX_B:-2048}
"$LLVQ" -v | tee "$O/llvq-version.txt"
shasum -a 256 "$LLVQ" | tee "$O/llvq-sha256.txt"
echo "LLVQ_THREADS=${LLVQ_THREADS:-unset} RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-unset} PREFIX_SECTIONS=$PREFIX_SECTIONS PREFIX_TOOLS=$PREFIX_TOOLS PREFIX_B=$PREFIX_B" | tee "$O/settings.txt"
H=http://127.0.0.1:11438
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
  { date '+%F %T'; footprint -p "$SRV" 2>&1 | grep -iE 'phys_footprint|Footprint|TOTAL' | head -8
    curl -sS --max-time 10 "$H/api/ps" 2>/dev/null; echo; } > "$O/$1"
}
# start <label> <home> [serve flags...]
start() {
  local label=$1 home=$2; shift 2
  echo "== $label: home $(basename "$home"), flags [$*] =="
  echo "$(date '+%F %T') $label start $(machine)" >> "$O/machine.txt"
  local t0; t0=$(python3 -I -c 'import time; print(time.time())')
  env LLVQ_HOME="$home" LLVQ_HOST=127.0.0.1:11438 LLVQ_KEEP_ALIVE=-1 \
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
  curl -sS --max-time 1800 -H 'Content-Type: application/json' --data-binary @"$O/$label.load.req.json" \
    "$H/api/chat" > "$O/$label.load.json"
  python3 -I -c "import time, sys; print(f'$label loaded in {time.time() - float(sys.argv[1]):.1f} s')" "$t0" | tee -a "$O/driver.txt"
  mem "$label.mem-loaded.txt"
}
stop() {
  mem "$1.mem-end.txt"
  kill "$SRV" 2>/dev/null || true
  wait "$SRV" 2>/dev/null || true
  SRV=""
}
# footprint of the server every second while a request runs
sample() {
  while kill -0 "$SRV" 2>/dev/null; do
    printf '%s %s\n' "$(python3 -I -c 'import time; print(f"{time.time():.1f}")')" \
      "$(footprint -p "$SRV" 2>/dev/null | grep -m1 -i 'phys_footprint:' | tr -s ' ')" >> "$O/$1"
    sleep 1
  done
}
ask() {
  local name=$1 r=$2 t0 smp
  quiet "$name"
  python3 -I "$C" req "$r" > "$O/$name.req.json"
  sample "$name.footprint.txt" & smp=$!
  t0=$(python3 -I -c 'import time; print(time.time())')
  curl -sS -N --max-time 3600 -H 'Content-Type: application/json' --data-binary @"$O/$name.req.json" \
    "$H/api/chat" 2> "$O/$name.curl-err" | python3 -I "$C" ndjson "$t0" "$O/$name.raw.txt" "$O/$name.json" || true
  kill "$smp" 2>/dev/null || true; wait "$smp" 2>/dev/null || true
  python3 -I "$C" show "$name" "$O/$name.json" | tee -a "$O/driver.txt"
}
check() { python3 -I "$C" check "$@" || true; }

ON=$O/home; OFF=$O/home-off

start s1 "$ON" --prefix-cache on
ask pa u4
ls -l "$ON/prefix" > "$O/pa.prefix-ls.txt" 2>&1 || true
stop s1

for i in 1 2 3; do
  start "s$((i + 1))" "$ON" --prefix-cache on
  ask "pb$i" u5
  stop "s$((i + 1))"
done

start s5 "$OFF" --prefix-cache off
ask ref u5
stop s5

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
start s6 "$ON" --prefix-cache on
ask pd u5
stop s6

check cold-save "$O/pa.json" "$O/pa.prefix-ls.txt" "$O/s1.server.log"
for i in 1 2 3; do check loaded "P-b$i" "$O/pb$i.json" "$O/ref.json" "$O/s$((i + 1)).server.log" 0; done
check damaged "$O/pd.json" "$O/ref.json" "$O/s6.server.log" "$F"
check off "$OFF" "$O/s5.server.log" "$O/ref.json"
python3 -I "$C" final
