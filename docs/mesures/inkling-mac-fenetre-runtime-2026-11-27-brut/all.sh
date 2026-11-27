#!/bin/bash
# Lot H, the measured part of prereg inkling-fenetre-runtime-2026-11-27, one job under the guard's lock, from the
# worktree root:  ~/.llvq-guard/heavy.sh "lotH: ..." docs/mesures/inkling-mac-fenetre-runtime-2026-11-27-brut/all.sh < /dev/null
# Every server: a fresh process on 127.0.0.1:11437, a working LLVQ_HOME under $B/work, the model file of the
# operator's store read in place (--local, as ops/inkling/serve-prefix-mac.sh), an empty /api/chat to load, then
# the long request u1 of ops/inkling/prefix_checks.py (8,791 tokens, temperature 0, 48 tokens), then the short
# request `chat` (temperature 0, 48 tokens). The first token is the client's, from the request to the first line
# that carries text (prefix_checks.py ndjson). 45 s between servers.
#   H-i    hi-before (target/loth/llvq-before, no switch) and hi-after (target/loth/llvq, no switch), cache off.
#   H-iii  p0 (discarded) then p1..p3: full (--attn-window full) then narrow, cache off. H-ii reads their tokens.
#   H-iv   cache on. iv1 full, home iv-a: writes file A. iv2 narrow, home iv-a: must not load A, writes B.
#          B alone moved to home iv-b; iv3 full, home iv-b: must not load B.
set -u
B=docs/mesures/inkling-mac-fenetre-runtime-2026-11-27-brut
W=$B/work; mkdir -p $W
C=ops/inkling/prefix_checks.py
M=$HOME/.llvq/models/sha256-1e9314dba89332cb1af444619dc5d31a2a1ed6a50a1112e0ca2b8fc55bb6bfd3/inkling.json
export MODEL=inkling2:mini O=$W
P=11437; H=http://127.0.0.1:$P
SRV=""
trap '[ -n "$SRV" ] && kill "$SRV" 2>/dev/null' EXIT
if env | grep '^LLVQ_' | grep -qv '^LLVQ_THREADS='; then echo 'refused: LLVQ_* set in the environment' >&2; exit 1; fi
python3 -I $C req u1 > $W/u1.req.json; python3 -I $C req chat > $W/chat.req.json
shasum -a 256 $W/u1.req.json $W/chat.req.json | tee $B/requests.sha256
echo "all: start $(date '+%F %T') at $(git rev-parse HEAD)"
shasum -a 256 target/loth/llvq-before target/loth/llvq
# ask <label>.<req>
ask() {
  local t0
  t0=$(python3 -I -c 'import time; print(time.time())')
  curl -sS -N --max-time 900 -H 'Content-Type: application/json' --data-binary @$W/$2.req.json "$H/api/chat" \
    2> $B/$1.$2.curl-err | python3 -I $C ndjson "$t0" $B/$1.$2.raw.txt $B/$1.$2.json
  python3 -I - $B/$1.$2.json "$1.$2" <<'PY'
import json, sys
s = json.load(open(sys.argv[1])); f = s.get("final") or {}
print(f"{sys.argv[2]}: first token {s['first_token_s']} s, prompt {f.get('prompt_eval_count')}, cached {f.get('prompt_eval_cached_count')}, read {(f.get('prompt_eval_duration') or 0)/1e9:.2f} s, eval {f.get('eval_count')}, errors {s['errors']}")
PY
}
# server <label> <binary> <home> [flags]
server() {
  local label=$1 bin=$2 home=$3; shift 3
  mkdir -p "$home"
  echo "== $label: $(basename $bin), home $(basename $home), flags [$*] =="
  echo "$(date '+%F %T') $label $($B/machine.sh)" | tee $B/$label.machine
  env LLVQ_HOME="$home" LLVQ_HOST=127.0.0.1:$P LLVQ_KEEP_ALIVE=-1 "$bin" serve --local $M "$@" < /dev/null > $B/$label.server.log 2>&1 &
  SRV=$!
  local up=0
  for _ in $(seq 1 60); do curl -sS --max-time 2 $H/api/version > /dev/null 2>&1 && { up=1; break; }; kill -0 $SRV 2>/dev/null || break; sleep 1; done
  [ $up = 1 ] || { echo "STOP: $label did not answer"; tail -20 $B/$label.server.log; exit 1; }
  printf '{"model": "%s", "messages": [], "stream": false, "keep_alive": -1}' "$MODEL" > $W/load.req.json
  curl -sS --max-time 1200 -H 'Content-Type: application/json' --data-binary @$W/load.req.json $H/api/chat > $B/$label.load.json
  ask $label u1
  ask $label chat
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; SRV=""
  ls "$home/prefix" 2>/dev/null | sed "s/^/$label prefix dir: /"
}
OLD=target/loth/llvq-before; NEW=target/loth/llvq
server hi-before $OLD $W/home-hi-before --prefix-cache off; sleep 45
server hi-after $NEW $W/home-hi-after --prefix-cache off; sleep 45
for r in 0 1 2 3; do
  server p$r-full $NEW $W/home-p$r-full --prefix-cache off --attn-window full; sleep 45
  server p$r-narrow $NEW $W/home-p$r-narrow --prefix-cache off --attn-window narrow; sleep 45
done
server iv1-full $NEW $W/home-iv-a; sleep 45
server iv2-narrow $NEW $W/home-iv-a --attn-window narrow; sleep 45
mkdir -p $W/home-iv-b/prefix
KA=$(grep -o 'key [0-9a-f]\{16\}' $B/iv1-full.server.log | head -1 | cut -d' ' -f2)
KB=$(grep -o 'key [0-9a-f]\{16\}' $B/iv2-narrow.server.log | head -1 | cut -d' ' -f2)
echo "keys: full $KA, narrow $KB"
for f in $W/home-iv-a/prefix/$KB-*.llvqprefix; do mv "$f" $W/home-iv-b/prefix/; done
ls $W/home-iv-b/prefix | sed 's/^/home iv-b before iv3: /'
server iv3-full $NEW $W/home-iv-b
echo "all: done $(date '+%F %T')"
