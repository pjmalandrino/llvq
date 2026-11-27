#!/bin/bash
# A-v: llvq serve and llvq run with the store's 4B, the binary before (expanded config) and after (Metal config).
# LLVQ_HOME is a working store under this directory; the 4B is registered from the local sealed file, no network.
set -u
cd /Users/pjmalandrino/Documents/Pro/workspace/poc/llvq/.claude/worktrees/mac-inkling
B=$PWD/docs/mesures/qwen-metal-servi-2026-11-26-brut; W=$B/work; O=$B/a-v
mkdir -p $O; export LLVQ_HOME=$W/home LLVQ_HOST=127.0.0.1:11437; unset LLVQ_THREADS
H=http://$LLVQ_HOST
[ -f $W/home/manifests/qwen3/4b.json ] || perl -e 'alarm shift; exec @ARGV' 600 $W/llvq-after pull qwen3:4b --from ~/q4b-sealed-2026-09-23/qwen3-4b-sealed.bin < /dev/null > $O/pull.txt 2>&1
PROMPT="Why is the sky blue? Answer in two sentences."
for which in before after; do
  L=$W/llvq-$which
  $L -v > $O/$which-version.txt
  $L serve < /dev/null > $O/$which-server.log 2>&1 &
  SRV=$!
  trap 'kill $SRV 2>/dev/null' EXIT
  for _ in $(seq 1 60); do curl -sS --max-time 2 $H/api/version >/dev/null 2>&1 && break; sleep 1; done
  # stdin closed: `llvq run` reads stdin to the end when it is not a terminal. Hard limit of 600 s.
  perl -e 'alarm shift; exec @ARGV' 600 $L run --temperature 0 --verbose qwen3:4b "$PROMPT" < /dev/null > $O/$which-run.txt 2> $O/$which-run.err
  echo "run exit $?" >> $O/$which-run.err
  printf '{"model":"qwen3:4b","prompt":"%s","stream":false,"options":{"temperature":0,"num_predict":200}}' "$PROMPT" > $O/generate.req.json
  curl -sS --max-time 600 -H 'Content-Type: application/json' --data-binary @$O/generate.req.json $H/api/generate > $O/$which-generate.json
  printf '{"model":"qwen3:4b","messages":[{"role":"user","content":"Write a haiku about the sea."}],"stream":false,"options":{"temperature":0,"num_predict":120}}' > $O/chat.req.json
  curl -sS --max-time 600 -H 'Content-Type: application/json' --data-binary @$O/chat.req.json $H/api/chat > $O/$which-chat.json
  kill $SRV; wait $SRV 2>/dev/null
  lsof -nP -iTCP:11437 -sTCP:LISTEN >/dev/null && { echo "port 11437 still held" >&2; exit 1; }
done
python3 -I - "$O" <<'PY' | tee $O/compare.txt
import json, sys, os
o = sys.argv[1]
def rd(p): return open(os.path.join(o, p)).read()
for kind, get in [("run", lambda w: rd(f"{w}-run.txt")),
                  ("generate", lambda w: json.loads(rd(f"{w}-generate.json"))["response"]),
                  ("chat", lambda w: json.loads(rd(f"{w}-chat.json"))["message"]["content"])]:
    a, b = get("before"), get("after")
    print(f"{kind}: {'SAME TEXT' if a == b else 'DIFFERENT'} ({len(a)} and {len(b)} characters)")
for w in ["before", "after"]:
    for line in rd(f"{w}-server.log").splitlines():
        if "loading qwen3:4b" in line: print(w, line)
PY
