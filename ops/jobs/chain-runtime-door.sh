#!/usr/bin/env bash
# The runtime's own door on the chain prefill (dev/trois-gains-2026-11-15, after the operator's serving decision of
# 2026-11-15): `llvq serve` on one L40S, the sealed 4B imported with `llvq pull --from`, requests through /api/chat as
# a user sends them. One job:
#   (1) the load check: the module cache copied from the last job's bucket folder, `llvq serve` with NO profile
#       variable, the 4B loaded through the API. /api/ps must name qwen3-4b-tetra-e4-4e-v9-chain.json, the server's
#       log must name that config and the served 4E v9 module, or the job STOPS. The log says whether the two tree
#       modules came from the cache (two hits) or were compiled (a miss: the load then takes about 509 s instead of 55,
#       the job goes on inside its timeout, and the summary says so);
#   (2) the default (the chain): a short prompt and the 8,191-token prompt, alternated, 1 discarded + 3 timed rounds,
#       streamed; time to the first content chunk on the client, the server's prompt_eval_duration, the text;
#   (3) the way back, `LLVQ_PROFILE=eager`, a new server: /api/ps must name qwen3-4b-tetra-e4-4e-v9.json; the same
#       requests; the texts compared with (2), and where they part;
#   (4) `LLVQ_PROFILE=bogus` and `LLVQ_PROFILE=chain`: the load is refused, the error names the variable.
# Output chain-runtime-door-2026-11-15.
# Preregistered: proofs/preregistration-chain-runtime-door-2026-11-15-DRAFT.md, to sign and stamp before this (the
# stamped file drops -DRAFT from its name; PREREG below names the stamped one).
# The image must carry this branch's runtime (a rebuild); no binary or path is added to the image.
#
#   DRY_RUN=1 bash ops/jobs/chain-runtime-door.sh                     parses, launches nothing
#   IMAGE_SHA=<space sha> bash ops/jobs/chain-runtime-door.sh         launches, after the operator's go
#
# Cost (l40sx1 $1.80/h), estimated from measured times (prereg's cost section): about 480 s, $0.24, with the cache
# hit; about 940 s, $0.47, if the key misses. Timeout 25m: $0.75 at worst.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=proofs/preregistration-chain-runtime-door-2026-11-15.md
OUT_NAME=chain-runtime-door-2026-11-15
IMAGE_SHA=${IMAGE_SHA:-}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/$OUT_NAME${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
P4E_V9_SHA=1b422b62328be6752df4add43280565fd2a49cfa20ee26953b301abe7a15c530
MODCACHE_SRC=/out/chain-prefill-served-2026-11-15/modcache
TIMEOUT=${TIMEOUT:-25m}
NTOK=256
ROUNDS=3
MODEL=qwen3:4b
SEALED="sealed-4b-2026-09-23/qwen3-4b-sealed.bin:1418224685"
SEALED_SHA=886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8
PROMPTS=/usr/local/share/llvq/spec-prompts-v1.jsonl
LONG=/usr/local/share/llvq/spec-long-8k-v1.jsonl
PROMPTS_SHA=$(shasum -a 256 < ops/data/spec-prompts-v1.jsonl | cut -d' ' -f1)
LONG_SHA=$(shasum -a 256 < ops/data/spec-long-8k-v1.jsonl | cut -d' ' -f1)
[ "$PROMPTS_SHA" = f749eb359c2cf5c8a648ca99fc0f856e620f8adb3f1d233f4c477ff322257f1a ] \
  || { echo "refused: ops/data/spec-prompts-v1.jsonl is not at its pinned sha256" >&2; exit 1; }
[ "$LONG_SHA" = da4edbd6f12184cf8ff938f057500f3e30e35d8c7a351d2e925316d225d62ebc ] \
  || { echo "refused: ops/data/spec-long-8k-v1.jsonl is not at its pinned sha256" >&2; exit 1; }
grep -q '"cuda": "qwen3-4b-tetra-e4-4e-v9-chain.json"' llvq-runtime/library.json \
  || { echo "refused: the library does not serve the 4B by the chain config" >&2; exit 1; }

if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  path=${SEALED%%:*}; bytes=${SEALED##*:}
  REMOTE=$(hf buckets ls "hf://buckets/$BUCKET/${path%/*}/" 2>/dev/null \
    | awk -v n="/${path##*/}" 'substr($NF, length($NF) - length(n) + 1) == n {print $1}' || true)
  [ "$REMOTE" = "$bytes" ] || { echo "refused: $path is ${REMOTE:-absent} B in the bucket, expected $bytes" >&2; exit 1; }
  NC=$(hf buckets ls "hf://buckets/$BUCKET/${MODCACHE_SRC#/out/}/" 2>/dev/null | grep -c '\.cubin' || true)
  echo "module cache in the bucket: $NC cubin file(s) (two expected; fewer means the job compiles, about 8.5 min more)"
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
  IMAGE_SHA=${IMAGE_SHA:?IMAGE_SHA=<the Space sha of the rebuild that carries this branch>}
  NOW=$(uv run --quiet --with huggingface_hub python -c \
    'from huggingface_hub import HfApi; print(HfApi().space_info("Pier-Jean/llvq-runner-cuda").sha)')
  [ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" >&2; exit 1; }
  echo "bucket copy verified by size; image $NOW"
fi

PRE=$(printf 'O=%q\nSEALED=%q\nSEALED_SHA=%q\nP4E_V9_SHA=%q\nMODCACHE_SRC=%q\nNTOK=%q\nROUNDS=%q\nMODEL=%q\nPROMPTS=%q\nLONG=%q\nPROMPTS_SHA=%q\nLONG_SHA=%q\nexport HF_HOME=/tmp/hf' \
  "$O" "$SEALED" "$SEALED_SHA" "$P4E_V9_SHA" "$MODCACHE_SRC" "$NTOK" "$ROUNDS" "$MODEL" "$PROMPTS" "$LONG" "$PROMPTS_SHA" "$LONG_SHA")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
echo '== control 1: oracle, the object, the prompt sets =='
timeout 4m oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle.txt" | tail -2
grep -q MATCH "$O/oracle.txt"
S=/out/${SEALED%%:*}; SB=${SEALED##*:}
test "$(stat -c %s "$S")" = "$SB"
test "$(sha256sum "$S" | cut -d' ' -f1)" = "$SEALED_SHA"
test "$(sha256sum "$PROMPTS" | cut -d' ' -f1)" = "$PROMPTS_SHA"
test "$(sha256sum "$LONG" | cut -d' ' -f1)" = "$LONG_SHA"
sha256sum "$S" "$PROMPTS" "$LONG" | tee "$O/files.sha256"
# the client: one streamed /api/chat a request, greedy, thinking off, the time to the first content chunk
cat > "$O/door_client.py" <<'PY'
import json, sys, time, urllib.request
host, model, prompts, long_file, ntok, rounds, out = sys.argv[1:8]
def first(path):
    for l in open(path):
        r = json.loads(l)
        if "_comment" not in r:
            return r["id"], r["prompt"]
reqs = [first(long_file), first(prompts)]
w = open(out, "w")
for rnd in range(int(rounds) + 1):
    for rid, text in reqs:
        body = {"model": model, "messages": [{"role": "user", "content": text}], "stream": True, "think": False,
                "options": {"temperature": 0, "num_predict": int(ntok)}, "keep_alive": -1}
        req = urllib.request.Request(f"http://{host}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time(); t_first = None; content = ""; last = {}
        with urllib.request.urlopen(req, timeout=600) as r:
            for line in r:
                v = json.loads(line)
                c = v.get("message", {}).get("content", "")
                if c and t_first is None:
                    t_first = time.time() - t0
                content += c
                if v.get("done"):
                    last = v
        rec = {"round": rnd, "id": rid, "ttft_client_s": t_first, "wall_s": time.time() - t0, "text": content,
               "prompt_eval_count": last.get("prompt_eval_count"), "prompt_eval_cached_count": last.get("prompt_eval_cached_count"),
               "prompt_eval_s": (last.get("prompt_eval_duration") or 0) / 1e9, "eval_count": last.get("eval_count"),
               "eval_s": (last.get("eval_duration") or 0) / 1e9, "done_reason": last.get("done_reason")}
        w.write(json.dumps(rec) + "\n"); w.flush()
        print(f"  round {rnd} {rid}: first content {t_first:.3f} s, prompt {rec['prompt_eval_count']} ({rec['prompt_eval_cached_count']} cached) "
              f"in {rec['prompt_eval_s']:.3f} s, {rec['eval_count']} tokens in {rec['eval_s']:.3f} s, {rec['done_reason']}", flush=True)
PY
cat > "$O/door_compare.py" <<'PY'
import json, statistics, sys
a = [json.loads(l) for l in open(sys.argv[1])]; b = [json.loads(l) for l in open(sys.argv[2])]
for rid in dict.fromkeys(r["id"] for r in a):
    ra = [r for r in a if r["id"] == rid]; rb = [r for r in b if r["id"] == rid]
    for name, rs in (("chain", ra), ("eager", rb)):
        texts = {r["text"] for r in rs}
        t = [r["ttft_client_s"] for r in rs if r["round"] > 0]
        print(f"{rid} {name}: first content median {statistics.median(t):.3f} s [{min(t):.3f}-{max(t):.3f}] over {len(t)} timed rounds; "
              f"prompt {rs[0]['prompt_eval_count']} tokens; texts identical across rounds: {len(texts) == 1}")
    x, y = ra[0]["text"], rb[0]["text"]
    if x == y:
        print(f"{rid}: the chain's text equals the eager prefill's, {ra[0]['eval_count']} tokens")
    else:
        i = next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), min(len(x), len(y)))
        print(f"{rid}: the texts part at character {i} of {len(x)} and {len(y)}: chain {x[max(0, i - 40):i + 40]!r} / eager {y[max(0, i - 40):i + 40]!r}")
PY
H=/tmp/llvq-home
LLVQ_HOME=$H timeout 5m llvq pull "$MODEL" --from "$S" > "$O/pull.txt" 2>&1 || { tail -5 "$O/pull.txt"; exit 1; }
MC=$O/modcache
mkdir -p "$MC"
cp -v "$MODCACHE_SRC"/*.cubin "$MC"/ 2>&1 | tee "$O/modcache-copy.txt" || echo "no cubin copied: the first load compiles" | tee -a "$O/modcache-copy.txt"
UP='import sys,urllib.request
try: urllib.request.urlopen("http://127.0.0.1:11435/api/version",timeout=2); sys.exit(0)
except Exception: sys.exit(1)'
api() { python3 -c 'import json,sys,urllib.request
req=urllib.request.Request("http://127.0.0.1:11435"+sys.argv[1],data=(sys.argv[2].encode() if len(sys.argv)>2 else None),headers={"Content-Type":"application/json"})
try: print(urllib.request.urlopen(req,timeout=1200).read().decode())
except urllib.error.HTTPError as e: print("HTTP", e.code, e.read().decode())' "$@"; }
# one server: start, load through the API, print the config /api/ps names
serve() {
  local tag=$1; shift
  env LLVQ_HOME=$H LLVQ_HOST=127.0.0.1:11435 LLVQ_KEEP_ALIVE=-1 LLVQ_MODULE_CACHE=$MC "$@" llvq serve > "$O/serve-$tag.txt" 2>&1 &
  SP=$!
  for i in $(seq 1 60); do python3 -c "$UP" && break; sleep 1; done
  local t0; t0=$(date +%s.%N)
  api /api/generate "{\"model\": \"$MODEL\", \"stream\": false}" > "$O/load-$tag.json"
  echo "load $tag: $(python3 -c "print(f'{$(date +%s.%N) - $t0:.1f} s')")" | tee -a "$O/check-summary.txt"
  api /api/ps > "$O/ps-$tag.json"
  python3 -c 'import json,sys; m=json.load(open(sys.argv[1]))["models"]; print("ps:", [(x["name"], x["llvq"]["state"], x["llvq"]["config"], x["llvq"]["error"]) for x in m])' "$O/ps-$tag.json" | tee -a "$O/check-summary.txt"
}
stop() { kill $SP; wait $SP 2>/dev/null || true; }
echo '== (1) the load check: llvq serve with no profile =='
date -u
serve chain
grep -aE '^llvq: (loading|compiling)|module cache|chain prefill: arm|4E: persistent module' "$O/serve-chain.txt" | tee -a "$O/check-summary.txt"
grep -q '"config": *"qwen3-4b-tetra-e4-4e-v9-chain.json"' "$O/ps-chain.json" \
  || { echo "STOP: /api/ps does not name the chain config" | tee -a "$O/check-summary.txt"; stop; exit 1; }
grep -aq "llvq: loading $MODEL on cuda with config qwen3-4b-tetra-e4-4e-v9-chain.json" "$O/serve-chain.txt" \
  || { echo "STOP: the server's log does not name the chain config" | tee -a "$O/check-summary.txt"; stop; exit 1; }
grep -aqF "4E: persistent module 252757 bytes, sha256 $P4E_V9_SHA, row form 7," "$O/serve-chain.txt" \
  || { echo "STOP: the served 4E v9 module is not in the log" | tee -a "$O/check-summary.txt"; stop; exit 1; }
HITS=$(grep -ac 'module cache: hit' "$O/serve-chain.txt" || true)
echo "module cache: $HITS hit(s) of 2$( [ "$HITS" = 2 ] || echo '; the key missed (another driver or sm?), the modules were compiled and the load is the cold one')" | tee -a "$O/check-summary.txt"
echo "the load check passed" | tee -a "$O/check-summary.txt"
set +e
echo '== (2) the default: the chain =='
date -u
timeout 8m python3 "$O/door_client.py" 127.0.0.1:11435 "$MODEL" "$PROMPTS" "$LONG" "$NTOK" "$ROUNDS" "$O/req-chain.jsonl" 2>&1 | tee "$O/req-chain.txt"
stop
echo '== (3) the way back: LLVQ_PROFILE=eager =='
date -u
serve eager LLVQ_PROFILE=eager
grep -q '"config": *"qwen3-4b-tetra-e4-4e-v9.json"' "$O/ps-eager.json" \
  && echo "the way back loads the v9 config" | tee -a "$O/check-summary.txt" \
  || echo "VOID: the way back does not load the v9 config" | tee -a "$O/check-summary.txt"
timeout 10m python3 "$O/door_client.py" 127.0.0.1:11435 "$MODEL" "$PROMPTS" "$LONG" "$NTOK" "$ROUNDS" "$O/req-eager.jsonl" 2>&1 | tee "$O/req-eager.txt"
stop
echo '== (4) refused profiles =='
for p in bogus chain; do
  serve "refused-$p" LLVQ_PROFILE=$p
  { grep -q 'LLVQ_PROFILE=' "$O/load-refused-$p.json" && grep -q "$p" "$O/load-refused-$p.json" \
      && ! grep -q '"config": *"qwen3' "$O/ps-refused-$p.json"; } \
    && echo "LLVQ_PROFILE=$p refused by name: $(head -c 300 "$O/load-refused-$p.json")" | tee -a "$O/check-summary.txt" \
    || echo "VOID: LLVQ_PROFILE=$p not refused by name: $(head -c 300 "$O/load-refused-$p.json")" | tee -a "$O/check-summary.txt"
  stop
done
date -u
python3 "$O/door_compare.py" "$O/req-chain.jsonl" "$O/req-eager.jsonl" > "$O/compare.txt" 2>&1
set -e
{ cat "$O/check-summary.txt"; cat "$O/compare.txt"; } > "$O/summary.txt"
cat "$O/summary.txt"
ls -la "$O"
JOB

if [ "$DRY" = 1 ]; then
  JOBSCRIPT="set -euo pipefail
$PRE
$BODY"
  printf '%s\n' "$JOBSCRIPT" | bash -n && echo "DRY_RUN: job script parses; nothing launched" >&2
  exit 0
fi

uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "chain-runtime-door${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
