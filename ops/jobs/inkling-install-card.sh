#!/usr/bin/env bash
# What a stranger gets: the released `llvq` archive installed on a machine that has neither our image nor a CUDA
# toolkit, Inkling 2 Mini pulled from the Hub by name, and served. One job on one card, in the image
# nvidia/cuda:12.4.1-base-ubuntu22.04, which carries the driver hooks and no cuBLAS, no NVRTC and no compiler.
#   install: llvq-runtime/install.sh from the release named by REL, piped to sh as its notes say;
#   `llvq serve` in the background, then `llvq pull inkling2:mini`, anonymous, timed;
#   `llvq run inklingmini-2-tetra "<a prompt>"`, the command the operator asked for;
#   then the requests and checks of ops/jobs/inkling-serve-card.sh, (a) to (f), on the pulled model.
# Every request and answer is saved raw under the output directory. summary.txt holds one line a check.
#
#   DRY_RUN=1 bash ops/jobs/inkling-install-card.sh              parses, launches nothing
#   PRINT_JOB=1 DRY_RUN=1 bash ops/jobs/inkling-install-card.sh  prints the job body
#   REL=<release tag> bash ops/jobs/inkling-install-card.sh
#   INSTALL=image IMAGE=hf.co/spaces/Pier-Jean/llvq-runner-cuda bash ops/jobs/inkling-install-card.sh
#     the `llvq` of our own image instead of a release: what a branch's library pulls and serves, before a release
#
# FLAVOR=l4x1 (default: an Ada card, the generation of the RTX 4060 Ti) or l40sx1.
# Preregistered: PREREG below, signed and stamped before this.
# Cost, estimated: about 15 minutes, $0.20 on l4x1; 40-minute timeout.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=${PREREG:-proofs/preregistration-inkling-install-card-2026-11-24.md}
FLAVOR=${FLAVOR:-l4x1}
REL=${REL:-}
INSTALL=${INSTALL:-release}
IMAGE=${IMAGE:-nvidia/cuda:12.4.1-base-ubuntu22.04}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/inkling-install-card-$FLAVOR-2026-11-24${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
TIMEOUT=${TIMEOUT:-40m}
CHECKS=ops/inkling/serve_checks.py

if [ "$DRY" != 1 ]; then
  case "$INSTALL" in
    release) [ -n "$REL" ] || { echo "refused: REL unset, the release tag whose archive is installed" >&2; exit 1; } ;;
    image) ;;
    *) echo "refused: INSTALL=$INSTALL is neither release nor image" >&2; exit 1 ;;
  esac
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
fi

PRE=$(printf 'O=%q\nREL=%q\nINSTALL=%q' "$O" "$REL" "$INSTALL")
# The helper travels in the command: C is its path inside the container.
PRE="$PRE"$'\n'"C=/tmp/serve_checks.py"$'\n'"cat > \"\$C\" <<'SERVECHECKS'"$'\n'"$(cat "$CHECKS")"$'\n'"SERVECHECKS"

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
if [ "$INSTALL" = release ]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq > /dev/null
  apt-get install -y -qq --no-install-recommends curl ca-certificates python3 > /dev/null
fi
cp "$C" "$O/serve_checks.py"
export O MODEL=inkling2:mini
nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.used,driver_version --format=csv,noheader | tee "$O/gpu-before.txt"
df -h / /tmp | tee "$O/disk.txt"
echo '== what the machine has before the install =='
{ ldconfig -p | grep -E 'cublas|nvrtc|curand|libcuda\.so' || true; } | tee "$O/system-cuda-libs.txt"
{ command -v nvcc cargo rustc llvq || true; } | tee "$O/system-tools.txt"

if [ "$INSTALL" = release ]; then
  echo '== install =='
  B=https://github.com/pjmalandrino/llvq-tetra/releases/download/$REL
  date -u
  curl -fsSL "$B/install.sh" | LLVQ_URL="$B/llvq-0.1.0-linux-x86_64-cuda.tar.gz" sh 2>&1 | grep -v '^#' | tee "$O/install.txt"
  date -u
fi
command -v llvq
llvq -v | tee "$O/llvq-version.txt"

echo '== the server =='
export LLVQ_HOME=/tmp/llvq-home LLVQ_HOST=127.0.0.1:11435
H=http://127.0.0.1:11435
env LLVQ_KEEP_ALIVE=-1 llvq serve > "$O/server.log" 2>&1 &
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

echo '== pull, anonymous =='
if env | grep -q '^HF_TOKEN='; then echo 'note: HF_TOKEN is set in the job; the repository is public either way'; fi
T0=$(date +%s)
llvq pull inkling2:mini 2>&1 | tr '\r' '\n' | tail -5 | tee "$O/pull.txt"
echo "pull seconds: $(( $(date +%s) - T0 ))" | tee -a "$O/pull.txt"
llvq ls | tee "$O/ls.txt"
du -sh "$LLVQ_HOME" | tee -a "$O/pull.txt"

echo '== llvq run, the alias the operator asked for =='
T0=$(date +%s)
timeout 10m llvq run inklingmini-2-tetra --verbose "What is the capital of France? Answer with one word." 2>&1 | tee "$O/run.txt" || echo "llvq run exit $?" | tee -a "$O/run.txt"
echo "run seconds, load included: $(( $(date +%s) - T0 ))" | tee -a "$O/run.txt"

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
  exit 0
fi

uv run ops/run.py bench \
  --image "$IMAGE" \
  --flavor "$FLAVOR" --any-flavor --timeout "$TIMEOUT" \
  --bucket "$BUCKET" --out-mount /out \
  --name "inkling-install-card-$FLAVOR${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
