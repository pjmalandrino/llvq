#!/usr/bin/env bash
# Inkling 2 Mini on a CPU, the real file set, one instance: `inkcpu` (load, prompt, decode rate, a token split by
# part) with the codes packed, then decoded once into nibbles, each under /usr/bin/time for its peak memory, then
# `llvq run inkling2:mini` through the server as a stranger would.
#
#   DRY_RUN=1 bash ops/jobs/inkling-cpu-x86.sh     parses, launches nothing
# Preregistered: PREREG below. Cost, estimated: 40 minutes of cpu-upgrade, $0.02; 90-minute timeout.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
PREREG=${PREREG:-proofs/preregistration-inkling-cpu-x86-2026-11-25.md}
FLAVOR=${FLAVOR:-cpu-upgrade}
COMMIT=${COMMIT:-$(git rev-parse HEAD)}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/inkling-cpu-x86-$FLAVOR-2026-11-25${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  git branch -r --contains "$COMMIT" | grep -q . || { echo "refused: $COMMIT is not pushed" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
fi
PRE=$(printf 'O=%q\nCOMMIT=%q' "$O" "$COMMIT")
IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
# The launcher's shell does not carry the image's PATH.
export PATH=/usr/local/cargo/bin:$PATH RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq > /dev/null
apt-get install -y -qq --no-install-recommends git ca-certificates pkg-config libssl-dev build-essential cmake curl time > /dev/null
lscpu | tee "$O/lscpu.txt" | grep -E 'Model name|Thread|^CPU\(s\)|L1d|L2|L3'; nproc | tee "$O/nproc.txt"
git clone -q https://github.com/pjmalandrino/llvq-tetra /src && cd /src
git checkout -q "$COMMIT" && git rev-parse HEAD | tee "$O/commit.txt"
cargo build --profile ci --locked -p llvq-runtime --bin llvq 2>&1 | tail -1
cargo build --profile ci --locked -p llvq-llm --bin inkcpu 2>&1 | tail -1
cargo test --profile ci --locked -p llvq-llm --test inkling_cpu 2>&1 | grep 'test result' | tee "$O/tests.txt"
stat() { head -1 /proc/stat > "$O/stat-$1.txt"; }
free -g | tee "$O/mem.txt"; df -h /tmp | tee "$O/disk.txt"

export LLVQ_HOME=/tmp/llvq-home LLVQ_HOST=127.0.0.1:11435
LLVQ=/src/target/ci/llvq
T0=$(date +%s); $LLVQ pull inkling2:mini 2>&1 | tr '\r' '\n' | tail -1; echo "pull seconds: $(( $(date +%s) - T0 ))" | tee "$O/pull.txt"
DIR=$(dirname "$(find "$LLVQ_HOME" -name config.json | head -1)")
ls -l "$DIR" | tee "$O/fileset.txt"; du -sh "$LLVQ_HOME" | tee -a "$O/fileset.txt"

stat start
ink() { env LLVQ_CPU_WEIGHTS=$1 /usr/bin/time -v /src/target/ci/inkcpu "$DIR" 32 > "$O/inkcpu-$1.txt" 2> "$O/time-$1.txt" || { echo "inkcpu $1 exit $?"; tail -5 "$O/time-$1.txt"; }
  grep -E '^phases|^cpu|loaded|load|prompt|decode|tok/s|stack|text' "$O/inkcpu-$1.txt" | cut -c1-240; grep -E 'Maximum resident|Elapsed' "$O/time-$1.txt"; }
ink packed
stat after-packed
ink nibbles
stat after-nibbles

echo '== the runtime =='
( env LLVQ_KEEP_ALIVE=-1 $LLVQ serve > "$O/server.log" 2>&1 ) &
for _ in $(seq 1 60); do curl -sS --max-time 2 "http://$LLVQ_HOST/api/version" > /dev/null 2>&1 && break; sleep 1; done
{ timeout 30m $LLVQ run inkling2:mini --verbose "Explain in about 100 words why the sky is blue." 2>&1 || echo "llvq run exit $?"; } \
  | tee "$O/run-inkling.txt" | tail -12
tail -5 "$O/server.log"
pkill -f 'llvq serve' || true
stat end
echo DONE | tee "$O/done.txt"
JOB
if [ "$DRY" = 1 ]; then
  printf 'set -euo pipefail\n%s\n%s\n' "$PRE" "$BODY" | bash -n && echo "DRY_RUN: job script parses; nothing launched" >&2
  exit 0
fi
uv run ops/run.py bench --image rust:1.95.0-bookworm --flavor "$FLAVOR" --any-flavor --timeout 90m \
  --bucket "$BUCKET" --out-mount /out --name "inkling-cpu-x86-$FLAVOR${RETRY:+-r$RETRY}" "$PRE" "$BODY"
