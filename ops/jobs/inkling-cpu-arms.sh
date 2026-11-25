#!/usr/bin/env bash
# Inkling 2 Mini on a CPU, the real file set, one arm a commit on one instance: `inkcpu <file set> 32` with the codes
# packed, under /usr/bin/time, for each commit of ARMS in order, then the first arm again for the drift.
#
#   DRY_RUN=1 bash ops/jobs/inkling-cpu-arms.sh     parses, launches nothing
# Preregistered: PREREG below. Cost, estimated: 40 minutes of cpu-upgrade, $0.02; 90-minute timeout.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
PREREG=${PREREG:-proofs/preregistration-inkling-cpu-arms-2026-11-25.md}
FLAVOR=${FLAVOR:-cpu-upgrade}
COMMIT=${COMMIT:-$(git rev-parse HEAD)}
ARMS=${ARMS:-"86361cf 72b16c5 2e3c3dc"}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/inkling-cpu-arms-$FLAVOR-2026-11-25${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  git branch -r --contains "$COMMIT" | grep -q . || { echo "refused: $COMMIT is not pushed" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
fi
PRE=$(printf 'O=%q\nCOMMIT=%q\nARMS=%q' "$O" "$COMMIT" "$ARMS")
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
free -g | tee "$O/mem.txt"
export LLVQ_HOME=/tmp/llvq-home
/src/target/ci/llvq pull inkling2:mini 2>&1 | tr '\r' '\n' | tail -1
DIR=$(dirname "$(find "$LLVQ_HOME" -name config.json | head -1)")
arm() { git checkout -q "$1" && git rev-parse HEAD > "$O/$2-commit.txt"
  cargo build --profile ci --locked -p llvq-llm --bin inkcpu 2>&1 | tail -1
  head -1 /proc/stat > "$O/$2-stat-before.txt"
  /usr/bin/time -v /src/target/ci/inkcpu "$DIR" 32 > "$O/$2-inkcpu.txt" 2> "$O/$2-time.txt" || { echo "inkcpu $2 exit $?"; tail -5 "$O/$2-time.txt"; }
  head -1 /proc/stat > "$O/$2-stat-after.txt"
  echo "== arm $2 =="; grep -E '^phases|^cpu|^load|^prompt|^decode|^generated' "$O/$2-inkcpu.txt" | cut -c1-200; grep 'Maximum resident' "$O/$2-time.txt"; }
for A in $ARMS; do arm "$A" "$A"; done
arm "${ARMS%% *}" "${ARMS%% *}-again"
cargo test --profile ci --locked -p llvq-llm --test inkling_cpu --test cpu_tetra 2>&1 | grep 'test result' | tee "$O/tests.txt"
echo DONE | tee "$O/done.txt"
JOB
if [ "$DRY" = 1 ]; then
  printf 'set -euo pipefail\n%s\n%s\n' "$PRE" "$BODY" | bash -n && echo "DRY_RUN: job script parses; nothing launched" >&2
  exit 0
fi
uv run ops/run.py bench --image rust:1.95.0-bookworm --flavor "$FLAVOR" --any-flavor --timeout 90m \
  --bucket "$BUCKET" --out-mount /out --name "inkling-cpu-arms-$FLAVOR${RETRY:+-r$RETRY}" "$PRE" "$BODY"
