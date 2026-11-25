#!/usr/bin/env bash
# The CPU matvec microbenchmark of llvq-llm/tests/cpu_tetra.rs on an x86_64 machine, one arm a commit, same job.
# For each commit of ARMS, in order: the six correctness tests of cpu_tetra, then the bench at 1, 4 and 8 threads.
# The first arm is run again at the end, at 1 thread, to read the drift of the machine. /proc/stat is saved around
# each arm for the steal time.
#
#   DRY_RUN=1 bash ops/jobs/cpu-matvec-x86-arms.sh     parses, launches nothing
# Preregistered: PREREG below. Cost, estimated: 45 minutes of cpu-upgrade, $0.02; 90-minute timeout.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
PREREG=${PREREG:-proofs/preregistration-cpu-matvec-x86-arms-2026-11-25.md}
FLAVOR=${FLAVOR:-cpu-upgrade}
ARMS=${ARMS:-"f8f27a0 d47526c b4a6684"}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/cpu-matvec-x86-arms-$FLAVOR-2026-11-25${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
fi
PRE=$(printf 'O=%q\nARMS=%q' "$O" "$ARMS")
IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
# The launcher's shell does not carry the image's PATH.
export PATH=/usr/local/cargo/bin:$PATH RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq > /dev/null
apt-get install -y -qq --no-install-recommends git ca-certificates pkg-config libssl-dev build-essential cmake > /dev/null
lscpu | tee "$O/lscpu.txt" | grep -E 'Model name|Thread|^CPU\(s\)'; nproc | tee "$O/nproc.txt"
grep -o -w -E 'avx2|fma|f16c|avx512f' /proc/cpuinfo | sort | uniq -c | tee "$O/simd.txt"
git clone -q https://github.com/pjmalandrino/llvq-tetra /src && cd /src
B="cargo test --profile ci -p llvq-llm --test cpu_tetra"
for A in $ARMS; do
  git checkout -q "$A" && git rev-parse HEAD | tee "$O/$A-commit.txt"
  $B 2>&1 | grep 'test result' | tee "$O/$A-tests.txt"
  head -1 /proc/stat > "$O/$A-stat-before.txt"
  for T in 1 4 8; do LLVQ_THREADS=$T $B -- --ignored --nocapture bench_ 2>&1 | grep '^bench' | tee "$O/$A-t$T.txt" | grep -E 'isa|one 4B token'; done
  head -1 /proc/stat > "$O/$A-stat-after.txt"
done
A=${ARMS%% *}
git checkout -q "$A"
LLVQ_THREADS=1 $B -- --ignored --nocapture bench_ 2>&1 | grep '^bench' | tee "$O/$A-t1-again.txt" | grep -E 'isa|one 4B token'
echo DONE | tee "$O/done.txt"
JOB
if [ "$DRY" = 1 ]; then
  printf 'set -euo pipefail\n%s\n%s\n' "$PRE" "$BODY" | bash -n && echo "DRY_RUN: job script parses; nothing launched" >&2
  exit 0
fi
uv run ops/run.py bench --image rust:1.95.0-bookworm --flavor "$FLAVOR" --any-flavor --timeout 90m \
  --bucket "$BUCKET" --out-mount /out --name "cpu-matvec-x86-arms-$FLAVOR${RETRY:+-r$RETRY}" "$PRE" "$BODY"
