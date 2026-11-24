#!/usr/bin/env bash
# The CPU matvec microbenchmark of llvq-llm/tests/cpu_tetra.rs on an x86_64 machine, AVX2 path, before and after the
# kernel change of docs/mesures/cpu-tetra-speed-2026-11-25.txt. Built from the public repository inside the job.
#   after:  the branch BRANCH as it is; the six correctness tests of cpu_tetra first, then the bench at 1, 4, 8 threads;
#   before: main at BASE with the bench commit BENCH picked on it, the bench at 1 and 8 threads.
#
#   DRY_RUN=1 bash ops/jobs/cpu-matvec-x86.sh     parses, launches nothing
# Preregistered: PREREG below. Cost, estimated: 40 minutes of cpu-upgrade, $0.02; 90-minute timeout.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
PREREG=${PREREG:-proofs/preregistration-cpu-matvec-x86-2026-11-25.md}
FLAVOR=${FLAVOR:-cpu-upgrade}
BRANCH=${BRANCH:-nuit/runtime-cpu-2026-11-25}
BASE=${BASE:-4071f5d}
BENCH=${BENCH:-bd5d528}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/cpu-matvec-x86-$FLAVOR-2026-11-25${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
if [ "$DRY" != 1 ]; then
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
fi
PRE=$(printf 'O=%q\nBRANCH=%q\nBASE=%q\nBENCH=%q' "$O" "$BRANCH" "$BASE" "$BENCH")
IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq > /dev/null
apt-get install -y -qq --no-install-recommends git ca-certificates pkg-config libssl-dev build-essential cmake > /dev/null
lscpu | tee "$O/lscpu.txt"; nproc | tee "$O/nproc.txt"
git clone -q https://github.com/pjmalandrino/llvq-tetra /src && cd /src
git config user.email job@example.invalid && git config user.name job
git checkout -q "origin/$BRANCH" && git rev-parse HEAD | tee "$O/after-commit.txt"
B="cargo test --profile ci -p llvq-llm --test cpu_tetra"
$B 2>&1 | tail -15 | tee "$O/after-tests.txt"
for T in 1 4 8; do LLVQ_THREADS=$T $B -- --ignored --nocapture bench_ 2>&1 | tee "$O/after-t$T.txt" | tail -14; done
git checkout -q "$BASE" && git cherry-pick "$BENCH" > /dev/null && git rev-parse HEAD | tee "$O/before-commit.txt"
for T in 1 8; do LLVQ_THREADS=$T $B -- --ignored --nocapture bench_ 2>&1 | tee "$O/before-t$T.txt" | tail -14; done
echo DONE | tee "$O/done.txt"
JOB
if [ "$DRY" = 1 ]; then
  printf 'set -euo pipefail\n%s\n%s\n' "$PRE" "$BODY" | bash -n && echo "DRY_RUN: job script parses; nothing launched" >&2
  exit 0
fi
uv run ops/run.py bench --image rust:1.95.0-bookworm --flavor "$FLAVOR" --any-flavor --timeout 90m \
  --bucket "$BUCKET" --out-mount /out --name "cpu-matvec-x86-$FLAVOR${RETRY:+-r$RETRY}" "$PRE" "$BODY"
