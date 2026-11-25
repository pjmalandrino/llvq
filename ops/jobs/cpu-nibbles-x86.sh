#!/usr/bin/env bash
# The Tetra codes decoded once into nibbles (LLVQ_CPU_WEIGHTS=nibbles) against the served words (packed), on the
# real Qwen3-4B, one instance: the bench with its nibble rows, then `cpurun phases` packed, nibbles, packed again,
# each under /usr/bin/time for its peak memory, the two modes' greedy tokens compared, and `llvq run` in both.
#
#   DRY_RUN=1 bash ops/jobs/cpu-nibbles-x86.sh     parses, launches nothing
# Preregistered: PREREG below. Cost, estimated: 30 minutes of cpu-upgrade, $0.02; 90-minute timeout.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
PREREG=${PREREG:-proofs/preregistration-cpu-nibbles-x86-2026-11-25.md}
FLAVOR=${FLAVOR:-cpu-upgrade}
COMMIT=${COMMIT:-$(git rev-parse HEAD)}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/cpu-nibbles-x86-$FLAVOR-2026-11-25${RETRY:+-r$RETRY}
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
cargo build --profile ci --locked -p llvq-llm --bin cpurun 2>&1 | tail -1
B="cargo test --profile ci --locked -p llvq-llm --test cpu_tetra"
$B 2>&1 | grep 'test result' | tee "$O/tests.txt"
bench() { LLVQ_THREADS=$1 $B -- --ignored --nocapture bench_ 2>&1 | grep '^bench' | tee "$O/bench-t$1$2.txt" | grep -E 'isa|one 4B token|same token|gate/up 9728'; }
stat() { head -1 /proc/stat > "$O/stat-$1.txt"; }

export LLVQ_HOME=/tmp/llvq-home LLVQ_HOST=127.0.0.1:11435
LLVQ=/src/target/ci/llvq
$LLVQ pull qwen3:4b 2>&1 | tr '\r' '\n' | tail -1
SEALED=$(find "$LLVQ_HOME" -type f -size +1G | head -1)
ls -l "$SEALED" | tee "$O/sealed.txt"
export LLVQ_CONFIG=/src/configs/qwen3-4b-tetra-e4.json
P="/src/target/ci/cpurun $SEALED phases 48"

stat start
bench 8 ""
bench 1 ""
stat after-bench
ph() { env LLVQ_CPU_WEIGHTS=$1 /usr/bin/time -v $P > "$O/phases-$1$2.txt" 2> "$O/time-$1$2.txt"; grep -E '^phases|^cpu:' "$O/phases-$1$2.txt"; grep 'Maximum resident' "$O/time-$1$2.txt"; }
ph packed ""
ph nibbles ""
ph packed "-again"
stat after-phases
for W in packed nibbles; do
  LLVQ_CPU_WEIGHTS=$W LLVQ_RUN_DUMP="$O/tokens-$W.json" /src/target/ci/cpurun "$SEALED" tetra 48 2>&1 | grep 'decode over'
done
python3 - "$O" <<'PY' | tee "$O/tokens-compare.txt"
import json, sys
a, b = (json.load(open(f"{sys.argv[1]}/tokens-{w}.json"))["prompts"] for w in ("packed", "nibbles"))
same = [x["output_ids"] == y["output_ids"] for x, y in zip(a, b)]
print("tokens packed against nibbles:", "IDENTICAL" if all(same) and len(a) == len(b) else f"DIFFERENT {same}", sum(len(x["output_ids"]) for x in a))
PY

echo '== the runtime, both modes =='
for W in packed nibbles; do
  ( unset LLVQ_CONFIG; env LLVQ_KEEP_ALIVE=-1 LLVQ_CPU_WEIGHTS=$W $LLVQ serve > "$O/server-$W.log" 2>&1 ) &
  for _ in $(seq 1 60); do curl -sS --max-time 2 "http://$LLVQ_HOST/api/version" > /dev/null 2>&1 && break; sleep 1; done
  ( unset LLVQ_CONFIG
    { timeout 15m $LLVQ run qwen3:4b --verbose "Explain in about 150 words why the sky is blue." 2>&1 || echo "llvq run exit $?"; } \
      | tee "$O/run-4b-$W.txt" | grep -E 'tok/s|exit' )
  grep -a VmHWM /proc/$(pgrep -f 'llvq serve' | head -1)/status | tee "$O/server-peak-$W.txt" || true
  grep -a 'cpu:' "$O/server-$W.log" || true
  pkill -f 'llvq serve' || true
  sleep 3
done
stat end
echo DONE | tee "$O/done.txt"
JOB
if [ "$DRY" = 1 ]; then
  printf 'set -euo pipefail\n%s\n%s\n' "$PRE" "$BODY" | bash -n && echo "DRY_RUN: job script parses; nothing launched" >&2
  exit 0
fi
uv run ops/run.py bench --image rust:1.95.0-bookworm --flavor "$FLAVOR" --any-flavor --timeout 90m \
  --bucket "$BUCKET" --out-mount /out --name "cpu-nibbles-x86-$FLAVOR${RETRY:+-r$RETRY}" "$PRE" "$BODY"
