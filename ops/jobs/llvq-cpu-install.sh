#!/usr/bin/env bash
# What a stranger without a GPU gets: the CPU archive of `llvq` installed on a plain x86_64 machine, Qwen3 pulled by
# name and decoded on the CPU, and the parallel pull of a file set timed against the Hub. One job on a CPU flavor,
# image ubuntu:22.04: no GPU, no compiler, none of our binaries.
#   install: llvq-runtime/install.sh run as it is, with LLVQ_URL naming the archive in the bucket (ARCH_DIR, below);
#            it must pick linux-x86_64-cpu by itself, since there is no nvidia-smi;
#   `llvq pull inkling2:mini` with no server, LLVQ_PULL_JOBS=4 then =1, timed, the model removed in between;
#   `llvq run inkling2:mini`: the CPU build must refuse it by name;
#   `llvq serve`, then `llvq run qwen3:4b --verbose` on three prompts and `llvq run qwen3:8b --verbose` on one.
# Every output is saved raw under the output directory.
#
#   DRY_RUN=1 bash ops/jobs/llvq-cpu-install.sh              parses, launches nothing
#   ARCH_DIR=<directory of the bucket holding the archive, its .sha256 and install.sh> bash ops/jobs/llvq-cpu-install.sh
#
# Preregistered: PREREG below, signed and stamped before this.
# Cost, estimated: about 25 minutes of cpu-upgrade at $0.03 an hour, under $0.05; 60-minute timeout.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

PREREG=${PREREG:-proofs/preregistration-llvq-cpu-install-2026-11-25.md}
FLAVOR=${FLAVOR:-cpu-upgrade}
ARCH_DIR=${ARCH_DIR:-}
IMAGE=${IMAGE:-ubuntu:22.04}
BUCKET=Pier-Jean/jobs-artifacts
O=/out/llvq-cpu-install-$FLAVOR-2026-11-25${RETRY:+-r$RETRY}
DRY=${DRY_RUN:-0}
TIMEOUT=${TIMEOUT:-60m}

if [ "$DRY" != 1 ]; then
  [ -n "$ARCH_DIR" ] || { echo "refused: ARCH_DIR unset, the bucket directory that holds the CPU archive" >&2; exit 1; }
  { test -f "$PREREG" && test -f "$PREREG.ots"; } || { echo "refused: $PREREG(.ots) missing" >&2; exit 1; }
  if hf buckets ls "hf://buckets/$BUCKET/${O#/out/}/" 2>/dev/null | grep -v '^(empty)$' | grep -q .; then
    echo "refused: ${O#/out/}/ already exists" >&2; exit 1
  fi
fi

VER=$(sed -n 's/^version = "\(.*\)"/\1/p' llvq-runtime/Cargo.toml | head -n 1)
PRE=$(printf 'O=%q\nA=%q\nVER=%q' "$O" "/out/$ARCH_DIR" "$VER")

IFS= read -r -d '' BODY <<'JOB' || true
mkdir -p "$O"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq > /dev/null
apt-get install -y -qq --no-install-recommends curl ca-certificates > /dev/null
lscpu | tee "$O/lscpu.txt"
grep -o -w -E 'avx2|fma|avx512f|avx512bw|avx512vnni' /proc/cpuinfo | sort | uniq -c | tee "$O/simd.txt"
nproc | tee "$O/nproc.txt"; free -g | tee "$O/mem.txt"; df -h / /tmp | tee "$O/disk.txt"
{ command -v nvidia-smi cargo rustc llvq || true; } | tee "$O/system-tools.txt"

echo '== install =='
ls -l "$A"
LLVQ_URL="file://$A/llvq-$VER-linux-x86_64-cpu.tar.gz" sh "$A/install.sh" 2>&1 | tee "$O/install.txt"
command -v llvq
llvq -v | tee "$O/llvq-version.txt"

export LLVQ_HOME=/tmp/llvq-home LLVQ_HOST=127.0.0.1:11435
echo '== pull of a file set, four files at a time then one =='
for J in 4 1; do
  T0=$(date +%s.%N)
  LLVQ_PULL_JOBS=$J llvq pull inkling2:mini 2>&1 | tr '\r' '\n' | tail -3 | tee "$O/pull-inkling-j$J.txt"
  T1=$(date +%s.%N)
  echo "jobs $J: $(echo "$T1 - $T0" | awk '{printf "%.1f", $1 - $3}') s" | tee -a "$O/pull-times.txt"
  du -sh "$LLVQ_HOME" | tee -a "$O/pull-times.txt"
  if [ "$J" = 4 ]; then llvq rm inkling2:mini | tee -a "$O/pull-times.txt"; du -sh "$LLVQ_HOME" | tee -a "$O/pull-times.txt"; fi
done

echo '== the server =='
env LLVQ_KEEP_ALIVE=-1 llvq serve > "$O/server.log" 2>&1 &
SRV=$!
trap 'kill "$SRV" 2>/dev/null || true' EXIT
UP=0
for _ in $(seq 1 60); do
  if curl -sS --max-time 2 "http://$LLVQ_HOST/api/version" > "$O/version.json" 2>/dev/null; then UP=1; break; fi
  kill -0 "$SRV" 2>/dev/null || break
  sleep 1
done
[ "$UP" = 1 ] || { echo 'STOP: the server did not answer /api/version within 60 s' >&2; tail -20 "$O/server.log" >&2; exit 1; }

echo '== Inkling 2 on a CPU build: a refusal by name is expected =='
{ timeout 5m llvq run inkling2:mini "Hello" 2>&1 || echo "llvq run exit $?"; } | tee "$O/run-inkling.txt"
llvq rm inkling2:mini || true

echo '== Qwen3 4B on the CPU =='
T0=$(date +%s)
llvq pull qwen3:4b 2>&1 | tr '\r' '\n' | tail -2 | tee "$O/pull-4b.txt"
echo "pull seconds: $(( $(date +%s) - T0 ))" | tee -a "$O/pull-4b.txt"
I=0
for P in "What is the capital of France? Answer with one word." \
         "Write a Python function that returns the n-th Fibonacci number, then explain it in three sentences." \
         "Explain in about 150 words why the sky is blue."; do
  I=$((I + 1))
  { timeout 15m llvq run qwen3:4b --verbose "$P" 2>&1 || echo "llvq run exit $?"; } | tee "$O/run-4b-$I.txt"
done
curl -sS "http://$LLVQ_HOST/api/ps" | tee "$O/ps-4b.json"; echo
grep -a VmHWM "/proc/$SRV/status" | tee "$O/server-peak-rss-4b.txt"

echo '== Qwen3 8B on the CPU =='
llvq pull qwen3:8b 2>&1 | tr '\r' '\n' | tail -2 | tee "$O/pull-8b.txt"
{ timeout 20m llvq run qwen3:8b --verbose "Explain in about 150 words why the sky is blue." 2>&1 || echo "llvq run exit $?"; } | tee "$O/run-8b.txt"
grep -a VmHWM "/proc/$SRV/status" | tee "$O/server-peak-rss-8b.txt"
grep -a -i 'cpu:' "$O/server.log" | tee "$O/cpu-lines.txt"
echo DONE | tee "$O/done.txt"
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
  --name "llvq-cpu-install-$FLAVOR${RETRY:+-r$RETRY}" \
  "$PRE" "$BODY"
