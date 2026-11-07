#!/usr/bin/env bash
# Prereg: proofs/preregistration-draft-0p6b-2026-11-07.md, stamped before this ran.
set -uo pipefail
OUT=$HOME/q06b-tetra-2026-11-07
WT=/Users/pjmalandrino/Documents/Pro/workspace/poc/llvq/.claude/worktrees/spec-0p6b-96ba73
B=$WT/target/release
caffeinate -i -w $$ &
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* inherited' >&2; exit 1; fi
cd "$WT"
git rev-parse HEAD > $OUT/commit.txt
date -u +%FT%TZ > $OUT/start.txt
echo "== build $(date -u +%FT%TZ)" | tee -a $OUT/queue.log
nice -n 10 cargo build --release -p llvq-llm --features metal,fast-linalg --bin oracle --bin ppl --bin smoke --bin seal --bin embedq > $OUT/build.log 2>&1 || { echo build failed; exit 1; }
nice -n 10 cargo build --release -p llvq-bench --bin rtbits >> $OUT/build.log 2>&1 || echo "rtbits build failed" | tee -a $OUT/queue.log
echo "== oracle $(date -u +%FT%TZ)" | tee -a $OUT/queue.log
nice -n 10 $B/oracle Qwen/Qwen3-0.6B 64 metal > $OUT/oracle.txt 2>&1; echo "oracle exit $?" | tee -a $OUT/queue.log
grep -q MATCH $OUT/oracle.txt || { echo 'refused: oracle did not print MATCH' | tee -a $OUT/queue.log; exit 1; }
echo "== smoke $(date -u +%FT%TZ)" | tee -a $OUT/queue.log
env LLVQ_MODEL=Qwen/Qwen3-0.6B LLVQ_CALIB=dclm-edu LLVQ_INT4_TYPES=v_proj,o_proj LLVQ_THREADS=12 \
    LLVQ_ARTIFACT=$OUT/qwen3-0p6b-dclm.llvq \
  nice -n 10 $B/smoke 64 2048 12 4096 metal nogs tetra1 999 rot > $OUT/smoke.txt 2>&1
echo "smoke exit $? $(date -u +%FT%TZ)" | tee -a $OUT/queue.log
echo "== seal $(date -u +%FT%TZ)" | tee -a $OUT/queue.log
env LLVQ_MODEL=Qwen/Qwen3-0.6B nice -n 10 $B/seal $OUT/qwen3-0p6b-dclm.llvq $OUT/qwen3-0p6b-sealed-f16.bin > $OUT/seal.txt 2>&1
echo "seal exit $?" | tee -a $OUT/queue.log
nice -n 10 $B/embedq $OUT/qwen3-0p6b-sealed-f16.bin $OUT/qwen3-0p6b-sealed.bin q4 > $OUT/embedq.txt 2>&1
echo "embedq exit $?" | tee -a $OUT/queue.log
ls -l $OUT/*.bin $OUT/*.llvq | tee -a $OUT/queue.log
shasum -a 256 $OUT/qwen3-0p6b-sealed.bin | tee $OUT/sealed.sha256
[ -x $B/rtbits ] && $B/rtbits $OUT/qwen3-0p6b-sealed.bin > $OUT/rtbits.txt 2>&1
echo "rtbits exit $?" | tee -a $OUT/queue.log
echo "== ppl $(date -u +%FT%TZ)" | tee -a $OUT/queue.log
env LLVQ_DTYPE=f16 nice -n 10 $B/ppl 4096 12 metal $OUT/qwen3-0p6b-sealed.bin > $OUT/ppl.txt 2>&1
echo "ppl exit $? $(date -u +%FT%TZ)" | tee -a $OUT/queue.log
date -u +%FT%TZ > $OUT/done.txt
