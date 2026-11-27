#!/bin/bash
# The measured part of prereg inkling-fenetre-2026-11-27 in one job under the guard's lock, from the worktree root:
#   ~/.llvq-guard/heavy.sh "lotD: ..." docs/mesures/inkling-mac-fenetre-2026-11-27-brut/all.sh < /dev/null
# The CUDA half type-checked, the release build of inkrun copied to target/lotd/inkrun (lot A builds in this
# worktree), gate D-ii (the arm at 2,000 and 9,000 tokens against the digests of 2026-11-26), then gate D-iii:
# 1 discarded + 4 timed pairs at 9,000 tokens, then 1 + 4 at 2,000, reference first in each pair.
set -u
B=docs/mesures/inkling-mac-fenetre-2026-11-27-brut
export INLOCK=1
echo "all: start $(date '+%F %T') at $(git rev-parse HEAD); tree: $(git status --short llvq-llm llvq-runtime configs | tr '\n' ' ')"
perl -e 'alarm shift; exec @ARGV or die' 1500 ops/check-linux-cuda.sh clippy -p llvq-llm --features cuda --all-targets \
  > $B/linux-check.log 2>&1; echo "all: linux check rc=$?"
perl -e 'alarm shift; exec @ARGV or die' 1500 cargo build --release -p llvq-llm --features metal --bin inkrun \
  > $B/build.log 2>&1 || { echo "all: STOP build"; exit 1; }
mkdir -p target/lotd && cp target/release/inkrun target/lotd/inkrun && shasum -a 256 target/lotd/inkrun | tee $B/inkrun.sha256
A=$B/arms.sh
$A 9000 dii-arm9000="--attn-window narrow"
sleep 45
$A 2000 dii-arm2000="--attn-window narrow"
sleep 45
$A 9000 t9-ref0= t9-arm0="--attn-window narrow" t9-ref1= t9-arm1="--attn-window narrow" \
  t9-ref2= t9-arm2="--attn-window narrow" t9-ref3= t9-arm3="--attn-window narrow" t9-ref4= t9-arm4="--attn-window narrow"
sleep 45
$A 2000 t2-ref0= t2-arm0="--attn-window narrow" t2-ref1= t2-arm1="--attn-window narrow" \
  t2-ref2= t2-arm2="--attn-window narrow" t2-ref3= t2-arm3="--attn-window narrow" t2-ref4= t2-arm4="--attn-window narrow"
echo "all: done $(date '+%F %T')"
