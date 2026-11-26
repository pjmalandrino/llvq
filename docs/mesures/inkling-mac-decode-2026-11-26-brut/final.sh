#!/bin/bash
# End of lot 3, from the worktree root: twin tests on the new default, K2's 5 rounds on it, then the server's 13 checks.
set -u
B=docs/mesures/inkling-mac-decode-2026-11-26-brut
quiet() { until awk -v l="$(sysctl -n vm.loadavg | awk '{print $2}')" 'BEGIN{exit !(l<6)}'; do sleep 10; done; }
quiet; ~/.llvq-guard/heavy.sh "mac-inkling: twin tests, new default" cargo test --release -p llvq-llm --features metal \
  --test inkling_decode_kernels --test inkling_wave2_moe > $B/tests-final.log 2>&1
$B/rounds.sh k2 5 ENV=CANDLE_METAL_COMPUTE_PER_BUFFER=30 || exit 1
quiet; ~/.llvq-guard/heavy.sh "mac-inkling: serve-mac 13 checks" bash ops/inkling/serve-mac.sh target/release/llvq \
  "$HOME/.llvq/models/sha256-1e9314dba89332cb1af444619dc5d31a2a1ed6a50a1112e0ca2b8fc55bb6bfd3/inkling.json" $B/serve > $B/serve.log 2>&1
echo "serve rc=$?" >> $B/serve.log
echo FINAL-DONE
