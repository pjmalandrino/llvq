#!/bin/bash
# End of lot 3 with both defaults (down one simdgroup a slot, 30 encoders a buffer), from the worktree root:
# build, three decode runs of the new default, then the server's 13 checks.
set -u
B=docs/mesures/inkling-mac-decode-2026-11-26-brut
quiet() { until awk -v l="$(sysctl -n vm.loadavg | awk '{print $2}')" 'BEGIN{exit !(l<6)}'; do sleep 10; done; }
quiet; nice -n 10 cargo build -j 4 --release -p llvq-llm -p llvq-runtime --features llvq-llm/metal,llvq-runtime/metal \
  --bin inkrun --bin llvq > $B/build-final.log 2>&1 || { echo STOP build; exit 1; }
for r in 1 2 3; do $B/decode_arm.sh fin-r$r || { echo STOP fin-r$r; exit 1; }; done
quiet; ~/.llvq-guard/heavy.sh "mac-inkling: serve-mac 13 checks" bash ops/inkling/serve-mac.sh target/release/llvq \
  "$HOME/.llvq/models/sha256-1e9314dba89332cb1af444619dc5d31a2a1ed6a50a1112e0ca2b8fc55bb6bfd3/inkling.json" $B/serve > $B/serve.log 2>&1
echo "serve rc=$?"
echo FINAL-DONE
