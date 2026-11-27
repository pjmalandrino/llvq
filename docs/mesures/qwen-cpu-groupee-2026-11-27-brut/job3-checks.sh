#!/bin/bash
# Lot G, job 3: clippy at zero warnings, cargo test in debug on llvq-llm, then
# the prereg's mutant on the committed llvq-llm/src/served.rs, seen red, restored.
set -uo pipefail
W=/Users/pjmalandrino/Documents/Pro/workspace/poc/llvq/.claude/worktrees/qwen-cpu
R=$W/docs/mesures/qwen-cpu-groupee-2026-11-27-brut
cd "$W"
T=$R/tlim.sh
$R/waitload.sh 600
echo "== clippy $(date '+%F %T')"
$T 3600 cargo clippy --all-targets > "$R/clippy.log" 2>&1; echo "clippy rc=$?"
grep -c "^warning" "$R/clippy.log" || true
echo "== cargo test -p llvq-llm (debug) $(date '+%F %T')"
$T 3600 cargo test -p llvq-llm > "$R/test-llvq-llm.log" 2>&1; echo "test rc=$?"
grep -c "^test result:" "$R/test-llvq-llm.log"; grep "^test result:" "$R/test-llvq-llm.log" | grep -vc " 0 failed" || true
echo "== mutant $(date '+%F %T')"
git diff --quiet -- llvq-llm/src/served.rs || { echo "served.rs not clean, mutant refused"; exit 2; }
sites=$(grep -c 'model.set_attn_heads(self.attention);' llvq-llm/src/served.rs); echo "sites before: $sites"
[ "$sites" = 1 ] || { echo "expected 1 site"; exit 2; }
sed -i '' 's/model.set_attn_heads(self.attention);/model.set_attn_heads(AttnHeads::Expanded);/' llvq-llm/src/served.rs
echo "sites after: $(grep -c 'model.set_attn_heads(AttnHeads::Expanded);' llvq-llm/src/served.rs) mutated, $(grep -c 'model.set_attn_heads(self.attention);' llvq-llm/src/served.rs) left"
git diff --stat -- llvq-llm/src/served.rs
$T 3600 cargo test -p llvq-llm --lib a_served_config_sets_the_store_and_the_attention_form > "$R/mutant.log" 2>&1; echo "mutant test rc=$? (non-zero = red)"
grep -E "^test result:|panicked|left:|right:" "$R/mutant.log" | head -8
git checkout -- llvq-llm/src/served.rs
git diff --quiet -- llvq-llm/src/served.rs && echo "restored, served.rs clean"
echo "== done $(date '+%F %T')"
