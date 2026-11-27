#!/bin/bash
# The three mutants of prereg inkling-fenetre-2026-11-27, in one job under the guard's lock, from the worktree root.
# Each is an in-place replacement on the committed layers.rs, its sites counted (zero sites stops the script), then
# tests/inkling_window.rs (debug, metal) must go red, then git checkout restores the file and the tree is checked clean.
set -u
F=llvq-llm/src/inkling/layers.rs
git diff --quiet -- $F || { echo "mutants: $F is not the committed file"; exit 1; }
mutant() {
  local name=$1 old=$2 new=$3
  local n
  n=$(OLD="$old" NEW="$new" python3 -I -c '
import os, sys
p = sys.argv[1]; s = open(p).read(); o, n = os.environ["OLD"], os.environ["NEW"]
c = s.count(o); open(p, "w").write(s.replace(o, n)); print(c)' $F)
  echo "=== mutant $name: $n site(s) replaced: '$old' -> '$new'"
  if [ "$n" = 0 ]; then echo "mutants: STOP, $name matched nothing"; git checkout -- $F; exit 1; fi
  perl -e 'alarm shift; exec @ARGV or die' 1200 cargo test -p llvq-llm --features metal --test inkling_window 2>&1 \
    | grep -E "^test |test result:|worst relative gap|panicked|error(\[|:)"
  echo "=== mutant $name: test exit ${PIPESTATUS[0]}"
  git checkout -- $F
  git diff --quiet -- $F && echo "=== mutant $name: $F restored, clean" || { echo "mutants: STOP, $F not restored"; exit 1; }
}
mutant late-start '(pos + 1).saturating_sub(window)' '(pos + 2).saturating_sub(window)'
mutant unshifted-bias 'score_bias(&rc, p0 - (from + lo), len' 'score_bias(&rc, p0 - from, len'
mutant full-narrowed 'if self.sliding && window_scores() {' 'if window_scores() {'
echo "=== the committed file, the same test"
perl -e 'alarm shift; exec @ARGV or die' 1200 cargo test -p llvq-llm --features metal --test inkling_window 2>&1 \
  | grep -E "^test |test result:|panicked"
echo "mutants: done; tree: $(git status --short llvq-llm | tr '\n' ' ')"
echo "=== lints on the committed tree"
perl -e 'alarm shift; exec @ARGV or die' 1200 cargo clippy -p llvq-llm --features metal --all-targets > docs/mesures/inkling-mac-fenetre-2026-11-27-brut/clippy-final.log 2>&1
echo "clippy metal rc=$? warnings $(grep -c '^warning' docs/mesures/inkling-mac-fenetre-2026-11-27-brut/clippy-final.log)"
perl -e 'alarm shift; exec @ARGV or die' 1200 ops/check-linux-cuda.sh clippy -p llvq-llm --features cuda --all-targets > docs/mesures/inkling-mac-fenetre-2026-11-27-brut/linux-check-final.log 2>&1
echo "linux check rc=$? warnings $(grep -c '^warning' docs/mesures/inkling-mac-fenetre-2026-11-27-brut/linux-check-final.log)"
