#!/bin/bash
# The two mutants of prereg inkling-fenetre-runtime-2026-11-27, in one job under the guard's lock, from the worktree
# root. Each is an in-place replacement on the committed inkling.rs, its sites counted (anything but one site stops the
# script), then the runtime's unit tests (debug, metal) must go red, then git checkout restores the file and the tree
# is checked clean. Then the debug test suites of the prereg, on the committed tree, logs kept whole.
set -u
B=docs/mesures/inkling-mac-fenetre-runtime-2026-11-27-brut
F=llvq-runtime/src/inkling.rs
git diff --quiet -- $F || { echo "mutants: $F is not the committed file"; exit 1; }
echo "mutants: start $(date '+%F %T') at $(git rev-parse HEAD)"
mutant() {
  local name=$1 old=$2 new=$3
  local n
  n=$(OLD="$old" NEW="$new" python3 -I -c '
import os, sys
p = sys.argv[1]; s = open(p).read(); o, n = os.environ["OLD"], os.environ["NEW"]
c = s.count(o); open(p, "w").write(s.replace(o, n)); print(c)' $F)
  echo "=== mutant $name: $n site(s) replaced: '$old' -> '$new'"
  if [ "$n" != 1 ]; then echo "mutants: STOP, $name matched $n sites"; git checkout -- $F; exit 1; fi
  perl -e 'alarm shift; exec @ARGV or die' 1200 cargo test -p llvq-runtime --features metal --lib > $B/mutant-$name.log 2>&1
  echo "=== mutant $name: test exit $?"
  grep -E "FAILED|failed|test result:" $B/mutant-$name.log | grep -v "^test .* ok$"
  git checkout -- $F
  git diff --quiet -- $F && echo "=== mutant $name: $F restored, clean" || { echo "mutants: STOP, $F not restored"; exit 1; }
}
mutant key-arm-constant 'attn_window: window.label().into(),' 'attn_window: AttnWindow::Full.label().into(),'
mutant parsed-not-applied '(AttnWindow::Narrow, Backend::Metal, true) => Ok(true),' '(AttnWindow::Narrow, Backend::Metal, true) => Ok(false),'
echo "mutants: done; tree: [$(git status --short llvq-llm llvq-runtime configs | tr '\n' ' ')]"
echo "=== tests in debug on the committed tree $(date '+%F %T')"
perl -e 'alarm shift; exec @ARGV or die' 1500 cargo test -p llvq-runtime > $B/tests-runtime.log 2>&1; echo "cargo test -p llvq-runtime rc=$?"
perl -e 'alarm shift; exec @ARGV or die' 1500 cargo test -p llvq-runtime --features metal > $B/tests-runtime-metal.log 2>&1; echo "cargo test -p llvq-runtime --features metal rc=$?"
perl -e 'alarm shift; exec @ARGV or die' 2700 cargo test -p llvq-llm --features metal > $B/tests-llm-metal.log 2>&1; echo "cargo test -p llvq-llm --features metal rc=$?"
for f in tests-runtime tests-runtime-metal tests-llm-metal; do
  python3 -I - $B/$f.log <<'PY'
import re, sys
p = q = f = ig = 0; n = 0
for l in open(sys.argv[1], errors="replace"):
    m = re.search(r"test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored", l)
    if m: n += 1; p += int(m[2]); f += int(m[3]); ig += int(m[4])
print(f"{sys.argv[1]}: {n} 'test result:' lines, {p} passed, {f} failed, {ig} ignored")
PY
done
echo "mutants+tests: done $(date '+%F %T')"
