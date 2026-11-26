"""Mutants of lot B and C (prereg servi-au-bit-2026-11-26): each applied in place on the committed
llvq-runtime/src/inkling.rs by an exact replace whose site count is checked, the unit tests of the module run,
then the file restored by git checkout and checked clean. Run from the worktree root, under the guard."""
import subprocess, sys

F = "llvq-runtime/src/inkling.rs"
M = [
    ("B-key: the backend removed from the key",
     "        backend: backend.label().into(),\n", "        backend: String::new(),\n"),
    ("B-off: the CPU branch of long_prompt answers Off",
     "        (Backend::Cpu, _) => LongPrompt::BiasOnHost,\n", "        (Backend::Cpu, _) => LongPrompt::Off,\n"),
    ("C-other: the kept logits of another segment handed back",
     "match matched.len().checked_sub(1).and_then(|i| self.held[i].last.clone()) {",
     "match self.held.iter().find_map(|h| h.last.clone()) {"),
    ("C-skip: the rollback to the mark skipped",
     "let (kept, from, base) = if kept < self.held.len() || self.lm.held() != base {",
     "let (kept, from, base) = if ready.is_none() && (kept < self.held.len() || self.lm.held() != base) {"),
]
log = open(sys.argv[1], "a")
def out(s):
    print(s); log.write(s + "\n"); log.flush()
assert subprocess.run(["git", "diff", "--quiet", "--", F]).returncode == 0, "the file is not clean"
for name, a, b in M:
    s = open(F).read()
    n = s.count(a)
    out(f"== {name}: {n} site(s)")
    if n != 1:
        out("   NOT APPLIED"); continue
    open(F, "w").write(s.replace(a, b))
    assert subprocess.run(["git", "diff", "--quiet", "--", F]).returncode == 1, "the mutant did not change the file"
    r = subprocess.run(["cargo", "test", "-j", "4", "-p", "llvq-runtime", "--lib", "inkling::tests"],
                       capture_output=True, text=True)
    lines = [l for l in (r.stdout + r.stderr).splitlines()
             if l.startswith("test ") and ("FAILED" in l or "... ok" in l) or "test result" in l or l.startswith("error")]
    for l in lines:
        out("   " + l)
    out(f"   verdict: {'RED' if r.returncode != 0 else 'GREEN (survived)'}")
    subprocess.run(["git", "checkout", "--", F], check=True)
    assert subprocess.run(["git", "diff", "--quiet", "--", F]).returncode == 0, "the file did not come back"
    out("   restored, clean")
