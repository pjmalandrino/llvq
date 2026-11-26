"""The mutants of prereg inkling-mac-prefix-2026-11-26, one at a time: apply (the site counted), run the test that
should go red, restore with git checkout. From the worktree root. Writes mutants.log beside this file."""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = open(os.path.join(HERE, "mutants.log"), "w")
MUTANTS = [
    ("M1 a different stored token accepted by the comparison", "llvq-runtime/src/prefix.rs",
     "stored.zip(&ids[..f.b]).position", "stored.zip(&ids[..f.b - 1]).position",
     ["cargo", "test", "-j", "4", "-p", "llvq-runtime", "--features", "metal", "--lib", "prefix::"]),
    ("M2 integrity check skipped", "llvq-runtime/src/prefix.rs",
     'anyhow::ensure!(digest == trailer, "its sha256 does not hold: the file is damaged");',
     "let _ = (digest, trailer);",
     ["cargo", "test", "-j", "4", "-p", "llvq-runtime", "--features", "metal", "--lib", "prefix::"]),
    ("M2b integrity check skipped, seen by the server test", "llvq-runtime/src/prefix.rs",
     'anyhow::ensure!(digest == trailer, "its sha256 does not hold: the file is damaged");',
     "let _ = (digest, trailer);",
     ["cargo", "test", "-j", "4", "-p", "llvq-runtime", "--features", "metal", "--test", "inkling_prefix"]),
    ("M3 save point moved by one piece", "llvq-runtime/src/prefix.rs",
     "Some(l / p * p).filter(|&b| b >= MIN_TOKENS)", "Some(l / p * p + p).filter(|&b| b >= MIN_TOKENS)",
     ["cargo", "test", "-j", "4", "-p", "llvq-runtime", "--features", "metal", "--lib", "prefix::"]),
    ("M4 a setting (kv_slack) removed from the key comparison", "llvq-runtime/src/prefix.rs",
     ".filter(|k| a.get(*k) != b.get(*k))", ".filter(|k| a.get(*k) != b.get(*k) && k.as_str() != \"kv_slack\")",
     ["cargo", "test", "-j", "4", "-p", "llvq-runtime", "--features", "metal", "--lib", "prefix::"]),
    ("M5 a loaded slot's buffers sized to what it holds, not to the saved capacity", "llvq-llm/src/inkling/layers.rs",
     "let kb = get_padded(r, nkv * d, held, 1, capacity, dev)?.reshape((nkv, d, capacity))?;\n                let vb = get_padded(r, nkv, held, d, capacity, dev)?;",
     "let kb = get_padded(r, nkv * d, held, 1, held, dev)?.reshape((nkv, d, held))?;\n                let vb = get_padded(r, nkv, held, d, held, dev)?;",
     ["cargo", "test", "-j", "4", "-p", "llvq-llm", "--features", "metal", "--test", "inkling_prefix_state"]),
]
only = sys.argv[1:] 
for name, path, a, b, cmd in MUTANTS:
    if only and name.split()[0] not in only:
        continue
    s = open(path).read()
    n = s.count(a)
    LOG.write(f"== {name}\n   {path}: {n} site(s)\n")
    if n != 1:
        LOG.write("   NOT APPLIED\n"); LOG.flush(); continue
    open(path, "w").write(s.replace(a, b))
    try:
        r = subprocess.run(["nice", "-n", "10"] + cmd, capture_output=True, text=True)
        out = r.stdout + r.stderr
        lines = [l for l in out.splitlines() if l.startswith("test ") or "test result" in l or "panicked" in l]
        LOG.write("   exit %d\n" % r.returncode + "".join(f"   {l}\n" for l in lines[:20]))
        LOG.write(f"   verdict: {'KILLED (red)' if r.returncode != 0 else 'SURVIVED (green)'}\n")
    finally:
        subprocess.run(["git", "checkout", "--", path], check=True)
        assert open(path).read() == s, "restore failed"
    LOG.flush()
LOG.close()
print(open(os.path.join(HERE, "mutants.log")).read())
