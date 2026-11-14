import hashlib, subprocess, sys
from pathlib import Path
P = Path("/Users/pjmalandrino/Documents/Pro/workspace/poc/llvq/.claude/worktrees/recherche-qualite/ops/layer_vs_output.py")
CMD = ["uv", "run", "--offline", "--no-project", "--python", "3.12", "--with", "torch==2.5.1", "--with", "safetensors==0.4.5",
       "--with", "tokenizers==0.22.2", "--with", "pyarrow==20.0.0", "--with", "huggingface-hub==1.19.0",
       "python", "-I", str(P), "--selftest"]
MUTANTS = [
    ("M1 swap ignored in the resolver", "            src = other if name in swapped else base\n",
     "            src = base\n"),
    ("M2 note 1 denominator on W_hat", "        y_ref = xa @ w.T\n", "        y_ref = xa @ w_hat.T\n"),
    ("M3 v_proj dropped from qkv", '    "qkv": ("q_proj", "k_proj", "v_proj"),\n', '    "qkv": ("q_proj", "k_proj"),\n'),
]
orig = P.read_bytes()
h0 = hashlib.sha256(orig).hexdigest()
print("original sha256", h0)
ok_all = True
for name, a, b in MUTANTS:
    src = orig.decode()
    n = src.count(a)
    print(f"\n== {name}: site count {n}")
    assert n == 1, "site not unique"
    P.write_text(src.replace(a, b))
    assert hashlib.sha256(P.read_bytes()).hexdigest() != h0, "mutant not applied"
    r = subprocess.run(CMD, capture_output=True, text=True, cwd="/private/tmp")
    fails = [l for l in r.stdout.splitlines() if "FAIL" in l]
    last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[-300:]
    caught = r.returncode != 0 and bool(fails)
    print(f"exit {r.returncode}; {last}")
    for l in fails:
        print("  " + l.strip())
    print("CAUGHT" if caught else "SURVIVED")
    ok_all &= caught
    P.write_bytes(orig)
    assert hashlib.sha256(P.read_bytes()).hexdigest() == h0
    print("restored, sha256", hashlib.sha256(P.read_bytes()).hexdigest())
print("\nall mutants caught" if ok_all else "\nA MUTANT SURVIVED")
sys.exit(0 if ok_all else 1)
