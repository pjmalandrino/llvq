import subprocess, hashlib, os
p="ops/guided_weights.py"
muts=[("P1 square replaced by abs","g.pow(2).view(t, G_MAX, d // G_MAX).sum(-1)","g.abs().view(t, G_MAX, d // G_MAX).sum(-1)"),
      ("P2 rows interleaved","g.pow(2).view(t, G_MAX, d // G_MAX).sum(-1)","g.pow(2).view(t, d // G_MAX, G_MAX).sum(1)"),
      ("P3 gate and up probes swapped","            g = g + pg\n            u = u + pu\n","            g = g + pu\n            u = u + pg\n")]
orig=open(p).read(); h0=hashlib.sha256(orig.encode()).hexdigest()
out=os.environ["OUT"]
for name,old,new in muts:
    assert orig.count(old)==1, name
    m=orig.replace(old,new); assert m.count(old)==0
    open(p,"w").write(m)
    try:
        r=subprocess.run(["uv","run","ops/guided_weights.py","--selftest"],capture_output=True,text=True)
    finally:
        open(p,"w").write(orig)
    assert hashlib.sha256(open(p).read().encode()).hexdigest()==h0
    lines=[l for l in (r.stdout+r.stderr).splitlines() if "FAIL" in l or "selftest:" in l or "Error" in l]
    print(f"== {name}: rc={r.returncode}; restored, sha256 {h0[:16]} checked", flush=True)
    for l in lines: print("   ", l, flush=True)
