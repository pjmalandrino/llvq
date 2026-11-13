import subprocess, sys, shutil, hashlib, os
S=os.environ['S']
L=S+'/logs'
muts=[
 ("M1 weights read but not applied","llvq-llm/src/calib.rs","h.accumulate_weighted(x, &wt)?;","h.accumulate(x)?; let _ = &wt;"),
 ("M2 row groups swapped","llvq-llm/src/calib.rs","for (range, factor) in ranges.into_iter().zip(factors) {","for (range, factor) in ranges.into_iter().zip(factors.iter().rev()) {"),
 ("M3 normalisation dropped","llvq-llm/src/guided.rs","out.push(w.into_iter().map(|x| (x / mean) as f32).collect());","out.push(w.into_iter().map(|x| x as f32).collect());"),
 ("M4 token cursor not advanced","llvq-llm/src/calib.rs","self.cursor = start + rows;","self.cursor = start;"),
 ("M5 every block reads layer 0","llvq-llm/src/calib.rs","g.load(t, name)?","g.load(0, name)?"),
 ("M6 block t reads layer t-1","llvq-llm/src/calib.rs","g.load(t, name)?","g.load(t - 1, name)?"),
 ("M7 gate and up swapped","llvq-llm/src/calib.rs","g.load(t, name)?","g.load(t, if name == crate::guided::TYPES[0] { crate::guided::TYPES[1] } else { crate::guided::TYPES[0] })?"),
]
def sha(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()
for name,path,old,new in muts:
    orig=open(path).read(); h0=sha(path)
    n=orig.count(old)
    print(f"== {name}: {n} site(s) in {path}", flush=True)
    assert n==1
    mutated=orig.replace(old,new)
    assert mutated.count(old)==0 and mutated!=orig
    open(path,'w').write(mutated)
    try:
      log=f"{L}/mutant-{name.split()[0]}.log"
      r=subprocess.run(["cargo","test","-p","llvq-llm","--lib","--test","guided","--","guided::","unit_weights","zeroing","two_identical","each_row_group","layers_and_types","fixture","malformed","unset_encode"],capture_output=True,text=True,env=dict(os.environ,CARGO_PROFILE_DEV_DEBUG="0",CARGO_INCREMENTAL="0"))
      open(log,'w').write(r.stdout+r.stderr)
    finally:
      open(path,'w').write(orig)
    assert sha(path)==h0, "restore failed"
    fails=[l for l in (r.stdout+r.stderr).splitlines() if l.startswith("test ") and l.endswith("FAILED")]
    results=[l for l in (r.stdout+r.stderr).splitlines() if "test result:" in l]
    print(f"   rc={r.returncode}; restored sha ok; results: {results}", flush=True)
    for f in fails: print("   caught by:", f, flush=True)
