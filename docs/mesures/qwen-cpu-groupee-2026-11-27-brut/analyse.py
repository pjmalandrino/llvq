#!/usr/bin/env python3
"""Lot G: reads the raw files of this directory and prints every computed number of the journal.
usage: python3 -I analyse.py <tag> <rounds>   (tag 4b or 8b)"""
import json, re, statistics, sys, os, hashlib
D = os.path.dirname(os.path.abspath(__file__)); tag = sys.argv[1]; rounds = int(sys.argv[2])
def p(*a): print(*a)
# G-i
L = os.path.join(D, f"gi-lockstep-{tag}.log")
if os.path.exists(L):
    steps = [l for l in open(L) if l.startswith("step ")]
    ds = [float(l.split()[3]) for l in steps]
    differ = sum(l.rstrip().endswith("differ") for l in steps)
    arg = sum(l.split()[5] != l.split()[6] for l in steps)
    mx = [max(float(l.split()[8]), float(l.split()[9])) for l in steps]
    first = next((int(l.split()[1]) for l in steps if l.rstrip().endswith("differ")), None)
    q = sorted(ds)
    p(f"G-i {tag}: {len(steps)} steps, bits differ {differ} (first at step {first}), argmax differ {arg}, "
      f"max|dlogit| {max(ds):.6e} at step {ds.index(max(ds))}, median {statistics.median(ds):.3e}, p99 {q[int(0.99*len(q))-1]:.3e}, "
      f"largest |logit| {max(mx):.6e}, ratio {max(ds)/max(mx):.2e}")
    for lo, hi in [(0, 256), (256, 1024), (1024, 2048)]:
        seg = ds[lo:hi]
        if seg: p(f"   steps {lo}-{hi-1}: max {max(seg):.3e} median {statistics.median(seg):.3e}")
    p("  ", [l.strip() for l in open(L) if l.startswith("lockstep:")][-1])
# G-ii
for pr in ["short", "2048"]:
    a, b = (os.path.join(D, f"gii-{tag}-{arm}-{pr}.json") for arm in ("exp", "grp"))
    if os.path.exists(a) and os.path.exists(b):
        x, y = (json.load(open(f))["run"]["output_ids"] for f in (a, b))
        h = hashlib.sha256(json.dumps(x).encode()).hexdigest()[:8]
        p(f"G-ii {tag} {pr}: {len(x)} and {len(y)} tokens, equal {x == y}, digest of the ids {h}")
# G-iii
def run(r, arm, pr):
    f = os.path.join(D, f"giii-{tag}-r{r}-{arm}-{pr}")
    j = json.load(open(f + ".json"))["run"]; log = open(f + ".log").read()
    peak = int(re.search(r"(\d+)\s+peak memory footprint", log).group(1))
    rss = int(re.search(r"(\d+)\s+maximum resident set size", log).group(1))
    load = re.search(r"load \{ ([\d.]+)", log).group(1)
    return dict(tok=j["decode_steps"] / j["decode_s"], ttft=j["first_token_s"], peak=peak, rss=rss, ids=j["output_ids"], load=load)
def med(v): return f"{statistics.median(v):.3f} [{min(v):.3f}-{max(v):.3f}]"
for pr in ["short", "2048"]:
    try: R = {arm: [run(r, arm, pr) for r in range(rounds + 1)] for arm in ("exp", "grp")}
    except FileNotFoundError: continue
    p(f"G-iii {tag} {pr} (round 0 discarded): loads exp {[x['load'] for x in R['exp']]} grp {[x['load'] for x in R['grp']]}")
    same = all(x["ids"] == R["exp"][0]["ids"] for arm in R for x in R[arm])
    p(f"   every run of both arms returns the same ids: {same}")
    for arm in ("exp", "grp"):
        t = R[arm][1:]
        p(f"   {arm}: decode tok/s {med([x['tok'] for x in t])}, first token s {med([x['ttft'] for x in t])}, "
          f"peak footprint GB {med([x['peak']/1e9 for x in t])}, max RSS GB {med([x['rss']/1e9 for x in t])}; round 0 {R[arm][0]['tok']:.3f} tok/s")
    rt = [g["tok"] / e["tok"] for e, g in zip(R["exp"][1:], R["grp"][1:])]
    rf = [g["ttft"] / e["ttft"] for e, g in zip(R["exp"][1:], R["grp"][1:])]
    dp = [(e["peak"] - g["peak"]) / 1e9 for e, g in zip(R["exp"][1:], R["grp"][1:])]
    p(f"   grp/exp by round: decode {[f'{x:.3f}' for x in rt]} median {med(rt)}; first token {[f'{x:.3f}' for x in rf]} median {med(rf)}; "
      f"peak exp-grp GB {[f'{x:.3f}' for x in dp]} median {med(dp)}")
