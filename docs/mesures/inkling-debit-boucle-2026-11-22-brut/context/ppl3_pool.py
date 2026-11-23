"""ppl3_pool.py <runs-dir>: the whole stack (every switch) against no switch on three windows, paired by position,
each window alone and the three pooled."""
import sys, json, math
R = sys.argv[1].rstrip('/') + '/'
def load(names):
    d = {}
    for n in names:
        j = json.load(open(R + n + '.nll.json'))
        for p, v in zip(j['positions'], j['nll']):
            d.setdefault(p, v)
    return d
W = {"w0": (["ppl_ref_from1", "ppl_ref_from1024"], ["ppl_n1_from1", "ppl_n1_from1024"]),
     "w1": (["ppl_w1_ref_from1", "ppl_w1_ref_from1024"], ["ppl_w1_all_from1", "ppl_w1_all_from1024"]),
     "w2": (["ppl_w2_ref_from1", "ppl_w2_ref_from1024"], ["ppl_w2_all_from1", "ppl_w2_all_from1024"])}
def stats(diff):
    n = len(diff); md = sum(diff) / n
    sd = math.sqrt(sum((x - md) ** 2 for x in diff) / (n - 1)); se = sd / math.sqrt(n)
    return n, md, se
pooled = []
for w, (a, b) in W.items():
    A, B = load(a), load(b); pos = sorted(set(A) & set(B)); diff = [B[p] - A[p] for p in pos]
    n, md, se = stats(diff); pooled += diff
    pa = math.exp(sum(A[p] for p in pos) / n); pb = math.exp(sum(B[p] for p in pos) / n)
    print(f"{w}: {n} positions, ppl no switch {pa:.5f}, every switch {pb:.5f}, ratio {100*(math.exp(md)-1):+.4f} % [{100*(math.exp(md-1.96*se)-1):+.4f}, {100*(math.exp(md+1.96*se)-1):+.4f}]")
n, md, se = stats(pooled)
lo, hi = 100*(math.exp(md-1.96*se)-1), 100*(math.exp(md+1.96*se)-1)
print(f"three windows pooled: {n} positions, ratio {100*(math.exp(md)-1):+.4f} % [{lo:+.4f}, {hi:+.4f}] (95 %, paired positions)")
print(f"band +-0.12 %: point {'inside' if abs(100*(math.exp(md)-1)) <= 0.12 else 'OUTSIDE'}, interval {'inside' if lo >= -0.12 and hi <= 0.12 else 'not inside'}")
