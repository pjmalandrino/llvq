"""Pair two `bin/ppl`-format window lists: per-window relative change of the
perplexity, B against A, and the paired interval (mean +- 1.96 s / sqrt(n))."""
import math, re, sys

def windows(path):
    out = {}
    for line in open(path):
        m = re.search(r"window\s+(\d+)/\d+\s+nll\s+([0-9.]+)\s+n\s+(\d+)", line)
        if m:
            out[int(m.group(1))] = (float(m.group(2)), int(m.group(3)))
    return out

a, b = windows(sys.argv[1]), windows(sys.argv[2])
common = sorted(set(a) & set(b))
assert common, "no window in common"
d = []
print("window   A ppl        B ppl        B / A - 1")
for w in common:
    (na, ca), (nb, cb) = a[w], b[w]
    assert ca == cb, f"window {w}: {ca} against {cb} predictions"
    r = (math.exp(nb - na) - 1) * 100
    d.append(r)
    print(f"{w:>6}   {math.exp(na):<11.6f}  {math.exp(nb):<11.6f}  {r:+.4f} %")
n = len(d)
mean = sum(d) / n
s = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else float("nan")
h = 1.96 * s / math.sqrt(n)
pa = math.exp(sum(a[w][0] * a[w][1] for w in common) / sum(a[w][1] for w in common))
pb = math.exp(sum(b[w][0] * b[w][1] for w in common) / sum(b[w][1] for w in common))
print(f"pooled ppl A {pa:.4f} B {pb:.4f} ({(pb / pa - 1) * 100:+.4f} %)")
print(f"paired mean {mean:+.4f} % [{mean - h:+.4f} ; {mean + h:+.4f}] over {n} windows; max |window| {max(map(abs, d)):.4f} %")
