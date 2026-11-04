"""Pair two `ppl` stderr logs window by window: mean dNLL (b minus a), paired t CI95 over the
windows (t_0.975,11 = 2.201 at 12 windows), as exp() - 1, windows worse, largest per-window dNLL."""
import math
import re
import sys

T = {11: 2.200985160082949}


def load(p):
    txt = open(p).read()
    w = [(int(m.group(1)), float(m.group(2)), int(m.group(3)))
         for m in re.finditer(r"window\s+(\d+)/\d+\s+nll\s+([0-9.]+)\s+n\s+(\d+)", txt)]
    fp = re.search(r"fingerprint ([0-9a-f]{16})", txt).group(1)
    dt = re.search(r"dtype (\w+)", txt).group(1)
    ppl = re.search(r"ppl = ([0-9.]+)", txt).group(1)
    return w, fp, dt, ppl


def pair(a_path, b_path, label):
    a, fa, da, pa = load(a_path)
    b, fb, db, pb = load(b_path)
    assert fa == fb, (fa, fb)
    assert [x[0] for x in a] == [x[0] for x in b] and all(x[2] == y[2] for x, y in zip(a, b))
    d = [y[1] - x[1] for x, y in zip(a, b)]
    n = len(d)
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
    half = T[n - 1] * sd / math.sqrt(n)
    lo, hi = mean - half, mean + half
    pct = lambda v: 100 * (math.exp(v) - 1)
    print(f"{label}")
    print(f"  a: {a_path.split('/')[-1]}  dtype {da}  ppl {pa}  fingerprint {fa}")
    print(f"  b: {b_path.split('/')[-1]}  dtype {db}  ppl {pb}  fingerprint {fb}")
    print("  window  nll a         nll b         b - a")
    for (i, x, _), (_, y, _), z in zip(a, b, d):
        print(f"  {i:>6}  {x:.9f}  {y:.9f}  {z:+.9f}")
    print(f"  paired mean dNLL, b minus a   {mean:+.7f} nats   (sd {sd:.7f}, n {n})")
    print(f"  CI95, paired t over {n} windows [{lo:+.7f}, {hi:+.7f}]")
    print(f"  as a perplexity ratio b/a - 1  {pct(mean):+.4f} % [{pct(lo):+.4f} %, {pct(hi):+.4f} %]")
    print(f"  windows worse under b          {sum(1 for z in d if z > 0)} of {n}")
    k = max(range(n), key=lambda i: abs(d[i]))
    print(f"  largest |dNLL|                 {d[k]:+.9f} at window {k + 1}")
    print(f"  token-weighted ppl a {math.exp(sum(x[1] * x[2] for x in a) / sum(x[2] for x in a)):.4f}, "
          f"b {math.exp(sum(x[1] * x[2] for x in b) / sum(x[2] for x in b)):.4f}")
    print()


if __name__ == "__main__":
    pair(sys.argv[1], sys.argv[2], sys.argv[3])
