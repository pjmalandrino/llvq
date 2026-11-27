"""Pairing of prereg inkling-fenetre-ppl-2026-11-27, from this folder: python3 -I pair.py

The statistics of ops/jobs/inkling-cuda-pieces-ppl.sh (ratio = exp(mean paired difference of NLL) - 1, 95 % interval
of the mean paired difference, 1.96 standard errors), for the arms `full` (reference) and `narrow`. Per text, then
the single windows pooled (secondary) and the joined texts pooled (primary), each against +-0.12 %. Then the
position-wise look on the joined texts: mean difference by position modulo 128 (the score chunk) and modulo 512
(the piece), the largest |difference| and its position, how many positions differ at all; and each run's read time.
"""
import json, math, os


def load(n):
    p = f"{n}.nll.json"
    if not os.path.exists(p):
        return None
    j = json.load(open(p))
    return dict(zip(j["positions"], j["nll"]))


def stats(d):
    n = len(d)
    md = sum(d) / n
    se = math.sqrt(sum((x - md) ** 2 for x in d) / (n - 1)) / math.sqrt(n)
    return n, md, se


def line(tag, d, extra=""):
    n, md, se = stats(d)
    lo, hi = 100 * (math.exp(md - 1.96 * se) - 1), 100 * (math.exp(md + 1.96 * se) - 1)
    pt = 100 * (math.exp(md) - 1)
    print(f"{tag}: {n} positions{extra}, ratio {pt:+.4f} % [{lo:+.4f}, {hi:+.4f}]")
    return pt, lo, hi


def band(tag, r):
    pt, lo, hi = r
    print(f"{tag} band +-0.12 %: point {'inside' if abs(pt) <= 0.12 else 'OUTSIDE'}, "
          f"interval {'inside' if lo >= -0.12 and hi <= 0.12 else 'not inside'}")


singles = [f"w{w}" for w in range(6)]
joined = ["jw012", "jw345"]
diffs = {}
for t in singles + joined:
    a, b = load(f"{t}-full"), load(f"{t}-narrow")
    if a is None or b is None:
        print(f"{t}: a dump is missing")
        continue
    pos = sorted(set(a) & set(b))
    assert len(pos) == len(a) == len(b), t
    d = [b[p] - a[p] for p in pos]
    diffs[t] = (pos, d)
    pa = math.exp(sum(a[p] for p in pos) / len(pos))
    pb = math.exp(sum(b[p] for p in pos) / len(pos))
    nz = sum(1 for x in d if x != 0.0)
    k = max(range(len(d)), key=lambda i: abs(d[i]))
    line(t, d, f", ppl full {pa:.5f}, narrow {pb:.5f}; {nz} differ; largest |d| {abs(d[k]):.4f} at {pos[k]}")

print()
sec = [x for t in singles if t in diffs for x in diffs[t][1]]
pri = [x for t in joined if t in diffs for x in diffs[t][1]]
if sec:
    band("secondary (single windows pooled)", line("secondary (single windows pooled)", sec))
if pri:
    band("primary (joined texts pooled)", line("primary (joined texts pooled)", pri))

print()
for t in joined:
    if t not in diffs:
        continue
    pos, d = diffs[t]
    print(f"{t}: position-wise")
    for m in (128, 512):
        groups = {}
        for p, x in zip(pos, d):
            groups.setdefault(p % m, []).append(x)
        means = [(r, sum(v) / len(v)) for r, v in sorted(groups.items())]
        allm = sum(d) / len(d)
        top = sorted(means, key=lambda rm: -abs(rm[1]))[:5]
        edge = [rm for rm in means if rm[0] in (0, 1, m - 1)]
        sd = math.sqrt(sum((x - allm) ** 2 for x in d) / (len(d) - 1))
        print(f"  modulo {m}: {len(groups)} residues, {len(d) // m} positions each or so; overall mean {allm:+.5f}, "
              f"sd of one difference {sd:.4f}, so a residue mean's standard error about {sd / math.sqrt(len(d) / m):.4f}")
        print(f"    largest |mean| residues: " + ", ".join(f"{r}: {v:+.4f}" for r, v in top))
        print(f"    boundary residues (0, 1, {m - 1}): " + ", ".join(f"{r}: {v:+.4f}" for r, v in edge))
    k = max(range(len(d)), key=lambda i: abs(d[i]))
    print(f"  largest |d| {abs(d[k]):.4f} (d {d[k]:+.4f}) at position {pos[k]} (mod 128 = {pos[k] % 128}, "
          f"mod 512 = {pos[k] % 512}); {sum(1 for x in d if x != 0.0)} of {len(d)} positions differ; "
          f"{sum(1 for x in d if abs(x) > 1.0)} above 1.0")

print()
for t in singles + joined:
    for arm in ("full", "narrow"):
        p = f"{t}-{arm}.json"
        if os.path.exists(p):
            r = json.load(open(p))
            print(f"{t}-{arm}: read {r['seconds_prefill']:.2f} s, scored {r['scored']}, ppl {r['ppl']:.5f}, "
                  f"attn_window {r['provenance'].get('attn_window')}")
