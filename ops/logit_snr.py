# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Logit-fidelity accounting on the MMLU dumps already on disk. Reads, computes, spends nothing.

Every published MMLU bar throws away 99% of what the run measured. `mmlu` writes the four
option logits verbatim for exactly this reason (`llvq-llm/src/bin/mmlu.rs:448`), and one
argmax per question is what the accuracy keeps. This script reads the logits back.

The model it fits is one line. For a question, write the four f16 logits centred on their own
mean as `x`, the arm's as `y`, and regress:

    y = beta * x + e,    SNR = beta * sd(x) / sd(e).

`beta` is how much of the reference signal survives quantization, `sd(e)` is what replaces it.
An argmax over `y` is invariant under a positive scale, so MMLU sees only their ratio; a
log-likelihood sees both. Nothing here is a new measurement: the dumps are the ones the paid
jobs produced, and every number this prints is *computed* on them.

Usage: `uv run ops/logit_snr.py` from the repository root. No argument, no network, no card.
"""

import math
import os
import random
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 20260912

# The 4B bank: one f16 reference, and every arm scored against it on the same 2,280 questions.
F16 = "docs/data/mmlu-dumps/mmlu-4b-f16.csv"
ARMS = {
    "awq4": "docs/data/mmlu-dumps/mmlu-4b-awq.csv",
    "planes14": "docs/data/mmlu-dumps/mmlu-4b-llvq.csv",
    "tetra": "docs/data/mmlu-dumps/mmlu-4b-tetra.csv",
    "tetra+q5": "docs/data/mmlu-dumps/mmlu-4b-tetra-q5.csv",
    "tetra+attn4": "docs/data/mmlu-dumps/mmlu-4b-tetra-attn4.csv",
    "seed1": "docs/data/bruit-mmlu-graines/mmlu-s1.csv",
    "seed2": "docs/data/bruit-mmlu-graines/mmlu-s2.csv",
    "seed3": "docs/data/bruit-mmlu-graines/mmlu-s3.csv",
}
TYPES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
for _t in TYPES + ["attn", "mlp"]:
    ARMS["M2:" + _t] = f"docs/data/m2-attribution/mmlu-restore-{_t}.csv"
    ARMS["M2rep:" + _t] = f"docs/data/m2rep-graine3/mmlu-restore-{_t}.csv"
ARMS["M2:v_int4"] = "docs/data/m2-attribution/mmlu-v_proj-int4g128.csv"
ARMS["M2b3:v_int4"] = "docs/data/m2b-graine3/mmlu-v4.csv"
ARMS["M2rep:shipped"] = "docs/data/m2rep-graine3/mmlu-shipped.csv"

# Qwen3-4B shapes, from the checkpoint config: 36 blocks, hidden 2560, 32 q heads and 8 kv
# heads of 128, intermediate 9728. Only the ratios between types are used.
SIZES = {
    "q_proj": 2560 * 32 * 128, "k_proj": 2560 * 8 * 128, "v_proj": 2560 * 8 * 128,
    "o_proj": 32 * 128 * 2560, "gate_proj": 2560 * 9728, "up_proj": 2560 * 9728,
    "down_proj": 9728 * 2560,
}


def load(path):
    """A dump keyed by (subject, index). Four qhashes collide in the bank, the pair does not."""
    rows = {}
    with open(os.path.join(ROOT, path)) as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("subject,") or not line.strip():
                continue
            p = line.strip().split(",")
            rows[(p[0], int(p[1]))] = dict(
                population=int(p[2]), answer=int(p[4]), pick=int(p[5]),
                correct=int(p[6]), logits=[float(x) for x in p[7:11]])
    return rows


def strata(rows, keys):
    """Population weights per subject. This estimator reproduces every published bar."""
    g = defaultdict(list)
    for k in keys:
        g[k[0]].append(k)
    pop = {s: rows[ks[0]]["population"] for s, ks in g.items()}
    tot = sum(pop.values())
    return {s: pop[s] / tot for s in pop}, g


def accuracy(rows, keys):
    w, g = strata(rows, keys)
    return 100.0 * sum(w[s] * sum(rows[k]["correct"] for k in ks) / len(ks) for s, ks in g.items())


def accuracy_se(rows, keys):
    w, g = strata(rows, keys)
    v = 0.0
    for s, ks in g.items():
        p = sum(rows[k]["correct"] for k in ks) / len(ks)
        v += w[s] ** 2 * p * (1 - p) / (len(ks) - 1)
    return 100.0 * math.sqrt(v)


def centred(rows, k):
    L = rows[k]["logits"]
    m = sum(L) / 4.0
    return [x - m for x in L]


def fidelity(f16, arm, keys):
    """(beta, sd of the residual, sd of the reference, SNR). Centring within a question
    removes a quarter of an iid variance, so a simulation needs sd * sqrt(4/3)."""
    X, Y = [], []
    for k in keys:
        X += centred(f16, k)
        Y += centred(arm, k)
    n = len(X)
    mx, my = sum(X) / n, sum(Y) / n
    vx = sum((x - mx) ** 2 for x in X) / n
    cov = sum((X[i] - mx) * (Y[i] - my) for i in range(n)) / n
    b = cov / vx
    res = [Y[i] - my - b * (X[i] - mx) for i in range(n)]
    sd = math.sqrt(sum(e * e for e in res) / n)
    return b, sd, math.sqrt(vx), b * math.sqrt(vx) / sd


def softmax(v, t=1.0):
    m = max(v)
    e = [math.exp((x - m) / t) for x in v]
    s = sum(e)
    return [x / s for x in e]


def nll4(rows, keys, t=1.0):
    w, g = strata(rows, keys)
    tot = 0.0
    for s, ks in g.items():
        a = 0.0
        for k in ks:
            a -= math.log(max(softmax(rows[k]["logits"], t)[rows[k]["answer"]], 1e-12))
        tot += w[s] * a / len(ks)
    return tot


def best_temperature(rows, keys):
    lo, hi = 0.1, 20.0
    for _ in range(70):
        a, b = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if nll4(rows, keys, a) < nll4(rows, keys, b):
            hi = b
        else:
            lo = a
    t = (lo + hi) / 2
    return t, nll4(rows, keys, t)


def shape_corr(f16, arm, k):
    a, b = centred(f16, k), centred(arm, k)
    ma, mb = sum(a) / 4, sum(b) / 4
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(4))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((x - mb) ** 2 for x in b))
    return num / (da * db) if da > 1e-9 and db > 1e-9 else 0.0


def simulate(f16, keys, beta, sd, draws, rng):
    """Accuracy of beta * f16 logits + iid gaussian noise. sd is in pre-centring units."""
    w, g = strata(f16, keys)
    out = []
    for _ in range(draws):
        hits = {s: 0 for s in g}
        for k in keys:
            v = [beta * x + rng.gauss(0, sd) for x in f16[k]["logits"]]
            if max(range(4), key=lambda i: v[i]) == f16[k]["answer"]:
                hits[k[0]] += 1
        out.append(100 * sum(w[s] * hits[s] / len(g[s]) for s in g))
    m = sum(out) / len(out)
    if len(out) < 2:
        return m, 0.0
    return m, math.sqrt(sum((x - m) ** 2 for x in out) / (len(out) - 1))


def replay(f16, arm, keys, beta, bands, draws, rng):
    """Give every question another question's residual vector, drawn inside its own band of
    f16 margin. bands=1 destroys the coupling between the error and the question entirely."""
    w, g = strata(f16, keys)
    err = {}
    b, _, _, _ = fidelity(f16, arm, keys)
    X, Y = [], []
    for k in keys:
        X += centred(f16, k)
        Y += centred(arm, k)
    my = sum(Y) / len(Y)
    mx = sum(X) / len(X)
    for k in keys:
        x, y = centred(f16, k), centred(arm, k)
        err[k] = [y[i] - my - b * (x[i] - mx) for i in range(4)]
    margin = {}
    for k in keys:
        L = sorted(f16[k]["logits"], reverse=True)
        margin[k] = L[0] - L[1]
    order = sorted(keys, key=lambda k: margin[k])
    pool = defaultdict(list)
    for i, k in enumerate(order):
        pool[i * bands // len(order)].append(k)
    out = []
    for _ in range(draws):
        mapping = {}
        for ks in pool.values():
            src = ks[:]
            rng.shuffle(src)
            for j, k in enumerate(ks):
                mapping[k] = src[j]
        hits = {s: 0 for s in g}
        for k in keys:
            x = centred(f16, k)
            e = err[mapping[k]]
            v = [beta * x[i] + e[i] for i in range(4)]
            if max(range(4), key=lambda i: v[i]) == f16[k]["answer"]:
                hits[k[0]] += 1
        out.append(100 * sum(w[s] * hits[s] / len(g[s]) for s in g))
    m = sum(out) / len(out)
    return m, math.sqrt(sum((x - m) ** 2 for x in out) / (len(out) - 1))


def main():
    rng = random.Random(SEED)
    f16 = load(F16)
    arms = {n: load(p) for n, p in ARMS.items()}
    keys = sorted(set(f16) & set.intersection(*[set(a) for a in arms.values()]))
    ROOT_SD = fidelity(f16, arms["planes14"], keys)[2]
    print(f"logit fidelity on the 4B MMLU dumps: {len(keys)} questions, "
          f"{4 * len(keys)} logits per arm, f16 sd(centred) = {ROOT_SD:.4f}")
    print(f"f16 accuracy {accuracy(f16, keys):.2f} +- {accuracy_se(f16, keys):.2f}\n")

    print("1. Fidelity and accuracy, every arm scored against the same f16 dump")
    print(f"{'arm':<16} {'beta':>6} {'sd_res':>7} {'SNR':>6} {'noise share':>12} "
          f"{'accuracy':>9} {'NLL4':>7}")
    table = []
    for n, a in arms.items():
        b, sd, sx, snr = fidelity(f16, a, keys)
        share = sd ** 2 / (b * b * sx * sx + sd ** 2)
        table.append((snr, accuracy(f16 if False else a, keys), n, b, sd))
        print(f"{n:<16} {b:6.3f} {sd:7.3f} {snr:6.3f} {100 * share:11.1f}% "
              f"{accuracy(a, keys):9.2f} {nll4(a, keys):7.4f}")

    print("\n2. One scalar accounts for the whole family. acc = a + c ln(SNR), fitted on the")
    print("   LLVQ-family arms only; awq4 is printed against it, not fitted into it.")
    fam = [r for r in table if not r[2].startswith("awq")]
    lx = [math.log(r[0]) for r in fam]
    ly = [r[1] for r in fam]
    n = len(lx)
    mlx, mly = sum(lx) / n, sum(ly) / n
    c = sum((lx[i] - mlx) * (ly[i] - mly) for i in range(n)) / sum((x - mlx) ** 2 for x in lx)
    a0 = mly - c * mlx
    print(f"   acc = {a0:.2f} + {c:.2f} ln(SNR)   over {n} arms")
    ss = 0.0
    for snr, ac, nm, b, sd in sorted(table):
        f = a0 + c * math.log(snr)
        if not nm.startswith("awq"):
            ss += (ac - f) ** 2
        print(f"   {nm:<16} SNR {snr:5.3f}  measured {ac:6.2f}  law {f:6.2f}  {ac - f:+6.2f}")
    print(f"   in-sample rms residual {math.sqrt(ss / n):.2f} pp, "
          f"per-arm sampling SE ~{accuracy_se(arms['planes14'], keys):.2f} pp")
    step = math.exp(1.0 / c)
    print(f"   the law's exchange rate: one MMLU point costs SNR x{step:.3f}, "
          f"that is noise-to-signal x{1 / step ** 2:.3f}")
    b0 = fidelity(f16, arms['tetra'], keys)[3]
    for target in (60.0, 65.0, 70.04):
        need = math.exp((target - a0) / c)
        print(f"   to read {target:.2f} the served Tetra arm needs SNR {need:.2f}, "
              f"x{need / b0:.2f} of today's, noise-to-signal /{(need / b0) ** 2:.1f}")

    print("\n3. Out of sample: SNR read on 28 subjects, accuracy on the other 29.")
    subs = sorted(set(k[0] for k in keys))
    res, spread = [], []
    for _ in range(12):
        sh = subs[:]
        rng.shuffle(sh)
        ka = [k for k in keys if k[0] in set(sh[:28])]
        kb = [k for k in keys if k[0] in set(sh[28:])]
        pts = [(fidelity(f16, a, ka)[3], accuracy(a, kb)) for nm, a in arms.items()
               if not nm.startswith("awq")]
        lx = [math.log(s) for s, _ in pts]
        ly = [y for _, y in pts]
        m = len(lx)
        mlx, mly = sum(lx) / m, sum(ly) / m
        cc = sum((lx[i] - mlx) * (ly[i] - mly) for i in range(m)) / sum((x - mlx) ** 2 for x in lx)
        aa = mly - cc * mlx
        r = [ly[i] - (aa + cc * lx[i]) for i in range(m)]
        res.append(math.sqrt(sum(x * x for x in r) / m))
        spread.append(math.sqrt(sum((y - mly) ** 2 for y in ly) / m))
    print(f"   rms residual {sum(res) / len(res):.2f} pp, spread to explain "
          f"{sum(spread) / len(spread):.2f} pp, sampling SE on 29 subjects "
          f"~{accuracy_se(arms['planes14'], [k for k in keys if k[0] in set(subs[:29])]):.2f} pp")

    print("\n4. The error is multiplicative: per-question rms residual by f16 margin decile.")
    margin = {}
    for k in keys:
        L = sorted(f16[k]["logits"], reverse=True)
        margin[k] = L[0] - L[1]
    order = sorted(keys, key=lambda k: margin[k])
    show = ["awq4", "planes14", "tetra", "seed1", "seed3"]
    per = {}
    for nm in show:
        b, sd, _, _ = fidelity(f16, arms[nm], keys)
        X = [v for k in keys for v in centred(f16, k)]
        Y = [v for k in keys for v in centred(arms[nm], k)]
        mx, my = sum(X) / len(X), sum(Y) / len(Y)
        per[nm] = {k: math.sqrt(sum((centred(arms[nm], k)[i] - my - b * (centred(f16, k)[i] - mx)) ** 2
                                    for i in range(4)) / 4) for k in keys}
    print(f"{'decile':>7} {'f16 margin':>12}" + "".join(f"{s:>12}" for s in show))
    N = len(order)
    for i in range(10):
        g = order[i * N // 10:(i + 1) * N // 10]
        line = f"{i + 1:>7} {margin[g[0]]:5.2f}-{margin[g[-1]]:5.2f}"
        for nm in show:
            line += f"{sum(per[nm][k] for k in g) / len(g):12.3f}"
        print(line)
    for nm in show:
        xs = [margin[k] for k in keys]
        ys = [per[nm][k] for k in keys]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        cv = sum((xs[i] - mx) * (ys[i] - my) for i in range(len(xs)))
        sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        sy = math.sqrt(sum((y - my) ** 2 for y in ys))
        print(f"   correlation with the f16 margin, {nm}: {cv / (sx * sy):+.3f}")

    print("\n5. What the amplitude of the error alone would cost, against what it does cost.")
    print(f"{'arm':<12} {'measured':>9} {'gaussian iid':>14} {'own residuals':>15} "
          f"{'banded replay':>15}")
    for nm in ["awq4", "planes14", "tetra", "tetra+q5", "seed1", "seed2", "seed3"]:
        b, sd, _, _ = fidelity(f16, arms[nm], keys)
        g, gs = simulate(f16, keys, b, sd * math.sqrt(4 / 3), 60, rng)
        fr, frs = replay(f16, arms[nm], keys, b, 1, 40, rng)
        bd, bds = replay(f16, arms[nm], keys, b, 20, 40, rng)
        print(f"{nm:<12} {accuracy(arms[nm], keys):9.2f} {g:9.2f}+-{gs:.2f} "
              f"{fr:10.2f}+-{frs:.2f} {bd:10.2f}+-{bds:.2f}")

    print("\n6. The loss is a flip on part of the bank, not a fade everywhere.")
    print("   Questions whose four-logit shape is reversed against f16 (rho_q < 0):")
    broken = {}
    for nm in ["awq4", "planes14", "tetra", "seed1", "seed2", "seed3"]:
        broken[nm] = set(k for k in keys if shape_corr(f16, arms[nm], k) < 0)
        acc_on = (100 * sum(arms[nm][k]["correct"] for k in broken[nm]) / len(broken[nm]))
        print(f"   {nm:<10} {len(broken[nm]):4d} ({100 * len(broken[nm]) / len(keys):4.1f}%), "
              f"arm accuracy on them {acc_on:4.1f}%, "
              f"f16 {100 * sum(f16[k]['correct'] for k in broken[nm]) / len(broken[nm]):4.1f}%")
    fam5 = ["planes14", "tetra", "seed1", "seed2", "seed3"]
    print("   overlap between independent encodings, observed against chance:")
    for i, x in enumerate(fam5):
        for y in fam5[i + 1:]:
            o = len(broken[x] & broken[y])
            e = len(broken[x]) * len(broken[y]) / len(keys)
            print(f"     {x:>9} & {y:<9} {o:4d} against {e:5.1f}, lift x{o / e:.2f}")
    union = set.union(*[broken[n] for n in fam5])
    print(f"   broken by at least one of the five: {len(union)} "
          f"({100 * len(union) / len(keys):.1f}% of the bank)")

    print("\n7. Restoring any projection type repairs the same questions.")
    for tag, base, d in (("M2", "docs/data/m2-attribution/mmlu-shipped.csv",
                          "docs/data/m2-attribution/"),
                         ("M2rep", "docs/data/m2rep-graine3/mmlu-shipped.csv",
                          "docs/data/m2rep-graine3/")):
        S = load(base)
        R = {t: load(d + f"mmlu-restore-{t}.csv") for t in TYPES + ["attn", "mlp"]}
        ks = sorted(set(f16) & set(S) & set.intersection(*[set(v) for v in R.values()]))
        b0, sd0, sx0, snr0 = fidelity(f16, S, ks)
        ns0 = (sd0 / (b0 * sx0)) ** 2
        print(f"   {tag}: base SNR {snr0:.3f}, noise-to-signal {ns0:.4f}, "
              f"accuracy {accuracy(S, ks):.2f}")
        print(f"   {'type':<11} {'accuracy':>9} {'N/S':>7} {'N/S removed':>12} "
              f"{'share':>7} {'weights':>8} {'noise per weight':>17}")
        tot = sum(SIZES.values())
        rem = {}
        for t in TYPES:
            b, sd, sx, _ = fidelity(f16, R[t], ks)
            ns = (sd / (b * sx)) ** 2
            rem[t] = ns0 - ns
            print(f"   {t:<11} {accuracy(R[t], ks):9.2f} {ns:7.4f} {rem[t]:12.4f} "
                  f"{100 * rem[t] / ns0:6.1f}% {100 * SIZES[t] / tot:7.1f}% "
                  f"{rem[t] / (SIZES[t] / tot):17.3f}")
        for gname, mem in (("attn", TYPES[:4]), ("mlp", TYPES[4:])):
            b, sd, sx, _ = fidelity(f16, R[gname], ks)
            j = ns0 - (sd / (b * sx)) ** 2
            p = sum(rem[m] for m in mem)
            print(f"     {gname}: measured {j:.4f} ({100 * j / ns0:.1f}%), "
                  f"sum of members {p:.4f}, sub-additivity {j / p:.2f}")
        rep = {t: set(k for k in ks if R[t][k]["correct"] and not S[k]["correct"]) for t in TYPES}
        ceiling = set(k for k in ks if f16[k]["correct"] and not S[k]["correct"])
        u = set.union(*rep.values())
        pairs = [(len(rep[x] & rep[y]) / len(rep[x] | rep[y]),
                  len(rep[x]) * len(rep[y]) / len(ks) /
                  (len(rep[x]) + len(rep[y]) - len(rep[x]) * len(rep[y]) / len(ks)))
                 for i, x in enumerate(TYPES) for y in TYPES[i + 1:]]
        print(f"     Jaccard between the sets each single type repairs: "
              f"{100 * min(p[0] for p in pairs):.1f} to {100 * max(p[0] for p in pairs):.1f}%, "
              f"chance {100 * min(p[1] for p in pairs):.1f} to "
              f"{100 * max(p[1] for p in pairs):.1f}%")
        print(f"     union of the seven repaired sets {len(u)}, sum of their sizes "
              f"{sum(len(v) for v in rep.values())}; f16 itself repairs {len(ceiling)}, "
              f"of which the union covers {100 * len(u & ceiling) / len(ceiling):.0f}%")

    print("\n8. An output temperature buys nothing back.")
    t16, n16 = best_temperature(f16, keys)
    print(f"   f16 NLL4 {nll4(f16, keys):.4f}, best temperature {t16:.2f} -> {n16:.4f}")
    print(f"   {'arm':<12} {'NLL4':>7} {'excess':>8} {'T*':>5} {'excess at T*':>13} {'recovered':>10}")
    for nm in ["awq4", "planes14", "tetra", "tetra+q5", "seed1", "seed3"]:
        n0 = nll4(arms[nm], keys)
        t, nt = best_temperature(arms[nm], keys)
        e0, e1 = n0 - nll4(f16, keys), nt - n16
        print(f"   {nm:<12} {n0:7.4f} {e0:+8.4f} {t:5.2f} {e1:+13.4f} "
              f"{100 * (1 - e1 / e0):9.1f}%")

    print("\n9. Attenuation costs no MMLU point. Only the ratio does.")
    print(f"   {'arm':<10} {'beta':>6} {'sd':>6} {'attenuation only':>18} {'noise only':>12} "
          f"{'both':>8} {'real':>8}")
    for nm in ["awq4", "planes14", "tetra"]:
        b, sd, _, _ = fidelity(f16, arms[nm], keys)
        sdi = sd * math.sqrt(4 / 3)
        a1, _ = simulate(f16, keys, b, 0.0, 1, rng)
        a2, _ = simulate(f16, keys, 1.0, sdi / b, 30, rng)
        a3, _ = simulate(f16, keys, b, sdi, 30, rng)
        print(f"   {nm:<10} {b:6.3f} {sdi:6.3f} {a1:18.2f} {a2:12.2f} {a3:8.2f} "
              f"{accuracy(arms[nm], keys):8.2f}")

    print("\n10. Across sizes, the same accounting.")
    for size in ("4b", "8b", "14b"):
        g16 = load(f"docs/data/mmlu-dumps/mmlu-{size}-f16.csv")
        for arm in ("awq", "llvq"):
            a = load(f"docs/data/mmlu-dumps/mmlu-{size}-{arm}.csv")
            ks = sorted(set(g16) & set(a))
            b, sd, sx, snr = fidelity(g16, a, ks)
            print(f"   {size.upper():<4} {arm:<5} beta {b:.3f} sd_res {sd:.3f} SNR {snr:5.3f} "
                  f"accuracy {accuracy(a, ks):5.2f} against f16 {accuracy(g16, ks):5.2f}")


# ---------------------------------------------------------------------------
# Self-test. Rule 10: mutate the code before calling a gate green.
# ---------------------------------------------------------------------------

def _synth(f16, keys, beta, sd, rng, where="all"):
    """A synthetic arm: beta * f16 logits + gaussian noise. `where` says which
    options the noise lands on, at constant total variance."""
    out = {}
    for k in keys:
        L = f16[k]["logits"]
        order = sorted(range(4), key=lambda i: -L[i])
        if where == "all":
            w = [1.0] * 4
        elif where == "tail":
            w = [0.0] * 4
            for i in order[2:]:
                w[i] = math.sqrt(2.0)          # same sum of squares over the four
        else:
            raise ValueError(where)
        v = [beta * L[i] + w[i] * rng.gauss(0, sd) for i in range(4)]
        out[k] = dict(population=f16[k]["population"], answer=f16[k]["answer"],
                      pick=max(range(4), key=lambda i: v[i]), logits=v)
        out[k]["correct"] = int(out[k]["pick"] == out[k]["answer"])
    return out


def _fit_law(f16, arms, keys_snr, keys_acc):
    pts = [(fidelity(f16, a, keys_snr)[3], accuracy(a, keys_acc)) for a in arms]
    lx = [math.log(s) for s, _ in pts]
    ly = [y for _, y in pts]
    n = len(lx)
    mlx, mly = sum(lx) / n, sum(ly) / n
    c = sum((lx[i] - mlx) * (ly[i] - mly) for i in range(n)) / sum((x - mlx) ** 2 for x in lx)
    a = mly - c * mlx
    r = [ly[i] - (a + c * lx[i]) for i in range(n)]
    return a, c, math.sqrt(sum(x * x for x in r) / n), r


def selftest():
    rng = random.Random(SEED)
    f16 = load(F16)
    keys = sorted(f16)
    ok = True

    print("S1. Oracle: build an arm with a known beta and sigma, ask the estimator for them.")
    print(f"   {'beta in':>8} {'sigma in':>9} {'beta out':>9} {'sigma out':>10} "
          f"{'expected sigma out':>19} {'verdict':>9}")
    for beta, sd in ((0.466, 1.638), (0.960, 0.988), (1.000, 0.100), (0.250, 3.000)):
        a = _synth(f16, keys, beta, sd, rng)
        b, sdc, _, _ = fidelity(f16, a, keys)
        want = sd * math.sqrt(3.0 / 4.0)          # centring removes a quarter of the variance
        good = abs(b - beta) < 0.03 * beta and abs(sdc - want) < 0.03 * want
        ok &= good
        print(f"   {beta:8.3f} {sd:9.3f} {b:9.3f} {sdc:10.3f} {want:19.3f} "
              f"{'passes' if good else 'FAILS':>9}")

    print("\n   Round trip on the accuracy: an arm built at (beta, sd), then the simulator")
    print("   fed the pair the estimator reads back. One arm is one draw, so the bar is its")
    print("   own spread, not zero.")
    for beta, sd in ((0.466, 1.638), (0.960, 0.988)):
        draws = [_synth(f16, keys, beta, sd, rng) for _ in range(12)]
        got = [accuracy(a, keys) for a in draws]
        mu = sum(got) / len(got)
        spread = math.sqrt(sum((x - mu) ** 2 for x in got) / (len(got) - 1))
        b, sdc, _, _ = fidelity(f16, draws[0], keys)
        sim, _ = simulate(f16, keys, b, sdc * math.sqrt(4 / 3), 40, rng)
        good = abs(sim - mu) < 1.0
        ok &= good
        print(f"   beta {beta:.3f} sd {sd:.3f}: twelve arms average {mu:.2f} "
              f"(spread {spread:.2f}), simulator says {sim:.2f}  "
              f"{'passes' if good else 'FAILS':>9}")

    print("\nS2. Mutants, on the real published dump. Each must move a headline number.")
    arm = load(ARMS["planes14"])
    ks = sorted(set(f16) & set(arm))
    b_ref, sd_ref, sx, snr_ref = fidelity(f16, arm, ks)
    n = 4 * len(ks)

    def _fit(xs, ys):
        mx, my = sum(xs) / n, sum(ys) / n
        vx = sum((x - mx) ** 2 for x in xs) / n
        b = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / n / vx
        res = [ys[i] - my - b * (xs[i] - mx) for i in range(n)]
        return b, math.sqrt(sum(e * e for e in res) / n), math.sqrt(vx)

    raw = _fit([v for k in ks for v in f16[k]["logits"]],
               [v for k in ks for v in arm[k]["logits"]])
    snr_raw = raw[0] * raw[2] / raw[1]
    killed = abs(snr_raw - snr_ref) > 0.05 * snr_ref
    ok &= killed
    print(f"   {'no centring within a question':<34} SNR {snr_raw:6.3f} against "
          f"{snr_ref:.3f}  {'killed' if killed else 'SURVIVES':>9}")
    back = _fit([v for k in ks for v in centred(arm, k)],
                [v for k in ks for v in centred(f16, k)])
    killed = abs(back[0] - b_ref) > 0.05 * b_ref
    ok &= killed
    print(f"   {'regression run backwards':<34} beta {back[0]:6.3f} against "
          f"{b_ref:.3f}  {'killed' if killed else 'SURVIVES':>9}")
    m_bad, _ = simulate(f16, ks, b_ref, sd_ref, 60, rng)
    m_ok, _ = simulate(f16, ks, b_ref, sd_ref * math.sqrt(4 / 3), 60, rng)
    killed = abs(m_bad - m_ok) > 1.0
    ok &= killed
    print(f"   {'simulation without sqrt(4/3)':<34} {m_bad:6.2f} against {m_ok:.2f}  "
          f"{'killed' if killed else 'SURVIVES':>9}")
    shuffled = dict(arm)
    perm = ks[:]
    rng.shuffle(perm)
    swapped = {k: arm[perm[i]] for i, k in enumerate(ks)}
    b_s, sd_s, _, snr_s = fidelity(f16, swapped, ks)
    killed = snr_s < 0.3 * snr_ref
    ok &= killed
    print(f"   {'arm paired to the wrong questions':<34} SNR {snr_s:6.3f} against "
          f"{snr_ref:.3f}  {'killed' if killed else 'SURVIVES':>9}")

    print("\nS3. Power. Twelve synthetic arms spanning the measured SNR range; in the second")
    print("   world half of them get their noise only on the options f16 ranks third and")
    print("   fourth, at the same total variance. Those arms are better than their SNR says.")
    snrs = [1.03 + i * (2.43 - 1.03) / 11 for i in range(12)]
    subs = sorted(set(k[0] for k in keys))
    rng2 = random.Random(SEED + 1)
    probe = _synth(f16, keys, 0.5, 1.0, rng2)
    sx_ref = fidelity(f16, probe, keys)[2]
    detected = None
    for label, mix in (("no second axis", 0), ("half the arms tilted", 1)):
        built, tilt = [], []
        for i, s in enumerate(snrs):
            sd = 0.5 * sx_ref / s
            where = "tail" if (mix and i % 2) else "all"
            built.append(_synth(f16, keys, 0.5, sd, rng2, where))
            tilt.append(where == "tail")
        sh = subs[:]
        rng2.shuffle(sh)
        ka = [k for k in keys if k[0] in set(sh[:28])]
        kb = [k for k in keys if k[0] in set(sh[28:])]
        _, _, rms, res = _fit_law(f16, built, ka, kb)
        if mix:
            detected = (sum(res[i] for i in range(12) if tilt[i]) / sum(tilt)
                        - sum(res[i] for i in range(12) if not tilt[i]) / (12 - sum(tilt)))
            print(f"   {label:<22} rms {rms:5.2f} pp, tilted arms {detected:+.2f} pp "
                  f"above the rest")
        else:
            print(f"   {label:<22} rms {rms:5.2f} pp")
    print("   The rms is a weak detector of a second axis; the group contrast is the one to use.")

    print("\nS4. Group contrast on the real arms, with the detector S3 qualified.")
    real = {nm: load(p) for nm, p in ARMS.items()}
    kr = sorted(set(f16) & set.intersection(*[set(a) for a in real.values()]))
    names = [nm for nm in real if not nm.startswith("awq")]
    g1 = [nm for nm in names if nm.startswith("M2:") or nm == "planes14"]
    g2 = [nm for nm in names if nm.startswith("M2rep") or nm.startswith("M2b3") or nm == "seed3"]

    def _contrast(kk):
        pts = [(math.log(fidelity(f16, real[nm], kk)[3]), accuracy(real[nm], kk), nm)
               for nm in names]
        N = len(pts)
        mlx = sum(p[0] for p in pts) / N
        mly = sum(p[1] for p in pts) / N
        c = (sum((p[0] - mlx) * (p[1] - mly) for p in pts)
             / sum((p[0] - mlx) ** 2 for p in pts))
        a0 = mly - c * mlx
        r = {p[2]: p[1] - (a0 + c * p[0]) for p in pts}
        return (sum(r[nm] for nm in g1) / len(g1) - sum(r[nm] for nm in g2) / len(g2)), r

    obs, r0 = _contrast(kr)
    boot = []
    for _ in range(300):
        boot.append(_contrast([kr[rng.randrange(len(kr))] for _ in range(len(kr))])[0])
    boot.sort()
    print(f"   published-file arms minus seed-3-file arms: {obs:+.2f} pp, "
          f"bootstrap CI95 [{boot[7]:+.2f}; {boot[292]:+.2f}]")
    print(f"   the law therefore excludes a second axis larger than about "
          f"{max(abs(boot[7]), abs(boot[292])):.1f} pp, and no smaller one is resolved.")

    print("\nS5. Bootstrap over questions, 400 resamples, on the headline quantities.")
    for nm in ("awq4", "planes14", "tetra"):
        a = real[nm]
        bs, ns, fl = [], [], []
        for _ in range(400):
            samp = [kr[rng.randrange(len(kr))] for _ in range(len(kr))]
            b, sd, sxx, _ = fidelity(f16, a, samp)
            bs.append(b)
            ns.append(100 * sd ** 2 / (b * b * sxx * sxx + sd ** 2))
            fl.append(100 * sum(1 for k in samp if shape_corr(f16, a, k) < 0) / len(samp))

        def ci(v):
            v = sorted(v)
            return v[int(0.025 * len(v))], v[int(0.975 * len(v))]

        b, sd, sxx, _ = fidelity(f16, a, kr)
        print(f"   {nm:<10} beta {b:.3f} [{ci(bs)[0]:.3f}; {ci(bs)[1]:.3f}]   noise share "
              f"{100 * sd ** 2 / (b * b * sxx * sxx + sd ** 2):.1f}% "
              f"[{ci(ns)[0]:.1f}; {ci(ns)[1]:.1f}]   reversed "
              f"{100 * sum(1 for k in kr if shape_corr(f16, a, k) < 0) / len(kr):.1f}% "
              f"[{ci(fl)[0]:.1f}; {ci(fl)[1]:.1f}]")

    print("\nS6. The reversal threshold is a choice. The share at other cuts:")
    print(f"   {'cut on rho_q':>13}" + "".join(f"{n:>12}" for n in ("awq4", "planes14", "tetra")))
    for cut in (-0.5, -0.25, 0.0, 0.25, 0.5):
        line = f"   {cut:>13.2f}"
        for nm in ("awq4", "planes14", "tetra"):
            line += f"{100 * sum(1 for k in kr if shape_corr(f16, real[nm], k) < cut) / len(kr):11.1f}%"
        print(line)

    print("\n" + ("self-test passes" if ok else "SELF-TEST FAILS"))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(selftest() if "selftest" in sys.argv[1:] else main())
