# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Direction-density proxy for sectioned Leech shape-gain codebooks.

The served code is shape-gain: the reconstruction is gain * y / |y|, so every
lattice point of the region is a direction. Around a unit direction u a region
of radial extent rho(u) holds rho(u)^24 / (24 V0) points per unit solid angle.
High-resolution angular distortion of a block x is then

    D_shape(x)  ~  |x|^2 * n(u)^(-2/23)

with n(u) the direction density. The members of a union cut one lattice, so a
point two members hold is indexed twice and n(u) is the MAX over the members,
not their sum (proxy_dir_sum.py is the rejected sum variant). V0 is the same
for every candidate at 47 bits. The gain part is common to all candidates.

Calibration targets, shape error per dim, from innovation-tetra-v2-forme:
  Tetra 0.07652 ; ball about 0.0695 (0.0780 total minus the same 0.0085 gain) ;
  joint budget 0.0716 ; selector k: total -0.183 dB.
"""
import itertools
import math
import time

import numpy as np

T0 = time.time()
V8 = math.pi**4 / 24
V24 = math.pi**12 / math.factorial(12)
MID = 2.0 ** (-1 / 8)
BASE = np.array([1.0, MID, 1.0])
BASEVOL = V8**3 * float(np.prod(BASE**8))
EXP = -2.0 / 23.0
SHAPE, GAIN = 0.07652, 0.00854  # measured split of Tetra's 0.08506

PAIRS = list(itertools.combinations(range(6), 2))
PIDX = {p: i for i, p in enumerate(PAIRS)}


def pairings():
    out = []

    def rec(rest, cur):
        if not rest:
            out.append(tuple(cur))
            return
        a = rest[0]
        for b in rest[1:]:
            rec([r for r in rest if r not in (a, b)], cur + [(a, b)])

    rec(list(range(6)), [])
    return out


PAIRINGS = pairings()
IDENT = ((0, 1), (2, 3), (4, 5))
ORDERS = list(itertools.permutations(range(3)))


def sample(m, rng):
    x = rng.standard_normal((m, 24))
    t = (x.reshape(m, 6, 4) ** 2).sum(-1)
    w = t.sum(1)
    inv = np.empty((m, 15))  # 1 / |u_section| for each tetrad pair
    for i, (a, b) in enumerate(PAIRS):
        inv[:, i] = np.sqrt(w / (t[:, a] + t[:, b]))
    return w, inv


def member(pairing, widths):
    idx = [PIDX[tuple(sorted(p))] for p in pairing]
    radii = BASE * 2.0 ** ((np.array(widths) - 11) / 8)
    return idx, radii, 33 - sum(widths), (pairing, tuple(widths))


def dens(inv, mem):
    """rho(u)^24 / BASEVOL: direction density of one member, up to N / 24."""
    rho = (inv[:, mem[0]] * mem[1]).min(1)
    return rho**24 / BASEVOL


def shape(w, inv, mems):
    d = np.zeros(len(w))
    for mem in mems:
        d = np.maximum(d, dens(inv, mem))
    return w * d**EXP


def greedy(w, inv, cands, tag):
    c = len(cands)
    dj = np.empty((len(w), c), dtype=np.float32)
    for k, mem in enumerate(cands):
        dj[:, k] = dens(inv, mem)
    kraft = np.array([2.0 ** (-mem[2]) for mem in cands])
    w32 = w.astype(np.float32)[:, None]
    single = (w32 * dj**EXP).mean(0)
    starts = [int(k) for k in np.argsort(np.where(kraft >= 0.25, single, np.inf))[:3]]
    best = None
    for first in starts:
        chosen, cur, left = [first], dj[:, first].copy(), 1.0 - kraft[first]
        while left > 1e-9:
            base = float((w32[:, 0] * cur**EXP).mean())
            obj = (w32 * np.maximum(cur[:, None], dj) ** EXP).mean(0)
            ratio = np.where(kraft <= left + 1e-12, (base - obj) / kraft, -1.0)
            ratio[chosen] = -1.0
            k = int(ratio.argmax())
            if ratio[k] <= 0:
                break
            chosen.append(k)
            cur = np.maximum(cur, dj[:, k])
            left -= kraft[k]
        val = float((w32[:, 0] * cur**EXP).mean())
        if best is None or val < best[0]:
            best = (val, chosen, left)
    print(f"  [{tag}] {c} candidates, kept {len(best[1])}, Kraft left {best[2]:.4f}", flush=True)
    return [cands[k] for k in best[1]]


def report(name, per, ref):
    r = per.mean() / ref.mean()
    d = per / per.mean() - ref / ref.mean()
    se = d.std(ddof=1) / math.sqrt(len(d))
    tot = 10 * math.log10((SHAPE * r + GAIN) / (SHAPE + GAIN))
    print(f"{name:74s} {r:6.3f}  {tot:+7.3f}  {10 / math.log(10) * se * SHAPE / (SHAPE + GAIN):.3f}")


def triples(lo, hi, lmax):
    return [w for w in itertools.product(range(lo, hi + 1), repeat=3) if 1 <= 33 - sum(w) <= lmax]


def main():
    rng = np.random.default_rng(20261126)
    wd, invd = sample(6000, rng)
    we, inve = sample(400_000, rng)

    ref = shape(we, inve, [member(IDENT, (11, 11, 11))])
    print(f"{'candidate':74s} {'shape':>6s}  {'tot dB':>7s}  SE")
    report("Tetra-like product, radii (1, 2^-1/8, 1)", ref, ref)

    ball = we * (1.0 / V24) ** EXP
    report("ball   [calibration: shape 0.908, total -0.375 measured]", ball, ref)

    sym = ([0, 9, 14], np.full(3, 2.0 ** (-1 / 24)), 0, None)
    assert [PAIRS[i] for i in sym[0]] == [(0, 1), (2, 3), (4, 5)]
    report("symmetric product of three 8-balls", shape(we, inve, [sym]), ref)

    kprof = [member(IDENT, w) for w in [(13, 9, 9), (9, 13, 9), (9, 9, 13), (10, 11, 10)]]
    report("k, 4 profiles   [calibration: shape 0.954, total -0.183 measured]", shape(we, inve, kprof), ref)

    for hi in (12, 13):
        mems = greedy(wd, invd, [member(IDENT, w) for w in triples(7, hi, 6)], f"lead 1, w<={hi}")
        report(f"lead 1: prefix profiles, widths <= {hi} ({len(mems)} profiles)", shape(we, inve, mems), ref)
        if hi == 13:
            for mem in sorted(mems, key=lambda q: q[2]):
                print(f"      l={mem[2]} widths {mem[3][1]}")

    cands = [member(tuple(pg[i] for i in od), (10, 10, 10)) for pg in PAIRINGS for od in ORDERS[:3]]
    mems = greedy(wd, invd, cands, "lead 2 pure")
    report(f"lead 2 pure: {len(mems)} pairings at (10,10,10), table halves", shape(we, inve, mems), ref)

    for hi in (11,):
        cands = [member(tuple(pg[i] for i in od), w) for pg in PAIRINGS for od in ORDERS for w in triples(8, hi, 5)]
        mems = greedy(wd, invd, cands, f"lead 2, w<={hi}")
        npair = len({tuple(sorted(tuple(sorted(q)) for q in mem[3][0])) for mem in mems})
        report(f"lead 2: pairings, widths <= {hi}, table unchanged ({len(mems)} members, {npair} pairings)",
               shape(we, inve, mems), ref)

    cands = [member(tuple(pg[i] for i in od), w) for pg in PAIRINGS for od in ORDERS for w in triples(8, 13, 5)]
    mems = greedy(wd, invd, cands, "leads 1+2")
    report(f"leads 1+2: pairings and prefix profiles, widths <= 13 ({len(mems)} members)", shape(we, inve, mems), ref)

    # Concentration of Tetra's excess shape error over the ball.
    ex = np.sort(ref - ball)[::-1]
    tot = ex.sum()
    print()
    for q in (0.01, 0.05, 0.10, 0.25):
        print(f"top {q:4.0%} of blocks carry {ex[: int(q * len(ex))].sum() / tot:6.1%} of Tetra's excess over the ball")
    print(f"blocks where Tetra is finer than the ball: {(ref < ball).mean():.1%}")

    lead3(rng)
    print(f"total {time.time() - T0:.1f} s")


def lead3(rng):
    """Two blocks, six sections, one prefix-coded profile. Conservative: the
    encoder takes the best single profile, densities are not summed."""
    trip = list(itertools.product(range(9, 14), repeat=3))
    sums = np.array([sum(w) for w in trip])
    rad = np.array([BASE * 2.0 ** ((np.array(w) - 11) / 8) for w in trip])
    lmat = 66 - (sums[:, None] + sums[None, :])
    ok = (lmat >= 1) & (lmat <= 6)
    kraft = np.where(ok, 2.0 ** (-lmat.astype(float)), np.inf)
    i0 = trip.index((11, 11, 11))

    def table(m):
        x = rng.standard_normal((m, 24))
        e = (x.reshape(m, 3, 8) ** 2).sum(-1)
        w = e.sum(1)
        inv = np.sqrt(w[:, None] / e)
        rho = (inv[:, None, :] * rad[None, :, :]).min(2)
        return (w[:, None] * (rho**24 / BASEVOL) ** EXP).astype(np.float32)

    a, b = table(6000), table(6000)
    chosen, left, cur = [], 1.0, None
    while left > 1e-9 and len(chosen) < 64:
        best = (-np.inf, None)
        base = float(cur.mean()) if chosen else 0.0
        for ja in range(125):
            cost = a[:, ja][:, None] + b
            if chosen:
                gain = base - np.minimum(cur[:, None], cost).mean(0)
                ratio = np.where(kraft[ja] <= left + 1e-12, gain / kraft[ja], -np.inf)
            else:
                ratio = np.where(ok[ja] & (lmat[ja] <= 2), -cost.mean(0), -np.inf)
            jb = int(ratio.argmax())
            if ratio[jb] > best[0]:
                best = (float(ratio[jb]), (ja, jb))
        if best[1] is None or (chosen and best[0] <= 0):
            break
        ja, jb = best[1]
        chosen.append((ja, jb))
        c = a[:, ja] + b[:, jb]
        cur = c if cur is None else np.minimum(cur, c)
        left -= 2.0 ** (-float(lmat[ja, jb]))
    print(f"  [lead 3] kept {len(chosen)} six-section profiles, Kraft left {left:.4f}")
    ea, eb = table(200_000), table(200_000)
    pair = np.full(200_000, np.inf, dtype=np.float32)
    for ja, jb in chosen:
        pair = np.minimum(pair, ea[:, ja] + eb[:, jb])
    ref = (ea[:, i0] + eb[:, i0]).astype(np.float64)
    report(f"lead 3: two blocks, six sections, {len(chosen)} prefix profiles (conservative)", pair.astype(np.float64), ref)
    # Same conservative rule on one block, to read lead 3 against lead 1 like for like.
    one = np.full(200_000, np.inf, dtype=np.float32)
    for w in [(13, 9, 9), (9, 13, 9), (9, 9, 13), (10, 11, 10)]:
        one = np.minimum(one, ea[:, trip.index(w)])
    report("   k under the same conservative rule (best single profile)", one.astype(np.float64), ea[:, i0].astype(np.float64))


if __name__ == "__main__":
    main()
