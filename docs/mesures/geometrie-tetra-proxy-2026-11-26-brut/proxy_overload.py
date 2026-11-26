# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Energy-space proxy for the shaping loss of sectioned Leech codebooks.

High-resolution model, not the real code. A codebook is a shaping region S in
R^24 holding 2^47 lattice points, plus one gain bit (two scales, the encoder
takes the better). Per block:

    D(x) = min_g [ 24 * G * g^2 * V0^(2/24) + dist^2(x, g*S) ]

G is the normalized second moment of the Leech lattice, V0 = vol(S) / 2^47.
Regions are unions of products of three 8-balls (sections), or the 24-ball.
A section of index width w has radius base * 2^((w - 11) / 8).
The Tetra-like base has radii (1, 2^(-1/8), 1): the middle section indexes
2,048 rank vectors over both classes, the ends 2,048 per class.

The proxy is calibrated on two measured points (journal
innovation-tetra-v2-forme-2026-11-11): ball vs Tetra -0.375 dB, and the
4-profile selector k at -0.18 dB.
"""
import itertools
import math
import sys
import time

import numpy as np

T0 = time.time()
N = 24
G = 0.065771
V8 = math.pi**4 / 24
V24 = math.pi**12 / math.factorial(12)
P47 = 2.0**47
MID = 2.0 ** (-1 / 8)
BASE = np.array([1.0, MID, 1.0])
BASEVOL = V8**3 * float(np.prod(BASE**8))
V0_PROD = BASEVOL / P47
GRAN_PROD = N * G * V0_PROD ** (2 / N)  # times g^2

PAIRS = list(itertools.combinations(range(6), 2))  # 15 tetrad pairs
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
assert len(PAIRINGS) == 15
IDENT = ((0, 1), (2, 3), (4, 5))


def sample(m, rng):
    x = rng.standard_normal((m, 24))
    t = (x.reshape(m, 6, 4) ** 2).sum(-1)
    s = np.empty((m, 15))
    for i, (a, b) in enumerate(PAIRS):
        s[:, i] = np.sqrt(t[:, a] + t[:, b])
    return t, s


def member(pairing, widths):
    """(pair indices by section, radii by section, prefix length)."""
    idx = [PIDX[tuple(sorted(p))] for p in pairing]
    radii = BASE * 2.0 ** ((np.array(widths) - 11) / 8)
    return idx, radii, 33 - sum(widths), (pairing, tuple(widths))


def over(s, mem, g):
    idx, radii = mem[0], mem[1]
    d = np.maximum(s[:, idx] - g * radii, 0.0)
    return (d * d).sum(1)


def best_two_gains(dg, gains):
    """dg[k, m] = per-block cost at gain k. Best pair, encoder takes the min."""
    best = (np.inf, 0, 0)
    for i in range(len(gains)):
        mins = np.minimum(dg[i][None, :], dg[i:]).mean(1)
        k = int(mins.argmin())
        if mins[k] < best[0]:
            best = (float(mins[k]), i, i + k)
    return best


def eval_members(s, mems, gains):
    dg = np.empty((len(gains), s.shape[0]))
    for k, g in enumerate(gains):
        o = np.full(s.shape[0], np.inf)
        for mem in mems:
            o = np.minimum(o, over(s, mem, g))
        dg[k] = GRAN_PROD * g * g + o
    val, i, j = best_two_gains(dg, gains)
    per = np.minimum(dg[i], dg[j])
    return per, (gains[i], gains[j])


def eval_ball(t, gains):
    r = np.sqrt(t.sum(1))
    gran = N * G * (V24 / P47) ** (2 / N)
    dg = np.empty((len(gains), t.shape[0]))
    for k, g in enumerate(gains):
        dg[k] = gran * g * g + np.maximum(r - g, 0.0) ** 2
    val, i, j = best_two_gains(dg, gains)
    return np.minimum(dg[i], dg[j]), (gains[i], gains[j])


def greedy(s, cands, g2, tag):
    """Pick members under Kraft sum <= 1. cands: list of member tuples."""
    m = s.shape[0]
    c = len(cands)
    dj = np.empty((m, c), dtype=np.float32)
    for k, mem in enumerate(cands):
        a = GRAN_PROD * g2[0] ** 2 + over(s, mem, g2[0])
        b = GRAN_PROD * g2[1] ** 2 + over(s, mem, g2[1])
        dj[:, k] = np.minimum(a, b)
    kraft = np.array([2.0 ** (-mem[2]) for mem in cands])
    means = dj.mean(0)
    best_final = None
    for first in np.argsort(means)[:3]:
        chosen = [int(first)]
        cur = dj[:, first].copy()
        left = 1.0 - kraft[first]
        while left > 1e-9:
            gain = cur.mean() - np.minimum(cur[:, None], dj).mean(0)
            ratio = np.where(kraft <= left + 1e-12, gain / kraft, -1.0)
            ratio[chosen] = -1.0
            k = int(ratio.argmax())
            if ratio[k] <= 0:
                break
            chosen.append(k)
            cur = np.minimum(cur, dj[:, k])
            left -= kraft[k]
        if best_final is None or cur.mean() < best_final[0]:
            best_final = (float(cur.mean()), chosen, left)
    print(f"  [{tag}] greedy: {c} candidates, kept {len(best_final[1])}, Kraft left {best_final[2]:.4f}", flush=True)
    return [cands[k] for k in best_final[1]]


def db(per, ref):
    r = per.mean() / ref.mean()
    d = per / per.mean() - ref / ref.mean()
    se = d.std(ddof=1) / math.sqrt(len(d))
    return 10 * math.log10(r), 10 / math.log(10) * se


def main():
    rng = np.random.default_rng(20261126)
    td, sd = sample(8000, rng)  # design
    te, se = sample(200_000, rng)  # evaluation
    gp = np.geomspace(2.0, 7.0, 64)
    gb = np.geomspace(3.5, 9.0, 64)

    tetra = [member(IDENT, (11, 11, 11))]
    ref, g_ref = eval_members(se, tetra, gp)
    print(f"Tetra-like product (1, 2^-1/8, 1): MSE/dim {ref.mean() / N:.5f}  gains {g_ref[0]:.3f} {g_ref[1]:.3f}")
    print("  measured real Tetra 0.08506")

    rows = []

    ball, g_b = eval_ball(te, gb)
    print(f"ball: MSE/dim {ball.mean() / N:.5f}   measured real ball 0.0780")
    rows.append(("ball, 24 dims (calibration, measured -0.375)", ball))

    sym = [(m[0], np.full(3, 2.0 ** (-1 / 24)), 0, None) for m in tetra]
    per, _ = eval_members(se, sym, gp)
    rows.append(("symmetric product of three 8-balls", per))

    kprof = [member(IDENT, w) for w in [(13, 9, 9), (9, 13, 9), (9, 9, 13), (10, 11, 10)]]
    per, _ = eval_members(se, kprof, gp)
    rows.append(("k: 4 profiles, 2-bit selector (calibration, measured -0.18)", per))

    def triples(lo, hi, lmax):
        return [w for w in itertools.product(range(lo, hi + 1), repeat=3) if 1 <= 33 - sum(w) <= lmax]

    # Lead 1: prefix-coded profiles, fixed trio.
    for hi in (12, 13):
        cands = [member(IDENT, w) for w in triples(7, hi, 6)]
        mems = greedy(sd, cands, g_ref, f"lead 1, w<={hi}")
        per, _ = eval_members(se, mems, gp)
        rows.append((f"lead 1: prefix profiles, widths <= {hi} ({len(mems)} profiles)", per))
        if hi == 13:
            for mem in sorted(mems, key=lambda q: q[2]):
                print(f"      l={mem[2]} widths {mem[3][1]}")

    # Lead 2: tetrad pairings, table does not grow.
    orders = list(itertools.permutations(range(3)))
    for hi in (10, 11):
        cands = []
        for pg in PAIRINGS:
            for od in orders:
                p = tuple(pg[i] for i in od)
                for w in triples(8, hi, 5):
                    cands.append(member(p, w))
        mems = greedy(sd, cands, g_ref, f"lead 2, w<={hi}")
        per, _ = eval_members(se, mems, gp)
        npair = len({tuple(sorted(tuple(sorted(q)) for q in mem[3][0])) for mem in mems})
        rows.append((f"lead 2: tetrad pairings, widths <= {hi} ({len(mems)} members, {npair} pairings)", per))

    # Lead 2 pure: 8 pairings at (10,10,10), 3-bit selector.
    cands = []
    for pg in PAIRINGS:
        for od in orders[:3]:
            p = tuple(pg[i] for i in (od if True else od))
            cands.append(member(p, (10, 10, 10)))
    mems = greedy(sd, cands, g_ref, "lead 2 pure, 8 x (10,10,10)")
    per, _ = eval_members(se, mems, gp)
    rows.append((f"lead 2 pure: {len(mems)} pairings at (10,10,10), 3-bit selector", per))

    # Leads 1 + 2 together.
    cands = []
    for pg in PAIRINGS:
        for od in orders:
            p = tuple(pg[i] for i in od)
            for w in triples(8, 13, 5):
                cands.append(member(p, w))
    mems = greedy(sd, cands, g_ref, "leads 1+2, w<=13")
    per, _ = eval_members(se, mems, gp)
    rows.append((f"leads 1+2: pairings and prefix profiles, widths <= 13 ({len(mems)} members)", per))

    print()
    print(f"{'candidate':78s} {'dB vs Tetra-like':>17s}  paired SE")
    for name, per in rows:
        d, e = db(per, ref)
        print(f"{name:78s} {d:+17.3f}  {e:.3f}")

    # Concentration of the excess over the ball, per block.
    ex = np.sort(ref - ball)[::-1]
    tot = ex.sum()
    for q in (0.01, 0.05, 0.10, 0.25):
        k = int(q * len(ex))
        print(f"top {q:4.0%} of blocks carry {ex[:k].sum() / tot:6.1%} of the excess over the ball")
    print(f"blocks where the product is no worse than the ball: {(ref <= ball).mean():.1%}")

    # Lead 3: two blocks, six sections, one prefix-coded profile.
    lead3(rng, g_ref)
    print(f"total {time.time() - T0:.1f} s")


def lead3(rng, g2):
    wid = list(range(9, 14))
    trip = list(itertools.product(wid, repeat=3))  # 125
    sums = np.array([sum(w) for w in trip])
    rad = np.array([BASE * 2.0 ** ((np.array(w) - 11) / 8) for w in trip])  # 125 x 3

    def block_cost(m, gains):
        x = rng.standard_normal((m, 24))
        e = np.sqrt((x.reshape(m, 3, 8) ** 2).sum(-1))  # m x 3
        out = np.full((m, 125), np.inf, dtype=np.float32)
        for g in gains:
            d = np.maximum(e[:, None, :] - g * rad[None, :, :], 0.0)
            out = np.minimum(out, (GRAN_PROD * g * g + (d * d).sum(-1)).astype(np.float32))
        return out

    md = 6000
    a, b = block_cost(md, g2), block_cost(md, g2)
    lmat = 66 - (sums[:, None] + sums[None, :])  # 125 x 125
    ok = (lmat >= 1) & (lmat <= 6)
    kraft = np.where(ok, 2.0 ** (-lmat.astype(float)), np.inf)
    i0 = trip.index((11, 11, 11))
    chosen, left = [], 1.0
    cur = np.full(md, np.inf, dtype=np.float32)
    # start from the two-block base split in halves is not allowed (l = 0), so seed by absolute gain
    while left > 1e-9:
        best = (-1.0, None)
        base = float(cur.mean()) if chosen else None
        for ja in range(125):
            cost = a[:, ja][:, None] + b  # md x 125
            if chosen:
                gain = base - np.minimum(cur[:, None], cost).mean(0)
                ratio = np.where(kraft[ja] <= left + 1e-12, gain / kraft[ja], -1.0)
            else:
                ratio = np.where(ok[ja] & (lmat[ja] <= 2), -cost.mean(0), -np.inf)
            for jb in np.argsort(ratio)[::-1][:1]:
                if (ja, int(jb)) not in chosen and ratio[jb] > best[0] or (not chosen and best[1] is None):
                    if ratio[jb] > best[0] or best[1] is None:
                        best = (float(ratio[jb]), (ja, int(jb)))
        if best[1] is None or (chosen and best[0] <= 0):
            break
        ja, jb = best[1]
        chosen.append((ja, jb))
        cur = np.minimum(cur, a[:, ja] + b[:, jb])
        left -= 2.0 ** (-float(lmat[ja, jb]))
        if len(chosen) >= 64:
            break
    print(f"  [lead 3] kept {len(chosen)} six-section profiles, Kraft left {left:.4f}", flush=True)

    me = 100_000
    best = None
    for f1 in (0.94, 1.0, 1.06):
        for f2 in (0.94, 1.0, 1.06):
            gains = (g2[0] * f1, g2[1] * f2)
            st = np.random.get_state()
            r2 = np.random.default_rng(777)
            nonlocal_rng = r2

            def bc(m):
                x = nonlocal_rng.standard_normal((m, 24))
                e = np.sqrt((x.reshape(m, 3, 8) ** 2).sum(-1))
                out = np.full((m, 125), np.inf, dtype=np.float32)
                for g in gains:
                    d = np.maximum(e[:, None, :] - g * rad[None, :, :], 0.0)
                    out = np.minimum(out, (GRAN_PROD * g * g + (d * d).sum(-1)).astype(np.float32))
                return out

            ea, eb = bc(me), bc(me)
            pair = np.full(me, np.inf, dtype=np.float32)
            for ja, jb in chosen:
                pair = np.minimum(pair, ea[:, ja] + eb[:, jb])
            single = ea[:, i0] + eb[:, i0]
            val = float(pair.mean())
            if best is None or val < best[0]:
                best = (val, gains, pair.astype(np.float64), None)
            if (f1, f2) == (1.0, 1.0):
                ref_single = single.astype(np.float64)
    val, gains, pair, _ = best
    d, e = db(pair, ref_single)
    print(f"{'lead 3: two blocks, six sections, prefix profiles, widths 9..13':78s} {d:+17.3f}  {e:.3f}")
    print(f"      (reference: two Tetra-like blocks at the reference gains, MSE/dim {ref_single.mean() / 48:.5f})")


if __name__ == "__main__":
    sys.exit(main())
