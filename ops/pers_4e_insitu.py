"""Rule D of the row-loop prereg: 4E's MLP and q/k/v phases predicted from the row loops' own card cycles, against
4E's phase tables, and what is left over.

    python3 -I ops/pers_4e_insitu.py

Measured inputs (labels as in the journals):
  row loops alone, cycles a step (32 blocks of one row) and clock: docs/mesures/pers-rowloop-2026-11-05.txt;
  4E v2 and v3 phase tables, mean CTA work in us a token: pers-4e-v2-2026-11-05.txt, pers-4e-v3-2026-11-05.txt;
  int4 rows in situ, 587 GB/s on the mean CTA's work (o and head phases of 4E v2): pers-4e-v3-host-2026-11-05.txt;
  rot_apply alone, 1,024 threads, one CTA: 7.82 us at 2,560 and 18.40 at 9,728: lot1-4b-2026-11-04.txt;
  grid barrier, 142 CTAs: 0.786 us at 768 threads: pers-l1-2026-11-04-brut/bench.txt.
A CTA's work in a phase is set by its busiest warp: rows a warp under 4E's split (pers::slices, even, then
`lo + warp + k x 16`), times ceil(nblocks / 32) steps a row, times the measured cycles a step.
"""

GRID, NW, LAYERS, TETRA_DOWN_LAYERS = 142, 16, 36, 24
H, I, NQ, NKV = 2560, 9728, 4096, 1024
CYC = {"v2": {2560: 1616, 9728: 1272, "mhz": 2456}, "v3": {2560: 1837, 9728: 1736, "mhz": 2502}}
MEAN_WORK = {"v2": {"q/k/v": 763.2, "MLP": 2834.3}, "v3": {"q/k/v": 847.1, "MLP": 2975.4}}
INT4_GBS = 587.0
Q4 = 0.53125  # bytes a weight, int4 g128


def slices(n):
    return [(n * c // GRID, n * (c + 1) // GRID) for c in range(GRID)]


def warp_rows(lo, hi, w):
    return -(-(hi - lo - w) // NW) if hi > lo + w else 0


def busiest(n):
    """Mean over CTAs of the busiest warp's rows, and of the mean warp's rows."""
    mx = mn = 0.0
    for lo, hi in slices(n):
        r = [warp_rows(lo, hi, w) for w in range(NW)]
        mx += max(r)
        mn += sum(r) / NW
    return mx / GRID, mn / GRID


def step_us(arm, d_in):
    return CYC[arm][d_in] / CYC[arm]["mhz"]


def int4_us(rows, d_in):
    """int4 rows at the in-situ rate on the mean CTA's work (that rate already carries the split)."""
    return rows * d_in * Q4 / INT4_GBS / 1e3


def phases(arm):
    gu_max, gu_mean = busiest(I)  # gate/up pairs; a pair is 2 rows of 4 steps
    dn_max, dn_mean = busiest(H)  # down rows, 13 steps at d_in 9,728
    gu = gu_max * 2 * 4 * step_us(arm, 2560)
    gu_ideal = gu_mean * 2 * 4 * step_us(arm, 2560)
    dn_t = dn_max * 13 * step_us(arm, 9728)
    dn_t_ideal = dn_mean * 13 * step_us(arm, 9728)
    dn_q = int4_us(H, I)
    mlp = LAYERS * gu + TETRA_DOWN_LAYERS * dn_t + (LAYERS - TETRA_DOWN_LAYERS) * dn_q
    mlp_split = LAYERS * (gu - gu_ideal) + TETRA_DOWN_LAYERS * (dn_t - dn_t_ideal)
    # q/k/v under v3's mixed split: q | k rows by warp, then the int4 v rows (one a warp at most)
    qk_max, qk_mean = busiest(NQ + NKV)
    qk = qk_max * 4 * step_us(arm, 2560)
    qk_ideal = qk_mean * 4 * step_us(arm, 2560)
    v = int4_us(NKV, H)
    qkv = LAYERS * (qk + v)
    return {"MLP": mlp, "q/k/v": qkv, "split MLP": mlp_split, "split q/k": LAYERS * (qk - qk_ideal),
            "gu": gu, "dn_t": dn_t, "dn_q": dn_q, "qk": qk, "v": v, "busiest": (gu_max, dn_max, qk_max),
            "means": (gu_mean, dn_mean, qk_mean)}


def main():
    print("Rule D: 4E's phases from the row loops' card cycles (us a token, mean CTA work)")
    for arm in ("v2", "v3"):
        p = phases(arm)
        gm, dm, qm = p["busiest"]
        print(f"\n  {arm}: busiest warp, mean over CTAs: {gm:.2f} gate/up pairs, {dm:.2f} down rows, {qm:.2f} q|k rows "
              f"(mean warp {p['means'][0]:.2f} / {p['means'][1]:.2f} / {p['means'][2]:.2f})")
        print(f"    a layer: gate/up {p['gu']:.1f}, Tetra down {p['dn_t']:.1f}, int4 down {p['dn_q']:.1f}, "
              f"q|k {p['qk']:.1f}, int4 v {p['v']:.1f}")
        for ph in ("MLP", "q/k/v"):
            m = MEAN_WORK[arm][ph]
            print(f"    {ph:<6} rows predicted {p[ph]:7.0f}  measured {m:7.1f}  ratio {p[ph] / m:.3f}  "
                  f"left over {m - p[ph]:6.0f} ({(m - p[ph]) / LAYERS:.1f} a layer)")
        print(f"    of the rows predicted, the uneven split inside a CTA: MLP {p['split MLP']:.0f}, "
              f"q|k {p['split q/k']:.0f}")
    l2, l3 = (MEAN_WORK[a]["MLP"] - phases(a)["MLP"] for a in ("v2", "v3"))
    d_pred = phases("v3")["MLP"] - phases("v2")["MLP"]
    d_meas = MEAN_WORK["v3"]["MLP"] - MEAN_WORK["v2"]["MLP"]
    print(f"\n  MLP left over: v2 {l2:.0f}, v3 {l3:.0f}; v3 - v2: rows predicted {d_pred:+.0f}, measured {d_meas:+.0f}")
    print("  The left-over does not follow the loop: a fixed cost a layer, outside the rows.")
    rot = (7.82, 18.40)
    named = LAYERS * rot[0] + TETRA_DOWN_LAYERS * rot[1] + LAYERS * 0.786
    print(f"\n  Named from isolated measurements (1,024-thread rot_apply, a 768-thread barrier; 4E runs 512):")
    print(f"    MLP: ln2's rotation {LAYERS * rot[0]:.0f} + the Tetra down's {TETRA_DOWN_LAYERS * rot[1]:.0f} + "
          f"its inner barrier {LAYERS * 0.786:.0f} = {named:.0f} of {l3:.0f}")
    print(f"    q/k/v: ln1's rotation {LAYERS * rot[0]:.0f} of {MEAN_WORK['v3']['q/k/v'] - phases('v3')['q/k/v']:.0f}")


if __name__ == "__main__":
    main()
