"""Rule D of the row-loop prereg: 4E's MLP and q/k/v phases predicted from the row loops' own card cycles, against
4E's phase tables, and what is left over. Then rule C of the shaker prereg: the same rows model against the shaker's
inner parts, and the serial round trips of the three parts every CTA computes redundantly (ln1, ln2, the down input).

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


# Shaker inner parts, 5-token prompt, mean CTA work, us a token (docs/mesures/shaker-4e-2026-11-05.txt, measured).
SHAKER = {
    "v3": {"ln1": 417.2, "qkv rows": 436.1, "ln2": 265.8, "gu rows": 1237.4, "wait": 191.3, "input": 798.1,
           "dn rows": 561.5},
    "v4": {"ln1": 406.6, "qkv rows": 404.2, "ln2": 261.6, "gu rows": 1184.4, "wait": 189.0, "input": 799.0,
           "dn rows": 462.0},
}
TOKEN_V4 = {5: 5268.8, 8192: 7136.6}  # timed 4E v4, us a token (the same journal)
LOOP = {"v3": "v3", "v4": "v2"}  # the row loop each form runs
BAND = 0.15
DRAM_GBS = 864.0  # L40S datasheet
# Serial global round trips a thread makes in each redundant part, read from the SASS of pers_4e_tokens_f1_t512
# (devel ptxas 12.4; every loop below loads one element an iteration and uses it before the next load issues):
#   exact_rms_scale at 2,560: 5 (a warp's two virtual warps, 3 + 2 columns); exact_rms_apply: 5 (x, alpha, signs);
#   pers_mix_column at k = 5: 5 (one row of `small` an iteration, 32 predicated generic loads);
#   pers_rot_load at 9,728: 19; pers_mix_column at k = 19: 19; the int4 widening: 19.
RT = {"ln": 15, "tetra input": 38, "int4 input": 19}


def rule_c():
    print("\nRule C of the shaker prereg: the rows parts against the row-loop cycles (mean CTA work, us a token)")
    for form in ("v3", "v4"):
        p = phases(LOOP[form])
        m = SHAKER[form]
        rows = {"qkv rows": p["q/k/v"], "gu rows": LAYERS * p["gu"],
                "dn rows": TETRA_DOWN_LAYERS * p["dn_t"] + (LAYERS - TETRA_DOWN_LAYERS) * p["dn_q"]}
        print(f"  {form} ({LOOP[form]}'s cycles)")
        for k, pred in rows.items():
            r = pred / m[k]
            verdict = "holds" if abs(r - 1) <= BAND else "outside"
            print(f"    {k:<9} predicted {pred:7.0f}  measured {m[k]:7.1f}  ratio {r:.3f}  {verdict}")
        # the down rows, if the Tetra term holds: what the int4 down rows would run at
        q = (m["dn rows"] - TETRA_DOWN_LAYERS * p["dn_t"]) / (LAYERS - TETRA_DOWN_LAYERS)
        gbs = H * I * Q4 / q / 1e3
        print(f"    if the Tetra down term holds, an int4 down layer takes {q:.1f} us: {gbs:.0f} GB/s, "
              f"above DRAM's {DRAM_GBS:.0f} (its words come from L2, prefetched before the inner wait)")

    print("\nThe redundant parts as serial round trips (us a token, v4, mean CTA work)")
    m = SHAKER["v4"]
    n_rt = {"ln1": LAYERS * RT["ln"], "ln2": LAYERS * RT["ln"],
            "input": TETRA_DOWN_LAYERS * RT["tetra input"] + (LAYERS - TETRA_DOWN_LAYERS) * RT["int4 input"]}
    for k, n in n_rt.items():
        print(f"    {k:<6} {m[k]:6.1f} us over {n:5d} round trips a token: at most {m[k] / n:.2f} us each "
              f"({m[k] / n * 2520:.0f} cycles at 2,520 MHz)")
    print("    an unloaded L2 hit is about 270 cycles: each round trip here waits 4.5 to 7 times that, on lines all "
          "142 CTAs read at once")

    # the lever: every redundant part in one or two round trips, `small` staged in shared memory during the
    # preceding wait, the k = 19 mix from shared memory. Bit for bit: the same operations on the same values.
    print("\nThe lever, estimated (us a token, v4 at a 5-token prompt)")
    price = {k: m[k] / n for k, n in n_rt.items()}
    # what stays per layer: WHT rounds, the mix's FMAs, the RMS reductions and barriers (estimated, issue-bound)
    after = {
        "ln1": LAYERS * (2 * price["ln1"] + 1.0),
        "ln2": LAYERS * (2 * price["ln2"] + 1.0),
        "input": TETRA_DOWN_LAYERS * (2 * price["input"] + 1.9) + (LAYERS - TETRA_DOWN_LAYERS) * (price["input"] + 0.3),
    }
    # the mix's issue count alone (the floor of the gain if round trips were not the cost): 105 instructions an
    # iteration of g, KMAX = 32 predicated, 19 iterations, 4 warps a scheduler, against about 1,900 cycles from shared
    mix_only = TETRA_DOWN_LAYERS * ((105 * 19 * 4 - 1900) / 2520)
    int4_dram = (LAYERS - TETRA_DOWN_LAYERS) * 5.0  # the int4 down's DRAM fill no longer hidden, at most
    saved = sum(m[k] - after[k] for k in after)
    for k in after:
        print(f"    {k:<6} {m[k]:6.1f} -> {after[k]:5.0f}  saves {m[k] - after[k]:5.0f}")
    lo = mix_only
    hi = saved
    mid = saved - int4_dram
    for ctx, t in TOKEN_V4.items():
        print(f"    context {ctx}: token {t:.0f} us, {1e6 / t:.1f} tok/s -> point {1e6 / (t - mid):.0f} tok/s, "
              f"range [{1e6 / (t - lo):.0f}, {1e6 / (t - hi):.0f}]")
    print(f"    low end: only the mix's issue count ({lo:.0f} us); high end: every round trip at its present price "
          f"({hi:.0f}); point: the high end less the int4 down's DRAM fill ({int4_dram:.0f})")


if __name__ == "__main__":
    main()
    rule_c()
