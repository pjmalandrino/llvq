# Arithmetic for wave5-deep/brick.md §3.4-§3.6 and §4: launch geometries, staging, atomics,
# the floor-plus-work model, the whole-step r(8), the integer bounds and the S grid.
#
# Not repository code. A tiny pure-Python script, run by the author with
#   python3 -I geometry_budget.py > geometry_budget.out
# on 2026-11-02. No file or network I/O besides stdout, no repository access. Inputs are
# quoted beside each constant with their source; everything printed is *computed* from them,
# except the lines marked ESTIMATED, which carry at least one estimated input.
import math

# ---- shapes of Qwen3-4B (configs, docs/fiche-4b.md) -------------------------------------
L = 36
TYPES = {  # name: (rows d_out, d_in)
    'q': (4096, 2560), 'k': (1024, 2560), 'v': (1024, 2560), 'o': (2560, 4096),
    'gate': (9728, 2560), 'up': (9728, 2560), 'down': (2560, 9728)}
# bench beta: the 252 records of the bare Tetra file (banc-252); served Tetra: 168 records,
# q, k, gate, up in 36 layers and down in layers 0-11 and 24-35 (configs/qwen3-4b-tetra-e4.json:
# "v_proj + o_proj + down_proj@12-23 as int4 g128").
BENCH = [(t, L) for t in TYPES]
SERVED = [('q', L), ('k', L), ('gate', L), ('up', L), ('down', 24)]


def mats(sel):
    for t, n in sel:
        R, d = TYPES[t]
        K = d // 24
        yield t, n, R, d, K


def totals(sel):
    rows = sum(n * R for _, n, R, _, _ in mats(sel))
    rk = sum(n * R * K for _, n, R, _, K in mats(sel))
    return rows, rk


for name, sel in (('bench 252', BENCH), ('served Tetra 168', SERVED)):
    rows, rk = totals(sel)
    print(f"{name}: rows {rows:,}, row-blocks {rk:,}")
rows252, rk252 = totals(BENCH)
print(f"bytes per row-block on the bench: 0.98e9 / {rk252:,} = {0.98e9 / rk252:.2f} B (banc-252 GB/pass)")

# ---- geometries -----------------------------------------------------------------------------
# S  : served, one warp per row, 8 rows per CTA, whole-row f32 staging (tv_tetra48_h.cu).
# Gx : brick, CTA = Rc rows (Rc/16 tiles of 16) x Cin K-groups inside the CTA, 16 warps,
#      K split Cx ways across CTAs, int16 planes staged (2 B a coordinate), int64 combine.
GEOMS = {'G2': (64, 4, 4), 'G3': (128, 2, 8), 'G4': (256, 1, 16)}
TABLES = 16384 + 2300 + 1024   # rows table + v3 trellis tables + S table, bytes (brick.md §3.5)


def served(sel, k):
    warps = sum(n * R for _, n, R, _, _ in mats(sel))
    ctas = sum(n * R // 8 for _, n, R, _, _ in mats(sel))
    stage = sum(n * (R // 8) * d * 4 for _, n, R, d, _ in mats(sel)) * k
    return warps, ctas, stage


def brick(sel, k, Rc, Cin, Cx):
    warps = ctas = stage = atom = 0
    for _, n, R, d, K in mats(sel):
        rg = -(-R // Rc)
        ctas += n * rg * Cx
        warps += n * rg * Cx * 16
        stage += n * rg * Cx * (-(-K // Cx)) * 24 * 2 * k
        atom += n * R * k * Cx if Cx > 1 else 0
    shared = 16 * 1024 + Cin * 2 * 384 * 2 * k + Rc * k * 8   # tiles + rings + int64 scratch
    return warps, ctas, stage, atom, shared


print("\ngeometry      k   warps/pass   CTAs/pass   L2->smem staging   int64 atomics   smem/CTA   tables if staged per CTA")
for k in (1, 8, 16):
    w, c, s = served(BENCH, k)
    print(f"S (served)  {k:3d}  {w:11,}  {c:10,}   {s / 1e9:7.3f} GB/pass      {'-':>12}   {'-':>8}   (read through L1)")
    for g, (Rc, Cin, Cx) in GEOMS.items():
        w, c, s, a, sh = brick(BENCH, k, Rc, Cin, Cx)
        print(f"{g} ({Rc},{Cin},{Cx}) {k:3d}  {w:11,}  {c:10,}   {s / 1e9:7.3f} GB/pass   {a / 1e6:9.1f} M   {sh / 1024:6.1f} KB   {c * TABLES / 1e9:.2f} GB")
# G1: one 16-row tile per CTA, 16 K-groups inside it, the WHOLE activation row staged once per CTA,
# no cross-CTA split and so no atomics. Its shared memory grows with d_in x k.
def g1(sel, k):
    stage = sum(n * (R // 16) * K * 24 * 2 * k for _, n, R, _, K in mats(sel))
    smem = {d: 16 * 1024 + (d // 24) * 24 * 2 * k + 16 * 16 * k * 8 for _, _, _, d, _ in mats(sel)}
    return stage, smem


for k in (1, 8, 16):
    st, sm = g1(BENCH, k)
    fits = {d: ('fits' if v <= 101376 else 'OVER 99 KiB') for d, v in sm.items()}
    print(f"G1 (16,16,1) {k:3d}  {rows252:11,}  {rows252 // 16:10,}   {st / 1e9:7.3f} GB/pass         0.0 M   "
          + ", ".join(f"d_in {d}: {v / 1024:.1f} KB {fits[d]}" for d, v in sorted(sm.items())))

# ---- growth of the floor with k under ASSUMED L2 rates (ESTIMATED: neither rate is measured) ----
# dF(k) = (staging(k) - staging(1)) / BW + (atomics(k) - atomics(1)) / AR
print("\nESTIMATED dF(k) in ms, best geometry for each pair of assumed L2 rates (G1 counted where it fits)")
for k in (8, 16):
    for bw in (3e12, 7e12):
        for ar in (0.1e12, 0.5e12):
            best = []
            st1, _ = g1(BENCH, 1)
            stk, smk = g1(BENCH, k)
            if all(v <= 101376 for v in smk.values()):
                best.append(('G1', (stk - st1) / bw * 1e3))
            for g, (Rc, Cin, Cx) in GEOMS.items():
                _, _, s1, a1, _ = brick(BENCH, 1, Rc, Cin, Cx)
                _, _, sk, ak, shk = brick(BENCH, k, Rc, Cin, Cx)
                if shk <= 101376:
                    best.append((g, (sk - s1) / bw * 1e3 + (ak - a1) / ar * 1e3))
            b = min(best, key=lambda t: t[1])
            r_lo, r_hi = 1 + b[1] / 4.6, 1 + b[1] / 3.4
            print(f"  k={k:2d} L2 {bw / 1e12:.0f} TB/s, {ar / 1e12:.1f} T atomics/s: best {b[0]} dF {b[1]:.3f} ms"
                  f" -> r(k) from the floor alone {r_lo:.3f}-{r_hi:.3f} (t(1) 3.4-4.6 ms)  [all: "
                  + ", ".join(f"{g} {v:.2f}" for g, v in best) + "]")

# the verifier's 4-way figure on the 168 served Tetra matrices
rows168, _ = totals(SERVED)
print(f"\nserved Tetra rows {rows168:,}; x 8 tokens x 4 K-chunks = {rows168 * 8 * 4 / 1e6:.1f} M int64 atomics a pass")

# ---- occupancy ------------------------------------------------------------------------------
REGS_SM, MAXW = 65536, 48   # sm_89
for regs, threads, label in ((40, 256, 'tv_tetra48_h, 40 regs (tv_tetra48_h.cu:180-182)'),
                             (64, 512, 'brick 16 warps at 64 regs (ESTIMATED regs)'),
                             (84, 512, 'brick 16 warps at 84 regs (ESTIMATED regs)')):
    ctas = min(REGS_SM // (regs * threads), MAXW * 32 // threads)
    print(f"occupancy {label}: {ctas} CTAs/SM, {ctas * threads // 32} warps/SM")

# ---- floor-plus-work model at k = 1 (banc-252, measured inputs) -----------------------------
nullk, tetra, gb, bw = 2.294, 3.734, 0.98, 864e9
dram = gb * 1e9 / bw * 1e3
print(f"\nW = t(Tetra48) - t(nullk) = {tetra - nullk:.3f} ms; DRAM floor of {gb} GB at 864 GB/s = {dram:.3f} ms;"
      f" W runs at {dram / (tetra - nullk) * 100:.0f} % of the DRAM rate")
tetra48h = 3.85   # ESTIMATED, prereg-brick-2 §6 (3.734 plus the served f16 tail and output)
print(f"served geometry floor + DRAM = {nullk + dram:.3f} ms = {(nullk + dram) / tetra48h:.3f} of tetra48h's estimated {tetra48h} ms (ESTIMATED denominator)")
clk = 2.52e9
print(f"measured pass per row-block per SM: {tetra * 1e-3 * clk * 142 / rk252:.2f} clk; DRAM part {dram * 1e-3 * clk * 142 / rk252:.2f} clk")

# ---- whole-step r(8) (ESTIMATED inputs: Tetra share, SIMT int4 and head rows per token) ------
def r8(step, tshare_lo, tshare_hi, rk_lo=1.00, rk_hi=1.20):
    lo = (rk_lo - 1) * tshare_lo / 7 + 0.07 + 0.04
    hi = (rk_hi - 1) * tshare_hi / 7 + 0.13 + 0.05
    return lo, hi, 1 + 7 * lo / step, 1 + 7 * hi / step


for rk_hi in (1.20, 1.35):
    lo, hi, a, b = r8(7.44, 2.5, 2.8, rk_hi=rk_hi)
    print(f"\nESTIMATED kernel r(8) 1.00-{rk_hi:.2f}, today (7.44 ms step, Tetra share 2.5-2.8 ms):"
          f" extra per token {lo:.3f}-{hi:.3f} ms, whole-step r(8) {a:.3f}-{b:.3f}")
    lo, hi, a, b = r8(2.6, 1.13, 1.5, rk_hi=rk_hi)
    print(f"ESTIMATED kernel r(8) 1.00-{rk_hi:.2f}, lean (2.6 ms step, Tetra share 1.13-1.5 ms):"
          f" extra per token {lo:.3f}-{hi:.3f} ms, whole-step r(8) {a:.3f}-{b:.3f}")

# ---- integer bounds and the S grid ----------------------------------------------------------
s = math.sqrt(24 * 416)
P = s * 32767
print(f"\nsum|y| <= {s:.3f}; |P| < 2^{math.log2(P):.3f}; after shift 3: 2^{math.log2(8 * P):.3f}")
for nb in (405, 1067):
    print(f"|I| < 2^{math.log2(8 * P * 2 ** 28 * nb):.3f} at {nb} blocks with |S_int| < 2^28")
print(f"S span within one level: sqrt(26) = {math.sqrt(26):.3f} = {math.log2(math.sqrt(26)):.3f} binades")
print(f"|S_int| < 2^28 needs a span of at most 4 binades: c1/c0 <= {16 / math.sqrt(26):.3f}")
print(f"planesbench GSCALE 1.375/0.625 = {1.375 / 0.625:.2f}: span {math.log2(1.375 / 0.625 * math.sqrt(26)):.2f} binades")
print(f"nonzero scales: 2 levels x 26 shells = {2 * 26}")
