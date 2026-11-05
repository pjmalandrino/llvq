"""Where 4E's token goes on the sealed 4B: the phases' critical paths, from the kernel's own work split.

    python3 -I ops/pers_4e_model.py

The 4E kernel (llvq-cuda/kernels/pers_4e.cu) runs one CTA of 512 threads a SM, 142 CTAs, 16 warps a
SM. Every phase gives each CTA a fixed contiguous slice of the rows (`nqkv * c / G`, pers.rs::slices),
and inside a CTA each warp claims one whole row at a time, one block (Tetra) or one word (int4) a lane
per iteration, with no load of the next iteration in flight (pers_tetra_row<24, false, false>,
pers_q4_row). A phase ends when its slowest CTA's last warp ends, then a grid barrier.

What is *computed* here, from the shapes and the kernel's split alone: the serial iterations on each
phase's slowest CTA (each CTA's 16 warps simulated claiming its rows in order, the next row to the first
free warp), against the same work spread evenly over all 2,272 warps, and the barrier count.

What is *estimated*: the time of one iteration, from three floors of a SM round (every warp one
iteration), none fitted to the card's 7.51 ms:
  latency  tau (a DRAM load under load, 0.5-0.8 us, assumed) plus the warp's own instruction chain
  issue    4 warps a scheduler x the loop's instructions (devel ptxas SASS of pers_4e_tokens_f1_t512:
           231 a Tetra block, 65-70 an int4 word)
  LSU      16 warps x the loop's wavefronts (tetragather on the sealed 4B, tables in shared, x at
           stride 24; the int4 loop's 8 LDS.32 at 32-byte stride are 8-way conflicts)
and the per-layer fixed work every CTA repeats (norms, rotations, table staging), from instruction and
round-trip counts, a range.

Measured inputs (docs/mesures/pers-4e-r2-2026-11-05.txt, pers-l1-2026-11-04.txt): 4E 133.1 tok/s (7.513 ms
a token) at a 5-token context; a grid barrier 0.786 us at 142 x 768.
"""

import heapq

L, H, I, NH, NKV, VOCAB = 36, 2560, 9728, 32, 8, 151936
G, WARPS = 142, 16
N_INT4_DOWN = 12  # sealed 4B: 168 Tetra matrices of 252 (tetragather), so 12 downs are int4 g128
F_LO, F_HI = 2.20e9, 2.52e9
MEASURED_MS = 1e3 / 133.1
BARRIER_US = 0.786

# Iterations of one row on its slowest lane.
T_IT = {2560: -(-(2560 // 24) // 32), 9728: -(-(9728 // 24) // 32)}  # 4, 13
Q_IT = {2560: 2560 // 8 // 32, 4096: 4096 // 8 // 32, 9728: 9728 // 8 // 32}  # 10, 16, 38

# Bytes a token (computed, as pers_projection.py): Tetra streams, int4 rows, the int4 g64 head.
W_T, W_Q, W_H = 0.7487e9, 0.4094e9, 0.2188e9


def slices(n, g=G):
    return [(n * c // g, n * (c + 1) // g) for c in range(g)]


def makespan(costs):
    """16 warps claiming rows in order, each row to the first free warp: the CTA's end."""
    free = [0.0] * WARPS
    heapq.heapify(free)
    for c in costs:
        heapq.heappush(free, heapq.heappop(free) + c)
    return max(free)


def phase(row_cost, n_rows, slicing=True):
    """(slowest CTA's makespan, even spread over all warps) for rows costed by row_cost(r)."""
    total = sum(row_cost(r) for r in range(n_rows))
    if slicing:
        crit = max(makespan([row_cost(r) for r in range(lo, hi)]) for lo, hi in slices(n_rows))
    else:
        crit = total / (G * WARPS)
    return crit, total / (G * WARPS)


def token(t_t, t_q, epi, fixed_us, balanced=False):
    """One token, us: per phase (critical, even) and the sum. Costs in us: t_t a Tetra block-iteration,
    t_q an int4 word-iteration, epi the lane-0 epilogue of a Tetra row (its tail and scale loads)."""
    nq, nk, nv = NH * 128, NKV * 128, NKV * 128
    s = not balanced

    def qkv(r):
        return T_IT[2560] * t_t + epi if r < nq + nk else Q_IT[2560] * t_q

    rows = {
        "q/k/v": phase(qkv, nq + nk + nv, s),
        "o": phase(lambda r: Q_IT[4096] * t_q, H, s),
        "gate/up": phase(lambda r: 2 * (T_IT[2560] * t_t + epi), I, s),
        "down Tetra": phase(lambda r: T_IT[9728] * t_t + epi, H, s),
        "down int4": phase(lambda r: Q_IT[9728] * t_q, H, s),
        "head": phase(lambda r: Q_IT[2560] * t_q, VOCAB, s),
    }
    per_layer = {k: v for k, v in rows.items() if k not in ("down Tetra", "down int4", "head")}
    out = {k: (L * v[0], L * v[1]) for k, v in per_layer.items()}
    out["down"] = (
        (L - N_INT4_DOWN) * rows["down Tetra"][0] + N_INT4_DOWN * rows["down int4"][0],
        (L - N_INT4_DOWN) * rows["down Tetra"][1] + N_INT4_DOWN * rows["down int4"][1],
    )
    out["head"] = rows["head"]
    out["fixed"] = (L * fixed_us, L * fixed_us)
    out["barriers"] = ((2 + 7 * L) * BARRIER_US, (2 + 7 * L) * BARRIER_US)
    return out


def iteration_us(tau_us, f):
    """(t_t, t_q, epi) from the three floors of a SM round, at clock f."""
    cyc = 1e6 / f  # us a cycle
    # Tetra block: 231 instructions, ~65 wavefronts (words 3.7, tables in shared 17, x24 44), its chain
    # 231 issues plus three dependent shared lookups (~30 cycles each).
    t_t = max(tau_us + (231 + 90) * cyc, 4 * 231 * cyc, 16 * 65 * cyc)
    # int4 word: 65-70 instructions; LSU 19 (two LDS.128) to 67 (eight LDS.32, 8-way) wavefronts.
    t_q = max(tau_us + 70 * cyc, 4 * 70 * cyc, 16 * 43 * cyc)
    # lane 0's epilogue: rscale and a 16-wide f16 tail, independent loads, one round trip, then the
    # warp's five-step reduction.
    epi = tau_us + 100 * cyc
    return t_t, t_q, epi


# Per layer, every CTA, us (estimated ranges): ln1 scale + apply + the 2,560 rotation (~2-3); the o
# phase's 4,096-wide ctx staging (~0.5); the MLP's table restage (18.8 KB, ~1), ln2 + its rotation
# (~2-3); the 9,728 down input and rotation (~2-3, Tetra downs); norm/RoPE/append (~1-2); attention
# at a short context, both halves (~2-4).
FIXED_LO, FIXED_HI = 10.5, 16.5


def show(label, parts, ref=None):
    crit = sum(v[0] for v in parts.values())
    print(f"  {label}: {crit / 1e3:.2f} ms a token" + (f" ({crit / 1e3 / ref:.2f} x the measured {ref:.2f})" if ref else ""))
    for k, (c, e) in parts.items():
        print(f"    {k:<10} {c / 1e3:6.3f} ms   even spread {e / 1e3:6.3f} ms   {100 * c / crit:5.1f} %")
    return crit


def main():
    print(f"4E on the sealed 4B: {L} layers, grid {G} x 512 (16 warps a SM), {2 + 7 * L} grid barriers a token")
    # Computed: serial iterations, critical against even, phase by phase (unit costs).
    print("Serial iterations of the slowest warp of the slowest CTA, against the even spread (computed):")
    for name, n, cost, kind in (
        ("q/k/v", NH * 128 + 2 * NKV * 128, None, "mixed"),
        ("o", H, Q_IT[4096], "int4"),
        ("gate/up", I, 2 * T_IT[2560], "Tetra"),
        ("down Tetra", H, T_IT[9728], "Tetra"),
        ("down int4", H, Q_IT[9728], "int4"),
        ("head", VOCAB, Q_IT[2560], "int4"),
    ):
        if cost is None:
            nqk = NH * 128 + NKV * 128
            c, e = phase(lambda r: T_IT[2560] if r < nqk else Q_IT[2560], n)
            ct = max(makespan([T_IT[2560] for r in range(lo, hi) if r < nqk]) for lo, hi in slices(n))
            print(f"  {name:<10} {n:6d} rows, {n / G:6.1f} a CTA: slowest {c:5.1f} (a v slice, int4 rows of 10), "
                  f"a q/k slice {ct:4.1f}, even {e:5.2f} iterations")
            continue
        c, e = phase(lambda r: cost, n)
        print(f"  {name:<10} {n:6d} rows, {n / G:6.1f} a CTA, {n / G / WARPS:5.2f} a warp: slowest {c:5.1f}, "
              f"even {e:6.2f}: {100 * e / c:5.1f} % of the warps' time does work ({kind})")
    print()
    for tau in (0.5, 0.65, 0.8):
        for f, fl in ((F_HI, "2.52"), (F_LO, "2.20")):
            t_t, t_q, epi = iteration_us(tau, f)
            for fixed in (FIXED_LO, FIXED_HI):
                parts = token(t_t, t_q, epi, fixed)
                crit = sum(v[0] for v in parts.values()) / 1e3
                even = sum(v[1] for v in parts.values()) / 1e3
                print(f"  tau {tau:.2f} us, {fl} GHz, fixed {fixed:4.1f} us a layer: t_T {t_t:.3f} t_Q {t_q:.3f} "
                      f"epi {epi:.3f} us -> {crit:.2f} ms ({crit / MEASURED_MS:.2f} x measured), "
                      f"even spread {even:.2f} ms")
    print()
    t_t, t_q, epi = iteration_us(0.65, 2.36e9)
    parts = token(t_t, t_q, epi, 0.5 * (FIXED_LO + FIXED_HI))
    crit = show("middle point (tau 0.65 us, 2.36 GHz, fixed 13.5 us a layer)", parts, MEASURED_MS)
    stream_ms = (W_T + W_Q + W_H) / 1e6
    work = sum(parts[k][0] for k in ("q/k/v", "o", "gate/up", "down", "head"))
    even = sum(parts[k][1] for k in ("q/k/v", "o", "gate/up", "down", "head"))
    gb = (W_T + W_Q + W_H) / 1e9
    print(f"  the rows' phases stream {gb:.3f} GB in {work / 1e3:.2f} ms: {gb / (work * 1e-6):.0f} GB/s; "
          f"evenly spread {gb / (even * 1e-6):.0f} GB/s")
    print(f"  loss to the split (slowest CTA against the even spread): {(work - even) / 1e3:.2f} ms a token")
    print(f"  fixed work and barriers: {(parts['fixed'][0] + parts['barriers'][0]) / 1e3:.2f} ms a token")
    # The split's share does not depend on the iteration's time, only on the Tetra/int4 ratio of it.
    print("The rows' phases, slowest CTA over the even spread, by t_Q / t_T (computed, any t_T):")
    for r in (0.6, 0.7, 0.8, 0.9, 1.0):
        p = token(1.0, r, 0.9 * r, 0.0)
        w = sum(p[k][0] for k in ("q/k/v", "o", "gate/up", "down", "head"))
        e = sum(p[k][1] for k in ("q/k/v", "o", "gate/up", "down", "head"))
        print(f"  t_Q / t_T {r:.1f}: x{w / e:.3f}, the split holds {100 * (1 - e / w):4.1f} % of the rows' time")
    fixed = (parts["fixed"][0] + parts["barriers"][0]) / 1e3
    rows_ms = MEASURED_MS - fixed
    print(f"If the measured {MEASURED_MS:.2f} ms has this structure: rows {rows_ms:.2f} ms, of which the split "
          f"{rows_ms * (1 - even / work):.2f} ms; evenly split, the rows would stream at "
          f"{gb / (rows_ms * even / work * 1e-3):.0f} GB/s (estimated)")


if __name__ == "__main__":
    main()
