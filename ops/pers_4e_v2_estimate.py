"""4E v2's token on the sealed 4B, estimated from R3's measured phase table.

    python3 -I ops/pers_4e_v2_estimate.py

Measured input (docs/mesures/pers-4e-r3-2026-11-05.txt, us a token, median of 62 stamped tokens): each
phase's wall and mean CTA work at a 5-token prompt and at 8,192. Everything else is an assumption, written
here with its range; nothing is fitted. What v2 changes, and how this script carries it:

  attention         rebuilt (llvq_pers.cuh section 10): a job is one (split, KV head) on a whole CTA, each
                    step stages 32 positions once for the 4 query heads, with the next step in flight. At a
                    short context: 8 jobs, ceil(l / 8) steps each (l from 6 to 69 over 64 tokens, mean 4.6),
                    a step costing one L2 round trip at best and two at worst; plus a fixed job cost (q
                    staging, state init, unit fold, the output) and the phase's grid wait. At 8,192: the KV
                    cache streams from DRAM, 8 KV heads x 8,257 positions x 512 B = 33.8 MB a layer.
  norm/RoPE/append  one head a CTA on 48 CTAs instead of 16 a CTA on 3 (whose f64 RoPE shared one SM's FP64
                    units): a head's work plus the grid wait.
  rows              the rows' phases (q/k/v, o, MLP, head) stream 1.377 GB a token; R3's mean CTA work gives
                    290 GB/s. v2's 512-thread symbols keep the next block or word in flight, its 768-thread
                    symbols run 24 warps a SM instead; both are estimated as a range of stream rates, and
                    each phase's wall keeps R3's ratio of wall to mean CTA work (the split and the waits).
"""

GB = 1.377e9
LAYERS = 36

# R3, us a token: (wall, mean CTA work), at 5 and at 8,192 prompt tokens.
R3 = {
    5: {"q/k/v": (889.1, 729.4), "norm": (220.4, 5.7), "attention": (2354.6, 2254.3), "o": (590.4, 517.9),
        "MLP": (3199.9, 3125.5), "head": (377.1, 376.7)},
    8192: {"q/k/v": (894.3, 734.1), "norm": (221.1, 5.8), "attention": (13273.3, 9346.3), "o": (590.1, 519.0),
           "MLP": (3209.9, 3134.1), "head": (383.2, 382.8)},
}
ROWS = ("q/k/v", "o", "MLP", "head")
BARRIER_US = 0.786  # a grid wait at 142 CTAs, L1 microbenchmark (measured)


def rows_us(ctx, gbps):
    """The rows' walls at a stream rate: mean work 1.377 GB / rate, times R3's wall / mean ratio."""
    r3 = R3[ctx]
    mean = sum(r3[p][1] for p in ROWS)
    wall = sum(r3[p][0] for p in ROWS)
    return GB / (gbps * 1e3) * wall / mean


def attention_us(ctx, step_us, job_us, kv_gbps):
    if ctx == 5:
        steps = sum(min(8, -(-(5 + t + 1) // 8)) for t in range(1, 64)) / 63  # l = 6 .. 69 over 63 decodes
        return LAYERS * (steps * step_us + job_us + BARRIER_US)
    kv = 8 * 8257 * 512  # bytes a layer
    # 33 splits x 8 KV heads = 264 jobs on 142 CTAs: the stream bound, or two rounds of 8 steps; plus the
    # split fold (5 chunks of 8, one round trip each) and the wait
    stream = kv / (kv_gbps * 1e3)
    rounds = 2 * 8 * step_us
    return LAYERS * (max(stream, rounds) + job_us + 5 * step_us + BARRIER_US)


def norm_us(head_us):
    return LAYERS * (head_us + BARRIER_US)


def token(ctx, gbps, step_us, job_us, kv_gbps, head_us):
    parts = {
        "rows": rows_us(ctx, gbps),
        "attention": attention_us(ctx, step_us, job_us, kv_gbps),
        "norm/RoPE": norm_us(head_us),
    }
    return parts, sum(parts.values())


def main():
    print("4E v2 on the sealed 4B, estimated from R3's phase table (us a token)")
    print(f"  R3 rows: mean CTA work {sum(R3[5][p][1] for p in ROWS):.0f} us at 5 tokens = "
          f"{GB / sum(R3[5][p][1] for p in ROWS) / 1e3:.0f} GB/s; wall {sum(R3[5][p][0] for p in ROWS):.0f} us")
    cases = {
        "low (no stream gain)": dict(gbps=290, step_us=1.2, job_us=3.0, kv_gbps=550, head_us=2.0),
        "middle": dict(gbps=380, step_us=0.8, job_us=2.0, kv_gbps=650, head_us=1.2),
        "high": dict(gbps=450, step_us=0.5, job_us=1.2, kv_gbps=750, head_us=0.8),
    }
    for ctx in (5, 8192):
        print(f"\n  context {ctx}: R3 token {sum(w for w, _ in R3[ctx].values()):.0f} us")
        for name, c in cases.items():
            parts, t = token(ctx, **c)
            detail = ", ".join(f"{k} {v:.0f}" for k, v in parts.items())
            print(f"    {name:<22} rows at {c['gbps']} GB/s: {t:7.0f} us = {1e6 / t:5.1f} tok/s ({detail})")
    need = 1e6 / 300
    fixed = attention_us(5, 0.8, 2.0, 650) + norm_us(1.2)
    r3 = R3[5]
    ratio = sum(r3[p][0] for p in ROWS) / sum(r3[p][1] for p in ROWS)
    print(f"\n  300 tok/s is {need:.0f} us a token; with the middle attention and norm ({fixed:.0f} us), the rows "
          f"must stream at {GB / ((need - fixed) / ratio * 1e3):.0f} GB/s or more on the mean CTA's work "
          f"(computed from the estimates above)")


if __name__ == "__main__":
    main()
