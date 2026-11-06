"""4E v3's token on the sealed 4B, estimated from the v2 card's phase table.

    python3 -I ops/pers_4e_v3_estimate.py

Measured input: docs/mesures/pers-4e-v2-2026-11-05.txt, us a token at a 5-token prompt and at 8,192, each phase's
wall and mean CTA work. Bytes are computed from the sealed 4B's shapes (the Tetra stream at 0.2616 B a weight,
pers_4e_model.py's W_T over its weights; int4 g128 at 0.53125, the g64 head at 0.5625). Everything else is an
assumption with its range; nothing is fitted to a v3 number.

What v3 changes:
  rows       the Tetra rows (q/k, gate/up, Tetra down) on the row stream: a warp's rows static, the lane's next
             4 blocks in flight across row boundaries, rscale, tail and residual loaded at the row's start, the
             SiLU lookup one pair late. v2's Tetra rows ran at about 250 GB/s on the mean CTA's work while its
             int4 rows ran at 530-590 (computed below): the gap is the per-row round trips the stream removes.
  q/k/v      every CTA takes the same share of q | k rows and of v rows (no int4-only CTAs at the end).
  attention  at a short context a step stages several tiles of each non-empty unit (2 steps for a row of 64
             positions instead of 8), and a CTA with an attention job issues its L2 prefetches after the job.
"""

LAYERS = 36
TB, Q4, H64 = 0.2616, 0.53125, 0.5625  # bytes a weight: Tetra stream, int4 g128, int4 g64 head
H, I, NQ, NKV, VOCAB = 2560, 9728, 4096, 1024, 151936
MB = {
    "qk_T": LAYERS * (NQ + NKV) * H * TB / 1e6,
    "v_Q": LAYERS * NKV * H * Q4 / 1e6,
    "o_Q": LAYERS * H * NQ * Q4 / 1e6,
    "gu_T": LAYERS * 2 * I * H * TB / 1e6,
    "dn_T": 24 * H * I * TB / 1e6,
    "dn_Q": 12 * H * I * Q4 / 1e6,
    "head": VOCAB * H * H64 / 1e6,
}
V2 = {  # (wall, mean CTA work), us a token
    5: {"q/k/v": (951.6, 763.2), "norm": (111.9, 23.8), "attention": (761.4, 260.2), "o": (493.5, 377.1),
        "MLP": (2933.5, 2834.3), "head": (373.0, 372.6)},
    8192: {"q/k/v": (966.7, 778.7), "norm": (111.3, 24.4), "attention": (2682.0, 2041.2), "o": (491.4, 378.0),
           "MLP": (2993.6, 2895.4), "head": (376.1, 375.7)},
}


def v2_rates(ctx, f_qkv, f_mlp, r_q):
    """v2's Tetra GB/s on the mean CTA's work, given the fixed work a layer and the int4 rate."""
    t = V2[ctx]
    tet_qkv = t["q/k/v"][1] - LAYERS * f_qkv - MB["v_Q"] / r_q * 1e3
    tet_mlp = t["MLP"][1] - LAYERS * f_mlp - MB["dn_Q"] / r_q * 1e3
    return (MB["qk_T"] + MB["gu_T"] + MB["dn_T"]) / (tet_qkv + tet_mlp) * 1e3, tet_qkv, tet_mlp


def token(ctx, r_t, f_qkv, f_mlp, r_q, attn_short_us):
    t = V2[ctx]
    qkv_mean = LAYERS * f_qkv + MB["v_Q"] / r_q * 1e3 + MB["qk_T"] / r_t * 1e3
    mlp_mean = LAYERS * f_mlp + MB["dn_Q"] / r_q * 1e3 + (MB["gu_T"] + MB["dn_T"]) / r_t * 1e3
    walls = {
        "q/k/v": qkv_mean * 1.05,  # the mixed split: v2's 1.25 wall/mean came from the int4-only CTAs
        "norm": t["norm"][0],
        "attention": attn_short_us if ctx == 5 else t["attention"][0],
        "o": t["o"][0],
        "MLP": mlp_mean * t["MLP"][0] / t["MLP"][1],
        "head": t["head"][0],
    }
    return walls, sum(walls.values())


def main():
    print("4E v3 on the sealed 4B, estimated from the v2 card table (us a token)")
    print("  bytes a token, MB: " + ", ".join(f"{k} {v:.1f}" for k, v in MB.items()))
    t = V2[5]
    r_o = MB["o_Q"] / (t["o"][1] - LAYERS * 1.0) * 1e3
    r_h = MB["head"] / t["head"][1] * 1e3
    print(f"  v2 int4 rates on the mean CTA's work: o {r_o:.0f} GB/s (ctx staging 1 us a layer out), head {r_h:.0f}")
    for f_qkv, f_mlp in ((2.0, 5.0), (3.0, 7.0), (4.0, 9.0)):
        r, a, b = v2_rates(5, f_qkv, f_mlp, 560.0)
        print(f"  v2 Tetra rows, fixed {f_qkv:.0f} / {f_mlp:.0f} us a layer: {r:.0f} GB/s "
              f"(q/k {a:.0f} us, gate/up + Tetra down {b:.0f} us)")
    cases = {
        "low": dict(r_t=330.0, f_qkv=4.0, f_mlp=9.0, r_q=530.0, attn_short_us=450.0),
        "middle": dict(r_t=480.0, f_qkv=3.0, f_mlp=7.0, r_q=560.0, attn_short_us=300.0),
        "high": dict(r_t=650.0, f_qkv=2.0, f_mlp=5.0, r_q=590.0, attn_short_us=200.0),
    }
    for ctx in (5, 8192):
        print(f"\n  context {ctx}: v2 token {sum(w for w, _ in V2[ctx].values()):.0f} us")
        for name, c in cases.items():
            walls, tok = token(ctx, **c)
            detail = ", ".join(f"{k} {v:.0f}" for k, v in walls.items())
            print(f"    {name:<7} Tetra rows {c['r_t']:.0f} GB/s: {tok:6.0f} us = {1e6 / tok:5.1f} tok/s ({detail})")
    # the Tetra rate 300 tok/s needs, middle fixed costs and attention
    for r in range(300, 1001, 10):
        _, tok = token(5, float(r), 3.0, 7.0, 560.0, 300.0)
        if tok <= 1e6 / 300:
            print(f"\n  300 tok/s at 5 tokens needs the Tetra rows at {r} GB/s or more on the mean CTA's work, with the "
                  "middle case's fixed work, int4 rate and attention (computed from the estimates above)")
            break
    else:
        print("\n  300 tok/s is out of reach of the Tetra rows alone under the middle case")


if __name__ == "__main__":
    main()
