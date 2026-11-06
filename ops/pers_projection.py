"""Route P and 4E projected from step L1's card readings: rule H of the L1 prereg.

    python3 -I ops/pers_projection.py                       the prereg's predicted inputs (a dry run)
    python3 -I ops/pers_projection.py --bpa 560 --bar 1.4 --launch 2.6 --coop 0.6 --drain 2.0

Inputs, all from the L1 job (*measured* there):
  --bpa     B_pa, GB/s: tpersall's stream rate, the f32 family (bench.txt, "B_pa = ...")
  --bar     µs a grid barrier at 142 x 768 (bench.txt, microbenchmarks)
  --launch  µs a candle-side launch in situ: process 2b's "empty x1" per-boundary cost
  --coop    µs a cooperative launch costs above a normal one (bench.txt)
  --drain   µs of drain and ramp at a phase end: process 2b's derived class

Outputs are *estimated* (the weakest label of their inputs): route P, step B complete, and 4E at a short
prompt on the sealed 4B. Each is a range: the persistent stream rate B_T runs from 0.85 x B_pa (24 warps a SM
instead of today's 48, U1) to B_pa, and the exposed wait from the barrier alone to 1.5 times it (arrival skew,
U2). The model is persistent.md §4's (`calc/persistent/route_p_model.py`), its terms named in the output.

The operator's target (2026-11-04): 300 tok/s or more on the 4B after a short prompt, the bar set by
llama.cpp IQ2_XXS on the same card, 312.85 tok/s (*measured*, job m4-iq2-cuda-debit, 2026-08-30,
docs/data/jobs.csv). Stated in tok/s only: no ratio is formed across stacks (CLAUDE.md rule 5).
"""

import argparse

W_T, W_Q, W_H = 0.7487, 0.4094, 0.2188  # GB a token on the sealed 4B (computed, row formulas)
W_ALL = W_T + W_Q + W_H
BQ_CAP, BH_CAP = 650.0, 650.0  # GB/s, the int4 rows' rate, capped near cuBLAS's 671 measured on this card
TARGET, IQ2 = 300.0, 312.85


def route_p(bt, a, c, h, drain, prolog_ms=0.15, fixed_ms=0.09):
    """persistent.md §4: 360 candle launches, 36 cooperative launches, 96 dependent waits, 36 drains."""
    bq, bh = min(bt, BQ_CAP), min(bt, BH_CAP)
    stream = (W_T / bt + W_Q / bq + W_H / bh) * 1e3
    fixed = (360 * a + 36 * c + 96 * h + 36 * drain) / 1e3 + prolog_ms + fixed_ms
    return stream + fixed, stream, fixed


def four_e(bt, h, drain, attn_ms=0.20, fixed_ms=0.10):
    """4E: one launch a token, 97 grid waits, 36 attention phases at a short prompt (attn_ms in all),
    a drain per layer, embedding, head and argmax in fixed_ms (estimated)."""
    stream = W_ALL / min(bt, BQ_CAP) * 1e3
    fixed = (97 * h + 36 * drain) / 1e3 + attn_ms + fixed_ms
    return stream + fixed, stream, fixed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bpa", type=float, default=490.0)
    p.add_argument("--bar", type=float, default=1.3)
    p.add_argument("--launch", type=float, default=2.5)
    p.add_argument("--coop", type=float, default=0.8)
    p.add_argument("--drain", type=float, default=1.5)
    a = p.parse_args()
    c = a.launch + a.coop
    print(f"inputs: B_pa {a.bpa:.0f} GB/s, barrier {a.bar:.2f} us, launch {a.launch:.2f} us, "
          f"cooperative +{a.coop:.2f} us, drain {a.drain:.2f} us")
    rows = []
    for label, bt, h in (("upper", a.bpa, a.bar), ("lower", 0.85 * a.bpa, 1.5 * a.bar)):
        t, s, f = route_p(bt, a.launch, c, h, a.drain)
        t4, s4, f4 = four_e(bt, h, a.drain)
        rows.append((label, 1e3 / t, 1e3 / t4))
        print(f"  {label}: B_T {bt:.0f} GB/s, h {h:.2f} us")
        print(f"    route P  {t:.3f} ms = stream {s:.3f} + fixed {f:.3f} -> {1e3 / t:.0f} tok/s")
        print(f"    4E       {t4:.3f} ms = stream {s4:.3f} + fixed {f4:.3f} -> {1e3 / t4:.0f} tok/s")
    (_, p_hi, e_hi), (_, p_lo, e_lo) = rows
    print(f"route P [{p_lo:.0f}, {p_hi:.0f}] tok/s, 4E [{e_lo:.0f}, {e_hi:.0f}] tok/s (estimated); "
          f"target {TARGET:.0f}, IQ2_XXS on this card {IQ2:.2f} (measured)")
    if p_hi < 250:
        print("rule H: route P's upper end is under 250: route P stands only as a step toward 4E")
    elif p_lo >= 250:
        print("rule H: route P's whole range clears 250")
    else:
        print("rule H: route P's range straddles 250")
    if e_hi < TARGET:
        print("rule H: 4E's upper end is under 300: the persistent path does not reach the target on these "
              "inputs; the binding term is the larger of stream and fixed above")
    elif e_lo >= TARGET:
        print("rule H: 4E's whole range clears 300")
    else:
        print("rule H: 4E's range straddles 300")


if __name__ == "__main__":
    main()
