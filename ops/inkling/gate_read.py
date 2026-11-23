"""Reads the quality gate of `proofs/preregistration-inkling2-mini-quality-gate-2026-11-21.md` on one layer set.

    python3 -I gate_read.py <set, e.g. 6-17> [--tetra NAME] [--data D] [--bits 2.0371]

Inputs: `D/eval/arm_<name>_<set>.json` for the lattice arm (`tetra` unless `--tetra` names another arm) and for the
four rungs, and `D/eval/gguf_bits.json` for the rungs' expert bits on the layers of the set. Standard library only.

The rule, as preregistered, on point estimates:
  PASS         KL_T <= the ladder's KL at b_T, linear in log KL against bits between the two rungs that bracket b_T;
               under the lowest rung, KL_T <= that rung's KL.
  FAIL         a rung with as many bits or fewer has a KL <= KL_T; or the student's top-1 agreement is under 90 %.
  NOT SETTLED  neither.
Rungs with the same bits and the same KL on the set are one rung (two files can carry the same tensors there).
Beside the rule, and outside it: the paired KL difference window by window against each rung.
"""

import argparse
import json
import math
import os
import sys

RUNGS = [("IQ1_S", "iq1s"), ("IQ1_M", "iq1m"), ("IQ2_XXS", "iq2xxs"), ("Q2_K_XL", "q2kxl")]


def layers_of(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def set_bits(per_layer, layers):
    w = sum(per_layer[str(layer)]["weights"] for layer in layers)
    return 8.0 * sum(per_layer[str(layer)]["bytes"] for layer in layers) / w


def ladder_at(rungs, b):
    """log-linear interpolation of KL at b bits; None above the highest rung, the lowest rung's KL under it."""
    rs = sorted(rungs, key=lambda r: r["bits"])
    if b <= rs[0]["bits"]:
        return rs[0]["kl"], f"under the lowest rung ({rs[0]['name']})"
    for lo, hi in zip(rs, rs[1:]):
        if lo["bits"] <= b <= hi["bits"]:
            if hi["bits"] == lo["bits"]:
                return min(lo["kl"], hi["kl"]), f"at {lo['name']}"
            t = (b - lo["bits"]) / (hi["bits"] - lo["bits"])
            return math.exp((1 - t) * math.log(lo["kl"]) + t * math.log(hi["kl"])), f"between {lo['name']} and {hi['name']}"
    return None, "above the highest rung"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("--tetra", default="tetra")
    ap.add_argument("--data", default=os.path.expanduser("~/llvq-inkling-mini"))
    ap.add_argument("--bits", type=float, default=2.037085262345679)  # the archive figure, the criterion's
    a = ap.parse_args()
    ev = os.path.join(a.data, "eval")
    layers = layers_of(a.set)
    bits = json.load(open(os.path.join(ev, "gguf_bits.json")))

    def arm(name):
        p = os.path.join(ev, f"arm_{name}_{a.set}.json")
        return json.load(open(p)) if os.path.exists(p) else None

    t = arm(a.tetra)
    if t is None:
        sys.exit(f"no lattice arm arm_{a.tetra}_{a.set}.json")
    rungs = []
    for gg, short in RUNGS:
        r = arm(short)
        if r is None:
            print(f"missing rung {gg}")
            continue
        rungs.append({"name": gg, "bits": set_bits(bits[gg]["per_layer"], layers), "kl": r["kl_mean"], "r": r})

    def row(name, b, r):
        return (f"{name:<10} {b:6.3f}  KL {r['kl_mean']:.5f} [{r['kl_ci95'][0]:.5f}, {r['kl_ci95'][1]:.5f}]  "
                f"top1 {100 * r['top1_agree']:6.2f} %  dNLL {r['nll_diff_mean']:+.5f} "
                f"[{r['nll_diff_ci95'][0]:+.5f}, {r['nll_diff_ci95'][1]:+.5f}]  ppl x{math.exp(r['nll_diff_mean']):.4f}")

    print(f"set {{{a.set}}}, {len(layers)} layers, {t['windows']} windows of {t['ctx']}, fingerprint {t['tokens_fingerprint']}")
    print(row("lattice", a.bits, t))
    for r in rungs:
        assert r["r"]["tokens_fingerprint"] == t["tokens_fingerprint"], "arms on different tokens"
        print(row(r["name"], r["bits"], r["r"]))

    kl_t, top1 = t["kl_mean"], t["top1_agree"]
    ref, where = ladder_at(rungs, a.bits)
    passed = ref is not None and kl_t <= ref
    beaten_by = [r["name"] for r in rungs if r["bits"] <= a.bits and r["kl"] <= kl_t]
    if top1 < 0.90:
        verdict = "FAIL (top-1 floor)"
    elif beaten_by:
        verdict = "FAIL (" + ", ".join(beaten_by) + " does as well with as many bits or fewer)"
    elif passed:
        verdict = "PASS"
    else:
        verdict = "NOT SETTLED"
    refs = "none" if ref is None else f"{ref:.5f}"
    print(f"ladder at {a.bits:.4f} bits: {refs} ({where}); lattice KL {kl_t:.5f}; top-1 {100 * top1:.2f} %")
    print(f"GATE on {{{a.set}}}: {verdict}")

    print("outside the rule, paired by window (lattice KL minus rung KL; negative = lattice better):")
    kt = [w["kl_mean"] for w in t["per_window"]]
    for r in rungs:
        kr = [w["kl_mean"] for w in r["r"]["per_window"]]
        d = [x - y for x, y in zip(kt, kr)]
        wins = sum(1 for x in d if x < 0)
        print(f"  vs {r['name']:<8} mean {sum(d) / len(d):+.5f}, lattice better on {wins}/{len(d)} windows, "
              f"ratio of means {sum(kt) / sum(kr):.3f}")


if __name__ == "__main__":
    main()
