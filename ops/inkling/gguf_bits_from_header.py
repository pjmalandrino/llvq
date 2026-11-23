# /// script
# requires-python = ">=3.11"
# ///
"""Expert bits per weight of GGUF rungs, from the tensor table only (no dequantization).

Bytes per tensor = n_weights / block * block_bytes (ggml type table of llama.cpp master, ggml.h), cross-checked
against the gap to the next tensor offset (alignment padding only). Expert bytes of a layer = ffn_gate_up_exps +
ffn_down_exps + their per-expert F32 `.scale` tensors when present.
Usage: uv run gguf_bits_from_header.py <out.json> <name=header.json> [...]
"""
import json, sys

# type id: (name, block size, bytes per block)
GGML = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 8: ("Q8_0", 32, 34), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
        12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 16: ("IQ2_XXS", 256, 66),
        17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18),
        21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 29: ("IQ1_M", 256, 56),
        30: ("BF16", 1, 2), 39: ("MXFP4", 32, 17), 40: ("NVFP4", 64, 36), 41: ("Q1_0", 128, 18),
        42: ("Q2_0", 64, 18), 43: ("F8_E4M3", 1, 1)}
SETS = {"6-8": range(6, 9), "6-11": range(6, 12), "6-17": range(6, 18), "0-5": range(0, 6), "18-23": range(18, 24),
        "all": range(0, 24)}

def nbytes(t):
    n = 1
    for d in t["ne"]: n *= d
    _, b, s = GGML[t["type"]]
    assert n % b == 0
    return n, n // b * s

def analyse(h):
    ts = sorted(h["tensors"], key=lambda t: t["offset"])
    for a, b in zip(ts, ts[1:]):  # cross-check sizes against offsets
        gap = b["offset"] - a["offset"]; _, nb = nbytes(a)
        assert 0 <= gap - nb < 64, (a["name"], gap, nb)
    by = {t["name"]: t for t in ts}
    layers = {}
    for L in range(24):
        types, byt, w = {}, 0, 0
        for kind in ("ffn_gate_up_exps", "ffn_down_exps"):
            t = by[f"blk.{L}.{kind}.weight"]; n, nb = nbytes(t)
            types[kind] = GGML[t["type"]][0]; byt += nb; w += n
            s = by.get(f"blk.{L}.{kind}.scale")
            if s: byt += nbytes(s)[1]; types[kind] += "+scale"
        layers[L] = {"types": types, "bytes": byt, "weights": w, "bits": byt * 8 / w}
    sets = {k: sum(layers[L]["bytes"] for L in r) * 8 / sum(layers[L]["weights"] for L in r) for k, r in SETS.items()}
    return {"per_layer": layers, "sets": sets}

if __name__ == "__main__":
    out = {}
    for arg in sys.argv[2:]:
        name, p = arg.split("=", 1); out[name] = analyse(json.load(open(p)))
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print("expert bits per weight (gate_up + down + per-expert scales, stored bytes * 8 / weights)")
    print(f"{'file':10s} " + " ".join(f"{k:>7s}" for k in SETS))
    for name, r in out.items():
        print(f"{name:10s} " + " ".join(f"{r['sets'][k]:7.4f}" for k in SETS))
    for name, r in out.items():
        print(name, " ".join(f"{L}:{v['types']['ffn_gate_up_exps'].replace('+scale','*')}/{v['types']['ffn_down_exps'].replace('+scale','*')}" for L, v in r["per_layer"].items()))
    print("(* = per-expert F32 .scale tensor present)")
