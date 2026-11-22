# /// script
# requires-python = ">=3.11"
# dependencies = ["gguf @ git+https://github.com/ggml-org/llama.cpp@690b87d881#subdirectory=gguf-py", "numpy"]
# ///
"""Self-test: uv run ops/inkling/ggufx_validate.py <D>/eval/ref_bf16 IQ1_S=<D>/gguf/Inkling-2-Mini-UD-IQ1_S.gguf ...
Pass = for every line, corr well above 0.8 and every control (transposed, wrong expert, wrong layer) under 0.01 in
absolute value; rel_err falls as the tensor type gains bits. Expected output: <D>/eval/gguf_validate.txt.

Orientation, expert order, layer index and relative error of GGUF rungs against the BF16 reference experts.

Usage: uv run ggufx_validate.py <ref_dir> NAME=path.gguf [...]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ggufx

CASES = [(0, 0, 127), (0, 127, 0), (0, 1, 0), (10, 0, 127), (10, 127, 0), (10, 1, 0), (23, 0, 127), (23, 127, 0)]

def corr(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    a = a - a.mean(); b = b - b.mean()
    return a @ b / np.sqrt((a @ a) * (b @ b))

def rel(a, r):
    return float(np.linalg.norm(a - r) / np.linalg.norm(r))

def ref(d, L, e, suf):
    for sub in ("", "extra"):
        p = os.path.join(d, sub, f"L{L:02d}_e{e:03d}_{suf}.npy")
        if os.path.exists(p): return np.load(p)

if __name__ == "__main__":
    d = sys.argv[1]
    for arg in sys.argv[2:]:
        name, p = arg.split("=", 1)
        for kind, suf in (("ffn_gate_up_exps", "w13"), ("ffn_down_exps", "w2")):
            for L, e, other in CASES:
                r = ref(d, L, e, suf)
                tn = f"blk.{L}.{kind}.weight"
                try:
                    w = ggufx._load(p, tn, e, e + 1)[0]
                    w_ns = ggufx._load(p, tn, e, e + 1, apply_scale=False)[0]
                    wo = ggufx._load(p, tn, other, other + 1)[0]
                except ggufx.UnsupportedType as ex:
                    print(f"{name} L{L} e{e} {suf}: UNSUPPORTED {ex}"); break
                La = L + 1 if L < 23 else L - 1
                adj = ggufx._load(p, f"blk.{La}.{kind}.weight", e, e + 1)[0]
                tr = f"{corr(r, w.reshape(w.shape[1], w.shape[0]).T):.4f}"  # same bytes read with the axes swapped
                typ = ggufx.GGML[ggufx._open(p)[1][tn]["type"]][0]
                print(f"{name} L{L} e{e} {suf} {typ}: corr={corr(r, w):.4f} transposed={tr} "
                      f"wrong_expert({other})={corr(r, wo):.4f} wrong_layer({La})={corr(r, adj):.4f} "
                      f"rel_err={rel(w, r):.4f} rel_err_noscale={rel(w_ns, r):.4f}", flush=True)
