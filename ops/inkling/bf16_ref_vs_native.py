# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "ml_dtypes"]
# ///
"""Axis and gate/up check of the BF16 GGUF reference against the native checkpoint's raw codes.

Up to one positive constant per expert (global NVFP4 scale), the native value is code * block scale; the
correlation with the BF16 reference is computed per half (gate rows 0..1151, up rows 1152..2303) and
for the transposed and wrong-expert candidates. The native w13 is de-interleaved as transformers does on load. Two nibble orders are tried for NVFP4.
Usage: uv run bf16_ref_vs_native.py <native.safetensors> <ref_dir> <layer:expert> [...]
"""
import json, struct, sys, os
import numpy as np, ml_dtypes

E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)

def st_open(path):
    f = open(path, "rb"); n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    def get(name, e):
        t = h[name]; a, b = t["data_offsets"]; shp = t["shape"]
        per = int(np.prod(shp[1:])) * {"U8": 1, "F8_E4M3": 1}[t["dtype"]]
        raw = np.asarray(mm[8 + n + a + e * per: 8 + n + a + (e + 1) * per])
        return raw.reshape(shp[1:]), t["dtype"]
    return get

def native(get, L, e, kind, lo_first=True):
    w = _native(get, L, e, kind, lo_first)
    if kind == "w13":  # transformers Interleave(dim=1): stored row 2j is gate j, row 2j+1 is up j
        w = np.concatenate([w[0::2], w[1::2]], axis=0)
    return w

def _native(get, L, e, kind, lo_first=True):
    p = f"model.llm.layers.{L}.mlp.experts.{kind}_weight"
    w, dt = get(p, e); s, sdt = get(p + "_scale", e)
    if dt == "U8":  # NVFP4
        lo, hi = E2M1[w & 15], E2M1[w >> 4]
        codes = np.stack([lo, hi] if lo_first else [hi, lo], -1).reshape(w.shape[0], -1)
        sc = s.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
        return codes * np.repeat(sc, 16, axis=1)
    vals = w.view(ml_dtypes.float8_e4m3fn).astype(np.float32)  # MXFP8, e8m0 scale per 32
    return vals * np.repeat(np.exp2(s.astype(np.float32) - 127), 32, axis=1)

def corr(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    a -= a.mean(); b -= b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))

def find(ref, L, e, suf):
    for d in (ref, os.path.join(ref, "extra")):
        p = os.path.join(d, f"L{L:02d}_e{e:03d}_{suf}.npy")
        if os.path.exists(p): return np.load(p)

if __name__ == "__main__":
    get = st_open(sys.argv[1]); ref = sys.argv[2]
    for spec in sys.argv[3:]:
        L, e = map(int, spec.split(":")); other = 127 if e != 127 else 0
        for kind, suf in (("w13", "w13"), ("w2", "w2")):
            r = find(ref, L, e, suf)
            orders = (True, False) if L < 20 else (True,)
            for lo in orders:
                n = native(get, L, e, kind, lo)
                line = f"L{L} e{e} {kind} lo_first={lo}: corr={corr(r, n):.6f}"
                if kind == "w13":
                    h = 1152
                    line += f" gate~gate={corr(r[:h], n[:h]):.4f} up~up={corr(r[h:], n[h:]):.4f} gate~up={corr(r[:h], n[h:]):.4f}"
                    line += f" transposed={corr(r, n.T):.4f}"
                line += f" wrong_expert({other})={corr(r, native(get, L, other, kind, lo)):.4f}"
                # scale-free check: ratio ref/native constant per expert?
                m = (np.abs(n) > 0) & (np.abs(r) > 0)
                q = r[m] / n[m]
                line += f" ratio median={np.median(q):.6g} rel_spread={np.std(q)/abs(np.median(q)):.2e}"
                print(line, flush=True)
