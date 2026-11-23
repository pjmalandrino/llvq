# /// script
# requires-python = ">=3.11"
# dependencies = ["requests", "numpy"]
# ///
"""Fetch single experts of the ggml-org BF16 GGUF by HTTP range requests and save them as float32 .npy.

GGUF stores ne = [in, out, expert] with ne[0] fastest, so a C-order numpy view is [expert, out, in].
Usage: uv run bf16_ref_fetch.py <header.json> <out_dir> <layer:expert> [...]
"""
import json, os, sys
import numpy as np, requests

def token():
    p = os.path.expanduser("~/.cache/huggingface/token")
    return open(p).read().strip() if os.path.exists(p) else None

def fetch(url, start, n):
    h = {"Range": f"bytes={start}-{start + n - 1}"}
    t = token()
    if t: h["Authorization"] = f"Bearer {t}"
    r = requests.get(url, headers=h, timeout=300)
    assert r.status_code == 206 and len(r.content) == n, (r.status_code, len(r.content), n)
    return r.content

def main():
    hdr = json.load(open(sys.argv[1])); out = sys.argv[2]
    tens = {t["name"]: t for t in hdr["tensors"]}
    for spec in sys.argv[3:]:
        L, e = map(int, spec.split(":"))
        for kind, suffix in (("ffn_gate_up_exps", "w13"), ("ffn_down_exps", "w2")):
            t = tens[f"blk.{L}.{kind}.weight"]
            assert t["type"] == 30, t  # GGML_TYPE_BF16
            n_in, n_out, n_exp = t["ne"]
            assert n_exp == 128
            per = n_in * n_out * 2
            raw = fetch(hdr["url"], hdr["data_start"] + t["offset"] + e * per, per)
            u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
            w = u.view(np.float32).reshape(n_out, n_in)
            path = os.path.join(out, f"L{L:02d}_e{e:03d}_{suffix}.npy")
            np.save(path, w)
            print(path, w.shape, float(np.abs(w).mean()))

if __name__ == "__main__":
    main()
