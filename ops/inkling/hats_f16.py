# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Stores a layer's dense reconstructions as float16 beside (or in place of) the float32 files.

    uv run hats_f16.py <layer_dir> [--out DIR] [--drop-f32]

`executor.py` reads `w13_hat.f16` / `w2_hat.f16` when the `.f32` file is absent. The float16 file is not the decoded
value bit for bit: it is its nearest float16, a relative step of at most 2^-11 a weight. The codes in `codes/` stay
the exact object. Prints the largest relative rounding error and the relative squared error of the rounding, a layer.
`--out` writes into another directory and never touches the source; `--drop-f32` removes the source after the float16
file is synced and read back.
"""

import argparse
import json
import os

import numpy as np

SHAPES = {"w13_hat": (128, 2304, 2304), "w2_hat": (128, 2304, 1152)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layer_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--drop-f32", action="store_true")
    a = ap.parse_args()
    out = a.out or a.layer_dir
    os.makedirs(out, exist_ok=True)
    rep = {}
    for name, shape in SHAPES.items():
        src = os.path.join(a.layer_dir, name + ".f32")
        dst = os.path.join(out, name + ".f16")
        x = np.memmap(src, dtype=np.float32, mode="r", shape=shape)
        num = den = 0.0
        worst = 0.0
        with open(dst + ".tmp", "wb") as f:
            for e in range(shape[0]):
                v = np.asarray(x[e])
                h = v.astype(np.float16)
                assert np.isfinite(h).all(), f"{name} expert {e}: not finite in float16"
                d = h.astype(np.float32) - v
                num += float((d.astype(np.float64) ** 2).sum())
                den += float((v.astype(np.float64) ** 2).sum())
                nz = v != 0
                if nz.any():
                    worst = max(worst, float(np.abs(d[nz] / v[nz]).max()))
                f.write(h.tobytes())
            f.flush()
            os.fsync(f.fileno())
        os.rename(dst + ".tmp", dst)
        back = np.memmap(dst, dtype=np.float16, mode="r", shape=shape)
        assert np.array_equal(np.asarray(back[shape[0] - 1]), np.asarray(x[shape[0] - 1]).astype(np.float16))
        rep[name] = {"rel_sq_err": num / den, "max_rel_err": worst}
        del x, back
        if a.drop_f32:
            assert a.out is None, "--drop-f32 only in place"
            os.remove(src)
    print(json.dumps({"layer_dir": a.layer_dir, "out": out, "dropped_f32": a.drop_f32, **rep}))


if __name__ == "__main__":
    main()
