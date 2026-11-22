# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "numpy",
#   "safetensors",
#   "tokenizers",
#   "pyarrow",
#   "transformers @ git+https://github.com/huggingface/transformers@046bdb8cc2",
# ]
# ///
"""Dequantize one layer of Inkling 2 Mini's native experts to float32.

`native_experts(layer) -> (w13 [128, 2304, 2304], w2 [128, 2304, 1152])`, float32, [out, in] per expert:
`w13[e]` is what `Inkling2LoopExperts` uses as `gate_up_proj[e]` (rows 0..1151 gate, 1152..2303 up) and `w2[e]`
is `down_proj[e]`.

The path is the library's own: transformers 046bdb8cc2, `conversion_mapping.py` maps
`mlp.experts.w13_weight{,_scale,_amax,_outer_scale_denominator}` to `gate_up_proj` through
`Inkling2LoopDequantizeExperts` then `Interleave(dim=1)` (the checkpoint interleaves gate and up rows), and the w2
quartet to `down_proj` through `Inkling2LoopDequantizeExperts` alone. We call those two ops directly with a float32
target dtype, on one layer read from the safetensors file by mmap.

Deviation from the library's default load: the library dequantizes into the model dtype, bfloat16 for this
checkpoint, which rounds every expert weight to bf16. Here the product e2m1 x e4m3 x (amax / denominator) (NVFP4) or
e4m3 x 2^(s-127) (MXFP8) is kept in float32, unrounded.

`own_dequant(layer)` is an independent reimplementation from the format description (numpy, no transformers),
used as a cross-check: `uv run nvfp4.py check <layer> [<layer> ...]`.
`uv run nvfp4.py g1` compares against the BF16 GGUF reference arrays in D/eval/ref_bf16/ if present.
"""

import _threads  # noqa: F401  (sets BLAS threads before numpy/torch)

import json
import os
import sys

import numpy as np

from _threads import NATIVE_FILE

PREFIX = "model.llm.layers.{layer}.mlp.experts."


class _DtypeShim:
    """`Inkling2LoopDequantizeExperts` reads the target dtype from `model.get_parameter(name)`."""

    def __init__(self, dtype):
        import torch

        self._p = torch.empty(0, dtype=dtype)

    def get_parameter(self, name):
        return self._p


def _load_quartet(f, layer, mat):
    base = PREFIX.format(layer=layer) + f"{mat}_weight"
    names = [base, base + "_scale", base + "_amax", base + "_outer_scale_denominator"]
    keys = set(f.keys())
    return names, {n: f.get_tensor(n) for n in names if n in keys}


def native_experts(layer: int, path: str = NATIVE_FILE, dtype=None):
    """The library's dequantization of layer `layer`'s experts, float32 by default."""
    import torch
    from safetensors import safe_open
    from transformers.core_model_loading import Inkling2LoopDequantizeExperts, Interleave

    dtype = dtype or torch.float32
    shim = _DtypeShim(dtype)
    out = []
    with safe_open(path, framework="pt") as f:
        for mat in ("w13", "w2"):
            names, tensors = _load_quartet(f, layer, mat)
            if names[0] not in tensors:
                raise KeyError(f"{names[0]} not in {path}")
            res = Inkling2LoopDequantizeExperts().convert(
                tensors, source_patterns=names, target_patterns=["t"], model=shim, full_layer_name="t"
            )["t"]
            del tensors
            if mat == "w13":
                res = Interleave(dim=1).convert({"t": res}, source_patterns=["t"], target_patterns=["t"])["t"]
            out.append(res.contiguous())
    w13, w2 = out
    assert tuple(w13.shape) == (128, 2304, 2304), w13.shape
    assert tuple(w2.shape) == (128, 2304, 1152), w2.shape
    return w13, w2


# ---------------------------------------------------------------- independent cross-check (numpy only)

_E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], np.float32)


def _e4m3_to_f32(u8: np.ndarray) -> np.ndarray:
    """OCP float8 e4m3fn: bias 7, no infinities, 0x7f/0xff NaN."""
    u = u8.astype(np.uint32)
    s = np.where(u & 0x80, -1.0, 1.0).astype(np.float32)
    e = ((u >> 3) & 0xF).astype(np.int32)
    m = (u & 0x7).astype(np.float32)
    val = np.where(e == 0, m / 8.0 * 2.0**-6, (1.0 + m / 8.0) * np.exp2(e.astype(np.float32) - 7.0))
    val = np.where((u & 0x7F) == 0x7F, np.nan, val)
    return (s * val).astype(np.float32)


def own_dequant(layer: int, experts=None, path: str = NATIVE_FILE):
    """Dequantize from the raw bytes, by the format description, for `experts` (default all)."""
    from safetensors import safe_open

    experts = list(range(128)) if experts is None else list(experts)
    res = []
    # numpy cannot hold float8; read through the pt framework and view the bytes as uint8
    import torch

    with safe_open(path, framework="pt") as f:
        for mat in ("w13", "w2"):
            names, t = _load_quartet(f, layer, mat)
            w = t[names[0]]
            sc = t[names[1]]
            wb = w.view(torch.uint8).numpy()[experts]
            sb = sc.view(torch.uint8).numpy()[experts]
            if names[2] in t:  # NVFP4
                amax = t[names[2]].numpy()[experts].astype(np.float32)
                den = t[names[3]].numpy().astype(np.float32)
                lo = _E2M1[wb & 0xF]
                hi = _E2M1[wb >> 4]
                vals = np.stack([lo, hi], -1).reshape(wb.shape[0], wb.shape[1], -1)  # even element = low nibble
                scale = _e4m3_to_f32(sb)
                vals = vals.reshape(*scale.shape, 16) * scale[..., None]
                vals = vals.reshape(wb.shape[0], wb.shape[1], -1) * (amax / den)[:, None, None]
            else:  # MXFP8
                vals = _e4m3_to_f32(wb)
                scale = np.exp2(sb.astype(np.float32) - 127.0)
                vals = (vals.reshape(*scale.shape, 32) * scale[..., None]).reshape(wb.shape)
            if mat == "w13":  # de-interleave rows: checkpoint row 2i = gate i, 2i+1 = up i
                vals = np.concatenate([vals[:, 0::2], vals[:, 1::2]], axis=1)
            res.append(vals.astype(np.float32))
    return res[0], res[1]


def _check(layers):
    report = {}
    for L in layers:
        w13, w2 = native_experts(L)
        ex = [0, 1, 63, 127]
        o13, o2 = own_dequant(L, ex)
        d13 = float(np.abs(w13[ex].numpy() - o13).max())
        d2 = float(np.abs(w2[ex].numpy() - o2).max())
        report[L] = {
            "max_abs_diff_w13": d13,
            "max_abs_diff_w2": d2,
            "absmax_w13": float(w13.abs().max()),
            "absmax_w2": float(w2.abs().max()),
            "finite": bool(np.isfinite(w13.numpy()).all() and np.isfinite(w2.numpy()).all()),
            "experts_checked": ex,
        }
        print(f"L{L:02d}", json.dumps(report[L]), flush=True)
    return report


def _bf16_round(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even float32 -> bfloat16 -> float32."""
    u = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return r.astype(np.uint32).view(np.float32)


def g1(ref_dir: str):
    """Compare against reference arrays written by the GGUF agent. Reads its README for names and orientation."""
    import glob
    import re

    files = sorted(glob.glob(os.path.join(ref_dir, "L*_e*_w*.npy")) + glob.glob(os.path.join(ref_dir, "extra", "L*_e*_w*.npy")))
    by_layer = {}
    for p in files:
        L, e, m = re.match(r"L(\d+)_e(\d+)_(w13|w2)\.npy", os.path.basename(p)).groups()
        by_layer.setdefault(int(L), []).append((int(e), m, p))
    rows = []
    for L, items in sorted(by_layer.items()):
        w13, w2 = native_experts(L)
        mats = {"w13": w13.numpy(), "w2": w2.numpy()}
        for e, m, p in items:
            ref = np.load(p).astype(np.float32)
            mine = mats[m][e]
            assert ref.shape == mine.shape, (p, ref.shape, mine.shape)
            r = _bf16_round(mine)
            exact = float((r == ref).mean())
            ulp = np.abs(ref).view(np.uint32) & 0x7F800000  # exponent of ref -> 1 bf16 ulp = 2^(exp-7)
            ulp = np.maximum(ulp.view(np.float32) * 2.0**-7, 1e-45)
            row = {
                "layer": L, "expert": e, "matrix": m,
                "bf16(mine)==ref fraction": exact,
                "max |bf16(mine)-ref| in bf16 ulps": float((np.abs(r - ref) / ulp).max()),
                "max |mine-ref| / |ref|max": float(np.abs(mine - ref).max() / np.abs(ref).max()),
                "corr": float(np.corrcoef(mine.ravel(), ref.ravel())[0, 1]),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
        del w13, w2, mats
    out = os.path.join(_threads.DATA, "eval", "g1.json")
    json.dump(rows, open(out, "w"), indent=2)
    return rows


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "check":
        _check([int(a) for a in sys.argv[2:]] or [0, 23])
    elif len(sys.argv) >= 2 and sys.argv[1] == "g1":
        g1(sys.argv[2] if len(sys.argv) > 2 else os.path.join(_threads.DATA, "eval", "ref_bf16"))
    else:
        print(__doc__)
