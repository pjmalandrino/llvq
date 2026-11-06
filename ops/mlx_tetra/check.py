# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx-lm==0.28.3", "mlx==0.32.4", "numpy==2.5.4", "safetensors==0.8.0", "tokenizers==0.23.3",
#                 "pyarrow==26.0.1", "transformers==5.22.1", "llvq-tetra"]
# [tool.uv.sources]
# llvq-tetra = { path = "../../llvq-tetra" }
# ///
"""The MLX path's kernels against each other, on the sealed 4B's layer 0 and 12.

* the one-dispatch rotation against the four-op reference, at 2,560 and 9,728;
* the dense prefill (f16 decode + GEMM) against the one-row kernel, row by row;
* the 8-row kernel against the one-row kernel.

Not bit for bit (other arithmetic orders, an f16 GEMM): the worst relative error
over a projection's outputs is printed and held under a bound.
"""
import sys

import mlx.core as mx
import mlx.nn as nn

import llvq_mlx as L

BOUND_ROT = 1e-5
BOUND_MV = 2e-2


def rel(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return (mx.max(mx.abs(a - b)) / mx.max(mx.abs(b))).item()


def main():
    model, pm, _ = L.build(sys.argv[1])
    bad = 0
    for li in (0, 12):
        layer = model.model.layers[li]
        att = layer.self_attn
        inner = lambda m: getattr(m, "inner", m)
        if hasattr(att.q_proj, "group"):
            for i in range(3):
                x = mx.random.normal((1, 1, inner(att.q_proj).d_in)).astype(mx.float16)
                qf, kf = att.q_proj(x), att.k_proj(x)
                inner(att.q_proj).rot._last = None
                e = max(rel(qf, inner(att.q_proj)(x)), rel(kf, inner(att.k_proj)(x)))
                ok = e == 0.0
                bad += not ok
                print(f"layer {li} stacked q+k row {i}: {e:.2e} {'ok (same bits)' if ok else 'DIFFERS'}")
        projs = [("q", inner(att.q_proj)), ("k", inner(att.k_proj)), ("gate", layer.mlp.gate_proj),
                 ("up", layer.mlp.up_proj), ("down", layer.mlp.down_proj)]
        for name, lin in projs:
            if type(lin).__name__ != "TetraLinear":
                print(f"layer {li} {name}: int4, skipped")
                continue
            x = mx.random.normal((20, lin.d_in)).astype(mx.float16)
            lin.rot._last = None
            e_rot = rel(lin.rot(x), lin.rot.reference(x))
            lin.rot._last = None
            one = mx.concatenate([lin(x[i:i + 1]) for i in range(20)], axis=0)
            e_rows = rel(lin(x[:8]), one[:8])
            lin.rot._last = None
            e_dense = rel(lin(x), one)
            ok = e_rot < BOUND_ROT and e_rows < BOUND_MV and e_dense < BOUND_MV
            bad += not ok
            print(f"layer {li} {name:<4} rotation {e_rot:.2e} | 8-row kernel {e_rows:.2e} | dense prefill {e_dense:.2e}"
                  f" {'ok' if ok else 'OVER BOUND'}")
        mlp = layer.mlp
        if type(mlp).__name__ == "FusedMLP":
            for i in range(4):
                x = mx.random.normal((1, 1, mlp.gate_proj.d_in)).astype(mx.float16)
                mlp.gate_proj.rot._last = None
                want = mlp.down_proj(nn.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
                mlp.gate_proj.rot._last = None
                e = rel(mlp(x), want)
                ok = e < BOUND_MV
                bad += not ok
                print(f"layer {li} fused MLP row {i}: {e:.2e} {'ok' if ok else 'OVER BOUND'}")
    print("all within bounds" if bad == 0 else f"{bad} projections over bound")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
