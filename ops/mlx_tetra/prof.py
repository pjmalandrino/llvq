# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx-lm==0.28.3", "numpy", "safetensors", "tokenizers", "pyarrow", "llvq-tetra"]
# [tool.uv.sources]
# llvq-tetra = { path = "../../llvq-tetra" }
# ///
"""Where a decode token goes on the MLX path, piece by piece, off protocol.

Each piece runs N times in a dependent chain (each call's input is cut from the
previous call's output), so the GPU cannot overlap the calls the way it can
overlap independent ones. One `mx.eval` closes the chain; the time a call is the
chain's time over N, median of rounds.

    uv run ops/mlx_tetra/prof.py <packed_dir> [n=200] [rounds=5]
"""
import statistics
import sys
import time

import mlx.core as mx

from llvq_mlx import DECODE, build

N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 5


def chain(f, x0, cut):
    def once():
        x = x0
        for _ in range(N):
            x = cut(f(x))
        mx.eval(x)
    once()
    ts = []
    for _ in range(ROUNDS):
        t = time.perf_counter()
        once()
        ts.append((time.perf_counter() - t) * 1e3 / N)
    return statistics.median(ts), min(ts), max(ts)


def main():
    model, pm, _ = build(sys.argv[1])
    L = model.model.layers
    attn, mlp = L[0].self_attn, L[0].mlp
    x2560 = mx.random.normal((1, 1, 2560)).astype(mx.float16)
    x9728 = mx.random.normal((1, 1, 9728)).astype(mx.float16)
    mx.eval(x2560, x9728)

    def kernel_only(lin):
        def f(xr):
            s = lin.shared
            return DECODE(inputs=[lin._words, s.rows, s.prefixes, s.branches, s.suffixes, lin._gscale,
                                  s.invnorm, lin._rscale, lin._tail, xr],
                          template=[("NB", lin.nb), ("TW", lin.tw), ("STRIDE", lin.stride)],
                          grid=(lin.d_out * 32, 1, 1), threadgroup=(256, 1, 1),
                          output_shapes=[(1, lin.d_out)], output_dtypes=[mx.float16])[0]
        return f

    rows = []
    for name, lin, x in [("q", attn.q_proj, x2560), ("k", attn.k_proj, x2560), ("gate", mlp.gate_proj, x2560),
                         ("up", mlp.up_proj, x2560), ("down", L[0].mlp.down_proj, x9728)]:
        if type(lin).__name__ != "TetraLinear":
            continue
        d_in = lin.d_in
        xr0 = mx.random.normal((1, d_in)).astype(mx.float32)
        mx.eval(xr0)
        cut = lambda y, d_in=d_in: mx.tile(y.astype(mx.float32), (1, -(-d_in // y.shape[-1])))[:, :d_in]
        rows.append((f"tetra.{name} kernel", chain(kernel_only(lin), xr0, cut), lin.device_bytes()))
        rot = lin.rot
        def rot_f(x, rot=rot):
            rot._last = None
            return rot(x)
        rows.append((f"rotation {d_in}", chain(rot_f, x.reshape(1, d_in), lambda y: y.astype(mx.float16)), 0))
    v = attn.v_proj
    rows.append(("int4 v (mlx qmm)", chain(lambda x: v(x), x2560,
                                            lambda y: mx.tile(y, (1, 1, 3))[..., :2560]), 0))
    o = attn.o_proj
    x4096 = mx.random.normal((1, 1, 4096)).astype(mx.float16)
    rows.append(("int4 o (mlx qmm)", chain(lambda x: o(x), x4096,
                                            lambda y: mx.tile(y, (1, 1, 2))[..., :4096]), 0))
    print(f"chains of {N} dependent calls, median [min-max] of {ROUNDS} rounds, ms a call, off protocol")
    for name, (m, lo, hi), b in rows:
        gbs = f"  {b / (m * 1e-3) / 1e9:6.1f} GB/s" if b else ""
        print(f"{name:<22} {m:7.4f} [{lo:.4f}-{hi:.4f}]{gbs}")


if __name__ == "__main__":
    main()
