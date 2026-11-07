# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx-lm==0.28.3", "mlx==0.32.4", "numpy==2.5.4", "safetensors==0.8.0", "tokenizers==0.23.3",
#                 "pyarrow==26.0.1", "transformers==5.22.1", "llvq-tetra"]
# [tool.uv.sources]
# llvq-tetra = { path = "../../llvq-tetra" }
# ///
"""The MLX path's gates: its kernels against each other, and against a reference outside MLX.

    uv run ops/mlx_tetra/check.py llvq-tetra/tests/fixtures/mini     # seconds, every section
    uv run ops/mlx_tetra/check.py <packed 4B> [layers=0,12]          # the sealed 4B, sections A to C

A. The kernels against each other: the one-dispatch rotation against the same
   rotation in four MLX ops; q and k stacked in one dispatch against each alone,
   bit for bit; the fused gate-up-SiLU against the three unfused ops.
B. Every projection of the chosen layers against the reference decoder:
   `llvq_tetra.PackedModel.dequantize`, which `checkdense` holds bit for bit to
   the Rust `decode_matrix`, in the natural basis. So the rotation, its block
   permutation, the decode, the scales and the tail are checked together, by
   three paths: the one-row kernel, the 8-row kernel (8 rows and 3 rows), the
   dense prefill (20 rows). The int4 records through MLX's quantized matmul too.
C. The int4 g64 embedding and its tied head against the same reference.
D. On a model of two layers or fewer (the fixture): the whole forward pass
   against `transformers` loading the same packed directory through
   `llvq-tetra` in f32 on the CPU, by prefill (16 rows), by decode (one row a
   step) and by chunks of 4 (the rows kernel).

Not bit for bit: the MLX path runs f16 activations and other arithmetic orders.
Each comparison prints the worst relative error over its outputs, max |a - b| /
max |b|, and holds it under its bound. `mutate.py` checks that the bounds bite.
Exit 0 when every comparison is inside, 1 otherwise.
"""
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import llvq_mlx as L

BOUND_ROT = 1e-5      # f32 rotation against f32 ops: two summation orders
BOUND_KERNEL = 2e-3   # f16 activation in, f32 accumulation, f16 out, against f64
BOUND_DENSE = 4e-3    # the rotated activation and the matrix rounded to f16, then an f16 GEMM
BOUND_EMBED = 2e-3    # MLX's affine dequantization in f16 against f32
BOUND_LOGITS = 1e-2   # a whole layer in f16 against f32


def rel(a, b) -> float:
    a = np.asarray(a.astype(mx.float32) if isinstance(a, mx.array) else a, dtype=np.float64)
    b = np.asarray(b.astype(mx.float32) if isinstance(b, mx.array) else b, dtype=np.float64)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


class Tally:
    def __init__(self):
        self.bad = 0
        self.n = 0

    def hold(self, label: str, e: float, bound: float, exact: bool = False) -> None:
        ok = e == 0.0 if exact else e < bound
        self.n += 1
        self.bad += not ok
        want = "same bits" if exact else f"< {bound:.0e}"
        print(f"  {label:<58} {e:.2e} {want:<10} {'ok' if ok else 'OVER BOUND'}")


def inner(m):
    return getattr(m, "inner", m)


def rows_f16(n: int, d: int, seed: int) -> mx.array:
    return mx.array(np.random.default_rng(seed).standard_normal((n, d)).astype(np.float16))


def section_a(model, layers, t: Tally) -> None:
    print("A. kernels against each other")
    for li in layers:
        layer = model.model.layers[li]
        att = layer.self_attn
        if hasattr(att.q_proj, "group"):
            grp = att.q_proj.group
            for i in range(3):
                x = rows_f16(1, inner(att.q_proj).d_in, 100 + i).reshape(1, 1, -1)
                qf, kf = att.q_proj(x), att.k_proj(x)
                inner(att.q_proj).rot._last = None
                e = max(rel(qf, inner(att.q_proj)(x)), rel(kf, inner(att.k_proj)(x)))
                t.hold(f"layer {li} q+k stacked against each alone, row {i}", e, 0.0, exact=True)
            # Again with k's two gains scaled apart from q's: a fixture can give
            # every record the same centroids (mini does), and then a stacked
            # dispatch that read q's gains for k's rows would pass the line above.
            saved = (grp.k._gscale, grp.gscale)
            grp.k._gscale = grp.k._gscale * 1.5
            grp.gscale = mx.concatenate([grp.q._gscale, grp.k._gscale])
            x = rows_f16(1, grp.q.d_in, 103).reshape(1, 1, -1)
            grp._last = None
            kf = att.k_proj(x)
            grp.q.rot._last = None
            t.hold(f"layer {li} q+k stacked against k alone, k's gains apart", rel(kf, grp.k(x)), 0.0, exact=True)
            grp.k._gscale, grp.gscale = saved
            grp._last = None
            grp.q.rot._last = None
        seen = set()
        for name, lin in projections(layer):
            if type(lin).__name__ != "TetraLinear" or id(lin.rot) in seen:
                continue
            seen.add(id(lin.rot))
            x = rows_f16(20, lin.d_in, 200)
            lin.rot._last = None
            t.hold(f"layer {li} rotation {lin.d_in} (via {name}) against 4 MLX ops",
                   rel(lin.rot(x), lin.rot.reference(x)), BOUND_ROT)
            lin.rot._last = None
        mlp = layer.mlp
        if type(mlp).__name__ == "FusedMLP":
            for i in range(3):
                x = rows_f16(1, mlp.gate_proj.d_in, 300 + i).reshape(1, 1, -1)
                mlp.gate_proj.rot._last = None
                want = mlp.down_proj(nn.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
                mlp.gate_proj.rot._last = None
                t.hold(f"layer {li} fused gate-up-SiLU against unfused, row {i}", rel(mlp(x), want), BOUND_KERNEL)


def projections(layer):
    att, mlp = layer.self_attn, layer.mlp
    return [("q", inner(att.q_proj)), ("k", inner(att.k_proj)), ("v", att.v_proj), ("o", att.o_proj),
            ("gate", mlp.gate_proj), ("up", mlp.up_proj), ("down", mlp.down_proj)]


def section_b(model, pm, layers, t: Tally) -> None:
    print("B. projections against the reference decoder (llvq-tetra, bit for bit with Rust)")
    for li in layers:
        layer = model.model.layers[li]
        for name, lin in projections(layer):
            block = "self_attn" if name in ("q", "k", "v", "o") else "mlp"
            w = pm.dequantize(f"model.layers.{li}.{block}.{name}_proj.weight").astype(np.float64)
            x = rows_f16(20, w.shape[1], 400 + li)
            ref = np.asarray(x.astype(mx.float32), dtype=np.float64) @ w.T
            if type(lin).__name__ != "TetraLinear":
                t.hold(f"layer {li} {name:<4} int4 g128, MLX quantized matmul", rel(lin(x), ref), BOUND_KERNEL)
                continue
            lin.rot._last = None
            one = mx.concatenate([lin(x[i:i + 1]) for i in range(20)], axis=0)
            t.hold(f"layer {li} {name:<4} one-row kernel, 20 rows", rel(one, ref), BOUND_KERNEL)
            lin.rot._last = None
            t.hold(f"layer {li} {name:<4} 8-row kernel, 8 rows", rel(lin(x[:8]), ref[:8]), BOUND_KERNEL)
            lin.rot._last = None
            t.hold(f"layer {li} {name:<4} 8-row kernel, 3 rows", rel(lin(x[8:11]), ref[8:11]), BOUND_KERNEL)
            lin.rot._last = None
            t.hold(f"layer {li} {name:<4} dense prefill, 20 rows", rel(lin(x), ref), BOUND_DENSE)
            lin.rot._last = None
        att = layer.self_attn
        if hasattr(att.q_proj, "group"):
            x = rows_f16(1, inner(att.q_proj).d_in, 500 + li).reshape(1, 1, -1)
            xs = np.asarray(x.astype(mx.float32), dtype=np.float64).reshape(1, -1)
            for name, part in (("q", att.q_proj), ("k", att.k_proj)):
                w = pm.dequantize(f"model.layers.{li}.self_attn.{name}_proj.weight").astype(np.float64)
                inner(att.q_proj).rot._last = None
                part.group._last = None
                t.hold(f"layer {li} {name:<4} stacked q+k dispatch, one row", rel(part(x).reshape(1, -1), xs @ w.T),
                       BOUND_KERNEL)


def section_c(model, pm, t: Tally) -> None:
    print("C. embedding and tied head against the reference")
    emb = model.model.embed_tokens
    ref = pm.dequantize_raw("model.embed_tokens.weight").astype(np.float64)
    ids = np.linspace(0, ref.shape[0] - 1, 64).astype(np.int64)
    t.hold("embedding rows", rel(emb(mx.array(ids)), ref[ids]), BOUND_EMBED)
    x = rows_f16(4, ref.shape[1], 600)
    t.hold("tied head, 4 rows", rel(emb.as_linear(x), np.asarray(x.astype(mx.float32), np.float64) @ ref.T),
           BOUND_KERNEL)


def section_d(model, path: str, t: Tally) -> None:
    print("D. whole forward pass against transformers through llvq-tetra, f32 on the CPU")
    import torch
    from mlx_lm.models.cache import make_prompt_cache
    from transformers import AutoModelForCausalLM

    import llvq_tetra  # noqa: F401  registers the quantization method

    ref_model, info = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"], info
    vocab = ref_model.config.vocab_size
    ids = np.random.default_rng(700).integers(0, vocab, 16)
    with torch.no_grad():
        ref = ref_model(torch.tensor(ids[None])).logits[0].numpy().astype(np.float64)
    t.hold("prefill, 16 rows (dense path)", rel(model(mx.array(ids[None]))[0], ref), BOUND_LOGITS)
    cache = make_prompt_cache(model)
    steps = [model(mx.array([[int(i)]]), cache=cache)[0, -1] for i in ids]
    t.hold("decode, 16 steps of one row", rel(mx.stack(steps), ref), BOUND_LOGITS)
    cache = make_prompt_cache(model)
    chunks = [model(mx.array(ids[None, s:s + 4]), cache=cache)[0] for s in range(0, 16, 4)]
    t.hold("chunks of 4 rows (rows kernel)", rel(mx.concatenate(chunks), ref), BOUND_LOGITS)


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__.strip().splitlines()[2].strip())
    path = sys.argv[1]
    model, pm, _ = L.build(path, ragged_embedding=True)
    n_layers = len(model.model.layers)
    layers = ([int(v) for v in sys.argv[2].split(",")] if len(sys.argv) == 3
              else [0, 12] if n_layers > 12 else list(range(n_layers)))
    t = Tally()
    section_a(model, layers, t)
    section_b(model, pm, layers, t)
    section_c(model, pm, t)
    if n_layers <= 2:
        section_d(model, path, t)
    else:
        print("D. skipped: the whole forward pass in f32 on the CPU is for the fixture only")
    if t.n == 0:
        sys.exit("0 comparisons: nothing was checked")
    print(f"{t.n} comparisons, all within bounds" if t.bad == 0 else f"{t.bad} of {t.n} comparisons over bound")
    sys.exit(1 if t.bad else 0)


if __name__ == "__main__":
    main()
