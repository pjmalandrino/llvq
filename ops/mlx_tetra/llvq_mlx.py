# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx-lm==0.28.3", "numpy", "safetensors", "tokenizers", "pyarrow", "llvq-tetra"]
# [tool.uv.sources]
# llvq-tetra = { path = "../../llvq-tetra" }
# ///
"""The sealed Tetra 4B on MLX: mlx-lm's Qwen3 with our projections.

Prototype of decision 2 of `docs/plan-metal-v2.md`: MLX as the engine on the Mac.
mlx-lm brings the attention, the norms, RoPE, the cache and the int4 matmul; this
file brings what MLX does not have:

* `TetraLinear`: the 168 Tetra records, resident in the served `tetra48` layout,
  multiplied by our Metal kernels through `mx.fast.metal_kernel`. One row runs the
  `lutd` loop (`llvq_tetra48_rows.metal`), several rows the `lutdr` loop, which
  decodes each block once for up to 8 rows. Both store f16.
* the rotation `Q = (Q_odd (x) H_m) D` in MLX ops, shared by the projections that
  read one activation (q and k, gate and up).
* the int4 records (v, o, down in 12 layers) and the int4 g64 embedding as MLX's
  own quantized layers: the file's affine format, `w = scale q + bias`, nibbles low
  first, is MLX's.

Not bit for bit with the Rust engine (other kernels for everything but Tetra,
another rotation arithmetic); the gate is the tokens and the M6 band.

    uv run ops/mlx_tetra/llvq_mlx.py <packed_dir> gen [n_new=64] [prompt_tokens=0]
    uv run ops/mlx_tetra/llvq_mlx.py <packed_dir> time <n_new> <timed_rounds> [prompt_tokens=0]
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import struct
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.generate import generate_step
from mlx_lm.models import qwen3

from llvq_tetra.fused import invnorm_table
from llvq_tetra.metal import DATA, stride_u32, to_tetra48
from llvq_tetra.reader import PackedModel
from llvq_tetra.tetra import split_stream

REPO = Path(__file__).resolve().parents[2]
KERNELS = REPO / "llvq-llm" / "kernels"
PROMPT = "The capital of France is"
WIKI = Path.home() / (
    ".cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/"
    "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/test-00000-of-00001.parquet"
)
DISCARDED = 2
ROWS_MAX = 8

# ---------------------------------------------------------------------------
# The kernels: the served unit and the row loops as the header, a body each.
# ---------------------------------------------------------------------------

HEADER = (
    "#define LLVQ_TILE_BLOCKS 64u\n"
    + (KERNELS / "llvq_tetra48.metal").read_text()
    + "\n"
    + (KERNELS / "llvq_tetra48_rows.metal").read_text()
)

DECODE_BODY = """
    uint row = thread_position_in_grid.x >> 5;
    uint lane = thread_index_in_simdgroup;
    const device uint* wrow = words + row * STRIDE;
    F1rTables tab = { rows, prefixes, branches, suffixes };
    float acc = 0.0f;
    for (uint j = lane; j < uint(NB); j += 32u) {
        uint lo, hi16;
        f1r_load(wrow, j, lo, hi16);
        // tetra48_dot_lut_dev's text, inline: MLX may bind gscale as `constant`.
        F1rV3Block b = f1r_v3_fetch(lo, hi16, tab);
        uint cs[3] = { b.c1, b.c2, b.c3 };
        uint rs[3] = { b.row1, b.row2, b.row3 };
        const device float* xb = x + j * 24u;
        float a = 0.0f;
        float n2 = 0.0f;
        for (uint sec = 0u; sec < 3u; ++sec) {
            uint c = cs[sec];
            uint rw = rs[sec];
            for (uint k = 0u; k < 8u; ++k) {
                uint o = b.p + 2u * ((c >> k) & 1u);
                uint rho = (rw >> (4u * k)) & 15u;
                float v = F1R_VAL[(o << 4) | rho];
                n2 = fma(v, v, n2);
                a = fma(v, xb[TETRA48_ORDER[sec * 8u + k]], a);
            }
        }
        uint m = (uint(n2) >> 4) & (TETRA48_SHELLS - 1u);
        uint g = (hi16 >> 15) & 1u;
        acc += a * gscale[g] * invnorm[m];
    }
    tetra48_rows_epilogue_h(acc, row, lane, rscale, tail, x, y, uint(NB), uint(TW));
"""

ROWS_BODY = """
    uint row = thread_position_in_grid.x >> 5;
    uint lane = thread_index_in_simdgroup;
    uint r0 = thread_position_in_grid.y * 8u;
    uint nrows = min(8u, nr[0] - r0);
    const device float* xc = x + r0 * uint(DIN);
    device half* yc = y + r0 * uint(DOUT);
    const device uint* wrow = words + row * STRIDE;
    F1rTables tab = { rows, prefixes, branches, suffixes };
    float acc[8];
    for (uint r = 0u; r < 8u; ++r) acc[r] = 0.0f;
    for (uint j = lane; j < uint(NB); j += 32u) {
        uint lo, hi16;
        f1r_load(wrow, j, lo, hi16);
        F1rV3Block b = f1r_v3_fetch(lo, hi16, tab);
        uint cs[3] = { b.c1, b.c2, b.c3 };
        uint rs[3] = { b.row1, b.row2, b.row3 };
        float v[24];
        float n2 = 0.0f;
        for (uint sec = 0u; sec < 3u; ++sec) {
            uint c = cs[sec];
            uint rw = rs[sec];
            for (uint k = 0u; k < 8u; ++k) {
                uint o = b.p + 2u * ((c >> k) & 1u);
                uint rho = (rw >> (4u * k)) & 15u;
                float vv = F1R_VAL[(o << 4) | rho];
                v[sec * 8u + k] = vv;
                n2 = fma(vv, vv, n2);
            }
        }
        uint m = (uint(n2) >> 4) & (TETRA48_SHELLS - 1u);
        uint g = (hi16 >> 15) & 1u;
        float gs = gscale[g];
        float inv = invnorm[m];
        for (uint r = 0u; r < 8u; ++r) {
            if (r < nrows) {
                const device float* xb = xc + r * uint(DIN) + j * 24u;
                float a = 0.0f;
                for (uint i = 0u; i < 24u; ++i) a = fma(v[i], xb[TETRA48_ORDER[i]], a);
                acc[r] += a * gs * inv;
            }
        }
    }
    for (uint r = 0u; r < 8u; ++r) {
        if (r < nrows) {
            tetra48_rows_epilogue_h(acc[r], row, lane, rscale, tail, xc + r * uint(DIN),
                                    yc + r * uint(DOUT), uint(NB), uint(TW));
        }
    }
"""

INPUTS = ["words", "rows", "prefixes", "branches", "suffixes", "gscale", "invnorm", "rscale", "tail", "x"]
DECODE = mx.fast.metal_kernel(name="llvq_tetra_lutd", input_names=INPUTS, output_names=["y"],
                              source=DECODE_BODY, header=HEADER)
ROWS = mx.fast.metal_kernel(name="llvq_tetra_lutdr", input_names=INPUTS + ["nr"], output_names=["y"],
                            source=ROWS_BODY, header=HEADER)


class Shared:
    """The decoder tables and invnorm, on the device once."""

    def __init__(self):
        from safetensors.numpy import load_file

        t = load_file(DATA / "tetra-tables.safetensors")
        self.rows = mx.array(np.ascontiguousarray(t["rows"], dtype="<u4"))
        self.prefixes = mx.array(np.ascontiguousarray(t["prefixes_flat"], dtype="<u1"))
        self.branches = mx.array(np.ascontiguousarray(t["branches_u16"], dtype="<u2"))
        self.suffixes = mx.array(np.ascontiguousarray(t["suffixes_flat"], dtype="<u1"))
        self.invnorm = mx.array(invnorm_table())


class Rotation:
    """`x <- Q x` in f32: the sign flip, the scaled transform per group of m, the mix."""

    def __init__(self, signs: np.ndarray, small: np.ndarray):
        self.n = signs.shape[0]
        self.m = self.n & (-self.n)
        self.k = self.n // self.m
        self.signs = mx.array(signs.astype(np.float32))
        self.small = mx.array(small.astype(np.float32))
        self._last = None

    def __call__(self, x: mx.array) -> mx.array:
        # q and k (gate and up) hand in the same array: rotate it once.
        if self._last is not None and self._last[0] is x:
            return self._last[1]
        lead = x.shape[:-1]
        v = (x.astype(mx.float32) * self.signs).reshape(-1, self.k, self.m)
        v = mx.hadamard_transform(v, scale=1.0 / math.sqrt(self.m))
        if self.k > 1:
            v = self.small @ v
        out = v.reshape(*lead, self.n)
        self._last = (x, out)
        return out


class TetraLinear(nn.Module):
    """One Tetra record, resident compressed."""

    def __init__(self, pm: PackedModel, name: str, rot: Rotation, shared: Shared):
        super().__init__()
        d = pm.records[name]
        self.d_out, self.d_in, self.nb, self.tw = d["d_out"], d["d_in"], d["nblocks"], d["tail_cols"]
        labels, gains = split_stream(pm.tensor(f"{d['prefix']}.codes"), self.d_out * self.nb,
                                     d["index_bits"], d["gain_bits"])
        words = np.ascontiguousarray(to_tetra48(labels, gains, self.d_out, self.nb)).reshape(-1)
        if words.dtype != np.uint32:
            words = words.view("<u4")
        assert words.size == self.d_out * stride_u32(self.nb), (words.size, self.d_out, self.nb)
        self._words = mx.array(words)
        self._gscale = mx.array(pm.tensor(f"{d['prefix']}.centroids").astype(np.float32))
        self._rscale = mx.array(pm.tensor(f"{d['prefix']}.row_scales").astype(np.float32))
        tail = (pm.tensor(f"{d['prefix']}.tail").astype(np.float16).reshape(-1) if self.tw
                else np.zeros(1, dtype=np.float16))
        self._tail = mx.array(tail)
        self.stride = stride_u32(self.nb)
        self.rot, self.shared = rot, shared
        self.freeze()

    def device_bytes(self) -> int:
        return sum(a.nbytes for a in (self._words, self._gscale, self._rscale, self._tail))

    def __call__(self, x: mx.array) -> mx.array:
        lead = x.shape[:-1]
        xr = self.rot(x).reshape(-1, self.d_in)
        n = xr.shape[0]
        s = self.shared
        args = [self._words, s.rows, s.prefixes, s.branches, s.suffixes, self._gscale, s.invnorm,
                self._rscale, self._tail, xr]
        tmpl = [("NB", self.nb), ("TW", self.tw), ("STRIDE", self.stride)]
        if n == 1:
            y = DECODE(inputs=args, template=tmpl, grid=(self.d_out * 32, 1, 1), threadgroup=(256, 1, 1),
                       output_shapes=[(1, self.d_out)], output_dtypes=[mx.float16])[0]
        else:
            y = ROWS(inputs=args + [mx.array([n], dtype=mx.uint32)],
                     template=tmpl + [("DIN", self.d_in), ("DOUT", self.d_out)],
                     grid=(self.d_out * 32, -(-n // ROWS_MAX), 1), threadgroup=(256, 1, 1),
                     output_shapes=[(n, self.d_out)], output_dtypes=[mx.float16])[0]
        return y.reshape(*lead, self.d_out)


def int4_linear(pm: PackedModel, name: str) -> nn.QuantizedLinear:
    d = pm.records[name]
    q = nn.QuantizedLinear(d["d_in"], d["d_out"], bias=False, group_size=d["group"], bits=4)
    p = d["prefix"]
    q.weight = mx.array(np.ascontiguousarray(pm.tensor(f"{p}.qweight")).view("<u4"))
    q.scales = mx.array(pm.tensor(f"{p}.scales").astype(np.float16))
    q.biases = mx.array(pm.tensor(f"{p}.biases").astype(np.float16))
    q.freeze()
    return q


def build(path: str):
    pm = PackedModel(path)
    cfg = {k: v for k, v in pm.config.items() if k != "quantization_config"}
    args = qwen3.ModelArgs.from_dict(cfg)
    model = qwen3.Model(args)
    shared = Shared()
    rots = {key: Rotation(*pm.rotation(key)) for key in pm.rotations}
    tetra_bytes = 0
    for name, d in pm.records.items():
        # model.layers.<i>.<self_attn|mlp>.<proj>.weight
        parts = name.split(".")
        layer, block, proj = int(parts[2]), parts[3], parts[4]
        owner = getattr(model.model.layers[layer], block)
        if d["kind"] == "tetra":
            mod = TetraLinear(pm, name, rots[d["rotation"]], shared)
            tetra_bytes += mod.device_bytes()
        elif d["kind"] == "int4g128":
            mod = int4_linear(pm, name)
        else:
            raise ValueError(f"{name}: kind {d['kind']}")
        setattr(owner, proj, mod)
    weights = []
    for name, d in pm.raw.items():
        if d["encoding"] == "f16":
            weights.append((name, mx.array(pm.tensor(name).astype(np.float16))))
    emb = pm.raw["model.embed_tokens.weight"]
    assert emb["encoding"] == "quant" and emb["bits"] == 4, emb
    qe = nn.QuantizedEmbedding(emb["dims"][0], emb["dims"][1], group_size=emb["group"], bits=4)
    p = emb["prefix"]
    qe.weight = mx.array(np.ascontiguousarray(pm.tensor(f"{p}.qweight")).view("<u4"))
    qe.scales = mx.array(pm.tensor(f"{p}.scales").astype(np.float16))
    qe.biases = mx.array(pm.tensor(f"{p}.biases").astype(np.float16))
    model.model.embed_tokens = qe
    model.load_weights(weights, strict=False)
    model.freeze()
    mx.eval(model.parameters())
    return model, pm, tetra_bytes


def prompt_ids(path: str, n: int) -> list[int]:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(Path(path) / "tokenizer.json"))
    if n == 0:
        return tok.encode(PROMPT, add_special_tokens=False).ids
    import pyarrow.parquet as pq

    head = "\n\n".join(pq.read_table(WIKI).column("text").to_pylist())[: n * 12]
    ids = tok.encode(head, add_special_tokens=False).ids
    assert len(ids) >= n, f"{n} prompt tokens asked, {len(ids)} found"
    return ids[:n]


def sha_u32(ids) -> str:
    return hashlib.sha256(b"".join(struct.pack("<I", i) for i in ids)).hexdigest()


def one_round(model, ids, n_new):
    out, t0, t1 = [], time.perf_counter(), None
    for (tok, _), _ in zip(generate_step(mx.array(ids), model, max_tokens=n_new), range(n_new)):
        out.append(int(tok))
        if t1 is None:
            t1 = time.perf_counter()
    t2 = time.perf_counter()
    return t1 - t0, (len(out) - 1) / (t2 - t1), len(out) / (t2 - t0), out


def main():
    path, mode = sys.argv[1], sys.argv[2]
    print(f"mlx {mx.__version__} | {mx.default_device()} | packed {path}")
    t = time.perf_counter()
    model, pm, tetra_bytes = build(path)
    print(f"built in {time.perf_counter() - t:.1f} s | Tetra resident {tetra_bytes / 1e9:.3f} GB | "
          f"active {mx.get_active_memory() / 1e9:.2f} GB")
    if mode == "gen":
        n_new = int(sys.argv[3]) if len(sys.argv) > 3 else 64
        n_prompt = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        ids = prompt_ids(path, n_prompt)
        first, decode, door, out = one_round(model, ids, n_new)
        print(f"prompt {len(ids)} tokens sha256 {sha_u32(ids)}; first token {first * 1e3:.1f} ms, "
              f"decode {decode:.2f} tok/s (one round, not a measurement)")
        print(f"tokens sha256 {sha_u32(out)}")
        print("ids", out[:24])
        from tokenizers import Tokenizer

        print("text:", repr(Tokenizer.from_file(str(Path(path) / "tokenizer.json")).decode(out)[:300]))
        return
    n_new, timed = int(sys.argv[3]), int(sys.argv[4])
    n_prompt = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    ids = prompt_ids(path, n_prompt)
    print(f"prompt {len(ids)} tokens, sha256 {sha_u32(ids)}; {n_new} new; {DISCARDED} discarded + {timed} timed")
    firsts, decodes, doors, ref = [], [], [], None
    for r in range(DISCARDED + timed):
        first, decode, door, out = one_round(model, ids, n_new)
        tag = "(discarded) " if r < DISCARDED else ""
        print(f"round {r} {tag}door {door:7.2f} tok/s | decode {decode:7.2f} tok/s | first token {first * 1e3:8.1f} ms")
        if r >= DISCARDED:
            firsts.append(first), decodes.append(decode), doors.append(door)
        if ref is None:
            ref = out
        elif out != ref:
            print("WARNING: tokens differ from round 0")
    med = statistics.median
    print(f"--- medians over {timed} timed rounds [min-max], measured on this Mac")
    print(f"llvq-mlx door {med(doors):7.2f} tok/s [{min(doors):.2f}-{max(doors):.2f}] | "
          f"decode {med(decodes):7.2f} tok/s [{min(decodes):.2f}-{max(decodes):.2f}] | "
          f"{1e3 / med(decodes):.2f} ms a decode token | first token {med(firsts) * 1e3:.1f} ms "
          f"[{min(firsts) * 1e3:.1f}-{max(firsts) * 1e3:.1f}]")
    print(f"peak memory {mx.get_peak_memory() / 1e9:.2f} GB")
    print(f"tokens sha256 {sha_u32(ref)}")


if __name__ == "__main__":
    main()
