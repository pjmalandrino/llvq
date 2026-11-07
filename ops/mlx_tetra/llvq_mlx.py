# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx-lm==0.28.3", "mlx==0.32.4", "numpy==2.5.4", "safetensors==0.8.0", "tokenizers==0.23.3",
#                 "pyarrow==26.0.1", "transformers==5.22.1", "llvq-tetra"]
# [tool.uv.sources]
# llvq-tetra = { path = "../../llvq-tetra" }
# ///
"""The sealed Tetra 4B on MLX: mlx-lm's Qwen3 with our projections.

A prototype of MLX as the engine on the Mac, the operator's choice of 2026-11-06
(`docs/plan-metal-debit.md`). It reads the sealed codes from the packed directory
without re-quantizing them. mlx-lm brings the attention, the norms, RoPE, the cache and the int4 matmul; this
file brings what MLX does not have:

* `TetraLinear`: the 168 Tetra records, resident in the served `tetra48` layout,
  multiplied by our Metal kernels through `mx.fast.metal_kernel`. One row runs the
  `lutd` loop (`llvq_tetra48_rows.metal`), several rows the `lutdr` loop, which
  decodes each block once for up to 8 rows. Both store f16.
* the rotation `Q = (Q_odd (x) H_m) D` in one kernel (`ROT`), held by `check.py`
  against the same rotation in four MLX ops, and shared by the projections that
  read one activation (q and k, gate and up).
* the int4 records (v, o, down in 12 layers) and the int4 g64 embedding as MLX's
  own quantized layers: the file's affine format, `w = scale q + bias`, nibbles low
  first, is MLX's.

Activations are f16. The path is not bit-exact with the served kernels: other
kernels for everything but Tetra, another rotation arithmetic. After the 5-token
prompt its 256 greedy tokens differ from the Rust engine's: it leaves the repeated
answer at token 243, the Rust engine at another token. After 4,096 tokens of
context its 64 tokens are the Rust engine's (M15 of
`docs/mesures/metal-nuit-2026-11-06.txt`). No perplexity has been measured on it
yet, so whether it holds the M6 band (+-0.12 %) is unknown.

This file is the engine, and its `main` the measurement door: no config, the
ablation modes of `LLVQ_MLX_ABLATE` accepted, the door M15 and M16 timed. The
served door is `serve.py`, which reads `configs/mlx/qwen3-4b-tetra-e4-mlx.json`
and refuses what that file does not name.

    uv run ops/mlx_tetra/llvq_mlx.py <packed_dir> gen [n_new=64] [prompt_tokens=0]
    uv run ops/mlx_tetra/llvq_mlx.py <packed_dir> time <n_new> <timed_rounds> [prompt_tokens=0]
    uv run ops/mlx_tetra/llvq_mlx.py <packed_dir> chat [temp=0.7] [max_new=512]
"""
from __future__ import annotations

import hashlib
import json
import math
import os
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
# The WikiText-2 test split at the revision `metalrun` reads, from the Hugging
# Face cache ($HF_HUB_CACHE, else $HF_HOME/hub, else ~/.cache/huggingface/hub).
# Nothing here downloads it: `wiki_parquet` names the command when it is missing.
WIKI_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKI_FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"


def wiki_parquet() -> Path:
    hub = os.environ.get("HF_HUB_CACHE") or (
        str(Path(os.environ["HF_HOME"]) / "hub") if os.environ.get("HF_HOME") else None
    )
    hub = Path(hub) if hub else Path.home() / ".cache/huggingface/hub"
    path = hub / "datasets--Salesforce--wikitext" / "snapshots" / WIKI_REVISION / WIKI_FILE
    if not path.is_file():
        sys.exit(
            f"no WikiText-2 test split at {path}\n"
            f"It is the dataset Salesforce/wikitext, config wikitext-2-raw-v1, revision {WIKI_REVISION[:8]}.\n"
            f"Fetch it once into the Hugging Face cache:\n"
            f"  hf download Salesforce/wikitext --repo-type dataset --revision {WIKI_REVISION} "
            f"--include '{WIKI_FILE}'"
        )
    return path
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
        // The activation arrives permuted block by block (the rotation wrote
        // x[24 j + ORDER[k]] at 24 j + k), so a block's 24 values are six
        // aligned float4 loads instead of 24 gathers. Same products, same order.
        const device float4* x4 = (const device float4*)(x + j * 24u);
        float xv[24];
        for (uint q = 0u; q < 6u; ++q) {
            float4 t4 = x4[q];
            xv[4u * q] = t4.x; xv[4u * q + 1u] = t4.y; xv[4u * q + 2u] = t4.z; xv[4u * q + 3u] = t4.w;
        }
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
                a = fma(v, xv[sec * 8u + k], a);
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
                for (uint i = 0u; i < 24u; ++i) a = fma(v[i], xb[i], a);
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

# The rotation, one dispatch a call: a threadgroup a row. Each group of M
# coordinates is staged in threadgroup memory, sign-flipped, transformed and
# scaled by 1/sqrt(M), and written to the output; then each thread mixes whole
# columns j across the K groups in place. Only M floats live in threadgroup
# memory, so every width the model has fits (9,728 = 19 x 512 included).
ROT_BODY = """
    // GB groups of M staged together (GB * M floats of threadgroup memory),
    // every group's butterfly stage in parallel: 9 barriers a batch, not 9 a
    // group. The same pairs in the same order, so the same values.
    threadgroup float s[GB * M];
    uint tid = thread_position_in_threadgroup.x;
    uint nt = threads_per_threadgroup.x;
    uint r = threadgroup_position_in_grid.y;
    const device half* xr = x + r * uint(N);
    device float* yr = y + r * uint(N);
    float inv = metal::rsqrt(float(M));
    for (uint g0 = 0u; g0 < uint(K); g0 += uint(GB)) {
        uint cnt = min(uint(GB), uint(K) - g0) * uint(M);
        uint base = g0 * uint(M);
        for (uint i = tid; i < cnt; i += nt) s[i] = float(xr[base + i]) * signs[base + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint half_m = uint(M) / 2u;
        for (uint len = 1u; len < uint(M); len <<= 1) {
            for (uint p = tid; p < cnt / 2u; p += nt) {
                uint grp = p / half_m;
                uint pp = p % half_m;
                uint j = grp * uint(M) + (pp / len) * (len << 1) + (pp % len);
                float a = s[j];
                float b = s[j + len];
                s[j] = a + b;
                s[j + len] = a - b;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint i = tid; i < cnt; i += nt) yr[base + i] = s[i] * inv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (uint(K) > 1u) {
        threadgroup_barrier(mem_flags::mem_device);
        for (uint j = tid; j < uint(M); j += nt) {
            float col[K];
            for (uint t = 0u; t < uint(K); ++t) col[t] = yr[t * uint(M) + j];
            for (uint g = 0u; g < uint(K); ++g) {
                float acc = 0.0f;
                for (uint t = 0u; t < uint(K); ++t) acc = fma(small[g * uint(K) + t], col[t], acc);
                yr[g * uint(M) + j] = acc;
            }
        }
    }
    // The Tetra kernels read a block's 24 coordinates in TETRA48_ORDER: hand
    // them over in that order, the tail columns as they are.
    threadgroup_barrier(mem_flags::mem_device);
    device float* pr = yp + r * uint(N);
    for (uint q = tid; q < uint(N); q += nt) {
        uint p = q < uint(NB) * 24u ? (q / 24u) * 24u + TETRA48_ORDER[q % 24u] : q;
        pr[q] = yr[p];
    }
"""
ROT_TG_FLOATS = 7168
ROT = mx.fast.metal_kernel(name="llvq_rot_tg", input_names=["x", "signs", "small"], output_names=["y", "yp"],
                           source=ROT_BODY, header=HEADER)

TETRA48_ORDER = [0, 1, 2, 3, 4, 7, 10, 12, 6, 11, 13, 14, 16, 17, 18, 19, 5, 8, 9, 15, 20, 21, 22, 23]


def permutation(n: int) -> np.ndarray:
    """Output position q holds natural coordinate perm[q]: the Tetra block order, tail as is."""
    nb = n // 24
    perm = np.arange(n)
    for j in range(nb):
        perm[24 * j:24 * j + 24] = 24 * j + np.array(TETRA48_ORDER)
    return perm

# A Tetra record decoded to f16 in the ROTATED basis, for the prefill's GEMM:
# W'[r, 24 j + ORDER[k]] = v_k * gscale[g] * invnorm[m] * rscale[r], and the
# tail columns as stored. One thread a block; block 0's thread writes the tail.
DEQ_BODY = """
    uint id = thread_position_in_grid.x;
    if (id >= uint(DOUT) * uint(NB)) return;
    uint row = id / uint(NB);
    uint j = id % uint(NB);
    F1rTables tab = { rows, prefixes, branches, suffixes };
    uint lo, hi16;
    f1r_load(words + row * STRIDE, j, lo, hi16);
    F1rV3Block b = f1r_v3_fetch(lo, hi16, tab);
    uint cs[3] = { b.c1, b.c2, b.c3 };
    uint rs[3] = { b.row1, b.row2, b.row3 };
    float v[24];
    float n2 = 0.0f;
    for (uint sec = 0u; sec < 3u; ++sec) {
        for (uint k = 0u; k < 8u; ++k) {
            uint o = b.p + 2u * ((cs[sec] >> k) & 1u);
            uint rho = (rs[sec] >> (4u * k)) & 15u;
            float vv = F1R_VAL[(o << 4) | rho];
            v[sec * 8u + k] = vv;
            n2 = fma(vv, vv, n2);
        }
    }
    uint m = (uint(n2) >> 4) & (TETRA48_SHELLS - 1u);
    uint g = (hi16 >> 15) & 1u;
    float sc = gscale[g] * invnorm[m] * rscale[row];
    device half* wr = w + row * uint(DIN);
    // Columns in the permuted order the rotation writes the activation in.
    for (uint k = 0u; k < 24u; ++k) wr[j * 24u + k] = half(v[k] * sc);
    if (j == 0u) {
        for (uint t = 0u; t < uint(TW); ++t) wr[uint(NB) * 24u + t] = tail[row * uint(TW) + t];
    }
"""
# `after` is the activation the matrix will multiply, read by no thread: it makes
# the decode depend on it, so MLX schedules each matrix at its use and frees it
# after, instead of decoding all 168 at the head of the graph.
DEQ_INPUTS = ["words", "rows", "prefixes", "branches", "suffixes", "gscale", "invnorm", "rscale", "tail", "after"]

INPUTS = ["words", "rows", "prefixes", "branches", "suffixes", "gscale", "invnorm", "rscale", "tail", "x"]
DECODE = mx.fast.metal_kernel(name="llvq_tetra_lutd", input_names=INPUTS, output_names=["y"],
                              source=DECODE_BODY, header=HEADER)
ROWS = mx.fast.metal_kernel(name="llvq_tetra_lutdr", input_names=INPUTS + ["nr"], output_names=["y"],
                            source=ROWS_BODY, header=HEADER)
DEQ = mx.fast.metal_kernel(name="llvq_tetra_deq", input_names=DEQ_INPUTS, output_names=["w"],
                           source=DEQ_BODY, header=HEADER)

# One row, RR output rows a simdgroup: a lane loads its activation block once
# and decodes the RR rows' blocks against it. Each row's dot, lane order,
# butterfly and epilogue are the one-row kernel's, so each row's f16 is too.
# Rows at DQ or past it read gscale[2], gscale[3]: q and k stacked in one record.
DECODE2_BODY = """
    uint row0 = (thread_position_in_grid.x >> 5) * uint(RR);
    uint lane = thread_index_in_simdgroup;
    F1rTables tab = { rows, prefixes, branches, suffixes };
    float acc[RR];
    for (uint r = 0u; r < uint(RR); ++r) acc[r] = 0.0f;
    for (uint j = lane; j < uint(NB); j += 32u) {
        const device float4* x4 = (const device float4*)(x + j * 24u);
        float xv[24];
        for (uint q = 0u; q < 6u; ++q) {
            float4 t4 = x4[q];
            xv[4u * q] = t4.x; xv[4u * q + 1u] = t4.y; xv[4u * q + 2u] = t4.z; xv[4u * q + 3u] = t4.w;
        }
        for (uint r = 0u; r < uint(RR); ++r) {
            uint row = row0 + r;
            uint lo, hi16;
            f1r_load(words + row * STRIDE, j, lo, hi16);
            F1rV3Block b = f1r_v3_fetch(lo, hi16, tab);
            uint cs[3] = { b.c1, b.c2, b.c3 };
            uint rs[3] = { b.row1, b.row2, b.row3 };
            float a = 0.0f, n2 = 0.0f;
            for (uint sec = 0u; sec < 3u; ++sec) {
                for (uint k = 0u; k < 8u; ++k) {
                    uint o = b.p + 2u * ((cs[sec] >> k) & 1u);
                    uint rho = (rs[sec] >> (4u * k)) & 15u;
                    float v = F1R_VAL[(o << 4) | rho];
                    n2 = fma(v, v, n2);
                    a = fma(v, xv[sec * 8u + k], a);
                }
            }
            uint m = (uint(n2) >> 4) & (TETRA48_SHELLS - 1u);
            uint g = (hi16 >> 15) & 1u;
            uint go = row < uint(DQ) ? 0u : 2u;
            acc[r] += a * gscale[go + g] * invnorm[m];
        }
    }
    for (uint r = 0u; r < uint(RR); ++r)
        tetra48_rows_epilogue_h(acc[r], row0 + r, lane, rscale, tail, x, y, uint(NB), uint(TW));
"""
DECODE2 = mx.fast.metal_kernel(name="llvq_tetra_lutd_rr", input_names=["words", "rows", "prefixes", "branches",
                               "suffixes", "gscale", "invnorm", "rscale", "tail", "x"], output_names=["y"],
                               source=DECODE2_BODY, header=HEADER)
RR = 2
TG = 64  # threadgroup of the Tetra decode kernels; 64 read 7-10 % over 256, interleaved (smoke)


def decode_rows(words, gscale, rscale, tail, xr, shared, d_out, nb, tw, stride, dq):
    """One activation row through DECODE2, `[1, d_out]` f16."""
    s = shared
    assert d_out % (RR * 8) == 0, d_out
    return DECODE2(inputs=[words, s.rows, s.prefixes, s.branches, s.suffixes, gscale, s.invnorm, rscale, tail, xr],
                   template=[("NB", nb), ("TW", tw), ("STRIDE", stride), ("RR", RR), ("DQ", dq)],
                   grid=(d_out // RR * 32, 1, 1), threadgroup=(TG, 1, 1),
                   output_shapes=[(1, d_out)], output_dtypes=[mx.float16])[0]

# gate, up and SiLU(gate) * up in one dispatch, for one row. A simdgroup takes
# output i: it decodes block j of gate's row i and of up's row i against the
# same activation block, loaded once, and its lane 0 writes SiLU(g) * u in f16.
# The two dots, butterflies and epilogues are the decode kernel's; the product
# is taken in f32 where the unfused path rounds g and u to f16 first.
GU_BODY = """
    uint row = thread_position_in_grid.x >> 5;
    uint lane = thread_index_in_simdgroup;
    F1rTables tab = { rows, prefixes, branches, suffixes };
    const device uint* wgr = wg + row * STRIDE;
    const device uint* wur = wu + row * STRIDE;
    float ag = 0.0f, au = 0.0f;
    for (uint j = lane; j < uint(NB); j += 32u) {
        const device float4* x4 = (const device float4*)(x + j * 24u);
        float xv[24];
        for (uint q = 0u; q < 6u; ++q) {
            float4 t4 = x4[q];
            xv[4u * q] = t4.x; xv[4u * q + 1u] = t4.y; xv[4u * q + 2u] = t4.z; xv[4u * q + 3u] = t4.w;
        }
        for (uint w = 0u; w < 2u; ++w) {
            uint lo, hi16;
            f1r_load(w == 0u ? wgr : wur, j, lo, hi16);
            F1rV3Block b = f1r_v3_fetch(lo, hi16, tab);
            uint cs[3] = { b.c1, b.c2, b.c3 };
            uint rs[3] = { b.row1, b.row2, b.row3 };
            float a = 0.0f, n2 = 0.0f;
            for (uint sec = 0u; sec < 3u; ++sec) {
                for (uint k = 0u; k < 8u; ++k) {
                    uint o = b.p + 2u * ((cs[sec] >> k) & 1u);
                    uint rho = (rs[sec] >> (4u * k)) & 15u;
                    float v = F1R_VAL[(o << 4) | rho];
                    n2 = fma(v, v, n2);
                    a = fma(v, xv[sec * 8u + k], a);
                }
            }
            uint m = (uint(n2) >> 4) & (TETRA48_SHELLS - 1u);
            uint g = (hi16 >> 15) & 1u;
            if (w == 0u) ag += a * gg[g] * invnorm[m];
            else au += a * gu[g] * invnorm[m];
        }
    }
    ag = warp_sum(ag);
    au = warp_sum(au);
    if (lane == 0u) {
        const device float* xt = x + uint(NB) * 24u;
        float tg_ = 0.0f, tu_ = 0.0f;
        for (uint i = 0u; i < uint(TW); ++i) {
            tg_ = fma(float(tailg[row * uint(TW) + i]), xt[i], tg_);
            tu_ = fma(float(tailu[row * uint(TW) + i]), xt[i], tu_);
        }
        float gv = fma(ag, rg[row], tg_);
        float uv = fma(au, ru[row], tu_);
        y[row] = half(gv / (1.0f + metal::exp(-gv)) * uv);
    }
"""
GU = mx.fast.metal_kernel(
    name="llvq_tetra_gate_up",
    input_names=["wg", "wu", "rows", "prefixes", "branches", "suffixes", "gg", "gu", "invnorm", "rg", "ru",
                 "tailg", "tailu", "x"],
    output_names=["y"], source=GU_BODY, header=HEADER)

# Ablation, a measurement mode of this prototype only: LLVQ_MLX_ABLATE names
# kernel families ("tetra", "rot", "gu") swapped for an empty kernel of the same
# grid and output. The tokens are then garbage; only the time means anything.
ABLATE = set(filter(None, os.environ.get("LLVQ_MLX_ABLATE", "").split(",")))
EMPTY = mx.fast.metal_kernel(name="llvq_empty", input_names=["x"], output_names=["y"], source="""
    uint i = thread_position_in_grid.x;
    if (i < uint(NOUT)) y[i] = T(0);
""")


def empty(x: mx.array, n: int, dtype, rows: int = 1) -> mx.array:
    return EMPTY(inputs=[x], template=[("NOUT", n * rows), ("T", dtype)], grid=(max(n * rows, 32), 1, 1),
                 threadgroup=(256, 1, 1), output_shapes=[(rows, n)], output_dtypes=[dtype])[0]

# Rows above which a Tetra projection decodes its matrix to f16 and multiplies
# with MLX's GEMM rather than running the row kernel.
DENSE_FROM = 9


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
        # ROT stages GB groups of m floats in 7,168 floats (28 KiB) of threadgroup
        # memory. GB = 0 would leave its group loop stepping by zero forever.
        self.gb = min(self.k, ROT_TG_FLOATS // self.m)
        assert self.gb >= 1, (
            f"rotation of width {self.n}: its power-of-two group m = {self.m} exceeds the {ROT_TG_FLOATS} floats "
            f"of threadgroup memory ROT stages; this width needs another rotation kernel"
        )
        self.signs = mx.array(signs.astype(np.float32))
        self.small = mx.array(small.astype(np.float32))
        self._last = None
        # Projections that read this rotation's output (2 for q and k, gate and
        # up; 1 for down). The kept output is dropped after the last of them:
        # kept longer, the 96 rotations would hold every layer's activations of
        # a prefill chunk alive.
        self.users = 1
        self._hits = 0

    def reference(self, x: mx.array) -> mx.array:
        """The same rotation in MLX ops, four dispatches: the check for `__call__`."""
        lead = x.shape[:-1]
        v = (x.astype(mx.float32) * self.signs).reshape(-1, self.k, self.m)
        v = mx.hadamard_transform(v, scale=1.0 / math.sqrt(self.m))
        if self.k > 1:
            v = self.small @ v
        return v.reshape(*lead, self.n)[..., mx.array(permutation(self.n))]

    def __call__(self, x: mx.array) -> mx.array:
        # q and k (gate and up) hand in the same array: rotate it once.
        if self._last is not None and self._last[0] is x:
            out = self._last[1]
            self._hits += 1
            if self._hits >= self.users:
                self._last = None
            return out
        lead = x.shape[:-1]
        flat = x.astype(mx.float16).reshape(-1, self.n)
        rows = flat.shape[0]
        if "rot" in ABLATE:
            out = empty(flat, self.n, mx.float32, rows).reshape(*lead, self.n)
            self._hits = 1
            self._last = (x, out) if self.users > 1 else None
            return out
        out = ROT(inputs=[flat, self.signs, self.small],
                  template=[("N", self.n), ("M", self.m), ("K", self.k), ("NB", self.n // 24),
                            ("GB", self.gb)],
                  grid=(1024, rows, 1), threadgroup=(1024, 1, 1),
                  output_shapes=[(rows, self.n), (rows, self.n)],
                  output_dtypes=[mx.float32, mx.float32])[1].reshape(*lead, self.n)
        self._hits = 1
        self._last = (x, out) if self.users > 1 else None
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

    def dense(self, after: mx.array) -> mx.array:
        """This record as f16 `[d_out, d_in]` in the rotated basis, for the prefill."""
        s = self.shared
        return DEQ(inputs=[self._words, s.rows, s.prefixes, s.branches, s.suffixes, self._gscale, s.invnorm,
                           self._rscale, self._tail, after],
                   template=[("NB", self.nb), ("TW", self.tw), ("STRIDE", self.stride), ("DIN", self.d_in),
                             ("DOUT", self.d_out)],
                   grid=(-(-self.d_out * self.nb // 256) * 256, 1, 1), threadgroup=(256, 1, 1),
                   output_shapes=[(self.d_out, self.d_in)], output_dtypes=[mx.float16])[0]

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
        if n >= DENSE_FROM:
            xh = xr.astype(mx.float16)
            return (xh @ self.dense(xh).T).reshape(*lead, self.d_out)
        if n == 1 and "tetra" in ABLATE:
            return empty(xr, self.d_out, mx.float16).reshape(*lead, self.d_out)
        if n == 1 and self.d_out % (RR * 8):
            # The one-row lutd kernel, for widths DECODE2 does not tile: the mini
            # fixture's o_proj (136 rows). Every record of the sealed 4B tiles.
            y = DECODE(inputs=args, template=tmpl, grid=(self.d_out * 32, 1, 1), threadgroup=(TG, 1, 1),
                       output_shapes=[(1, self.d_out)], output_dtypes=[mx.float16])[0]
        elif n == 1:
            y = decode_rows(self._words, self._gscale, self._rscale, self._tail, xr, s, self.d_out, self.nb,
                            self.tw, self.stride, self.d_out)
        else:
            y = ROWS(inputs=args + [mx.array([n], dtype=mx.uint32)],
                     template=tmpl + [("DIN", self.d_in), ("DOUT", self.d_out)],
                     grid=(self.d_out * 32, -(-n // ROWS_MAX), 1), threadgroup=(256, 1, 1),
                     output_shapes=[(n, self.d_out)], output_dtypes=[mx.float16])[0]
        return y.reshape(*lead, self.d_out)


class QKFused:
    """q and k stacked in one record, one dispatch for one row: k's 1,024 rows alone
    fill a quarter of the GPU, stacked under q's 4,096 they ride along."""

    def __init__(self, q: "TetraLinear", k: "TetraLinear"):
        assert q.rot is k.rot and (q.d_in, q.nb, q.tw) == (k.d_in, k.nb, k.tw)
        self.q, self.k = q, k
        self.words = mx.concatenate([q._words, k._words])
        self.gscale = mx.concatenate([q._gscale, k._gscale])
        self.rscale = mx.concatenate([q._rscale, k._rscale])
        self.tail = mx.concatenate([q._tail, k._tail])
        mx.eval(self.words, self.gscale, self.rscale, self.tail)
        self._last = None

    def both(self, x: mx.array) -> tuple[mx.array, mx.array]:
        if self._last is not None and self._last[0] is x:
            out = self._last[1]
            self._last = None
            return out
        q, k = self.q, self.k
        lead = x.shape[:-1]
        if math.prod(lead) != 1 or "tetra" in ABLATE:
            out = (q(x), k(x))
        else:
            xr = q.rot(x).reshape(1, q.d_in)
            y = decode_rows(self.words, self.gscale, self.rscale, self.tail, xr, q.shared, q.d_out + k.d_out,
                            q.nb, q.tw, q.stride, q.d_out)
            out = (y[:, :q.d_out].reshape(*lead, q.d_out), y[:, q.d_out:].reshape(*lead, k.d_out))
        self._last = (x, out)
        return out


class QPart(nn.Module):
    def __init__(self, group: QKFused):
        super().__init__()
        self.group, self.inner = group, group.q
        self.freeze()

    def __call__(self, x):
        return self.group.both(x)[0]


class KPart(nn.Module):
    def __init__(self, group: QKFused):
        super().__init__()
        self.group, self.inner = group, group.k
        self.freeze()

    def __call__(self, x):
        return self.group.both(x)[1]


class FusedMLP(nn.Module):
    """`down(SiLU(gate x) * up x)`, with gate, up and the product in one dispatch for one row."""

    def __init__(self, gate: "TetraLinear", up: "TetraLinear", down):
        super().__init__()
        assert gate.rot is up.rot and (gate.d_out, gate.nb, gate.tw) == (up.d_out, up.nb, up.tw)
        self.gate_proj, self.up_proj, self.down_proj = gate, up, down
        self.freeze()

    def __call__(self, x: mx.array) -> mx.array:
        g, u = self.gate_proj, self.up_proj
        lead = x.shape[:-1]
        if math.prod(lead) != 1:
            return self.down_proj(nn.silu(g(x)) * u(x))
        xr = g.rot(x).reshape(1, g.d_in)
        if "gu" in ABLATE or "tetra" in ABLATE:
            return self.down_proj(empty(xr, g.d_out, mx.float16).reshape(*lead, g.d_out))
        s = g.shared
        h = GU(inputs=[g._words, u._words, s.rows, s.prefixes, s.branches, s.suffixes, g._gscale, u._gscale,
                       s.invnorm, g._rscale, u._rscale, g._tail, u._tail, xr],
               template=[("NB", g.nb), ("TW", g.tw), ("STRIDE", g.stride)],
               grid=(g.d_out * 32, 1, 1), threadgroup=(TG, 1, 1),
               output_shapes=[(1, g.d_out)], output_dtypes=[mx.float16])[0]
        return self.down_proj(h.reshape(*lead, g.d_out))


def int4_linear(pm: PackedModel, name: str) -> nn.QuantizedLinear:
    d = pm.records[name]
    q = nn.QuantizedLinear(d["d_in"], d["d_out"], bias=False, group_size=d["group"], bits=4)
    p = d["prefix"]
    q.weight = mx.array(np.ascontiguousarray(pm.tensor(f"{p}.qweight")).view("<u4"))
    q.scales = mx.array(pm.tensor(f"{p}.scales").astype(np.float16))
    q.biases = mx.array(pm.tensor(f"{p}.biases").astype(np.float16))
    q.freeze()
    return q


def build(path: str, ragged_embedding: bool = False):
    """The model on the GPU, every weight read from the packed directory as stored.

    `ragged_embedding`: for the check on `llvq-tetra/tests/fixtures/mini` only,
    whose hidden size 136 is not a multiple of the table's group of 64, which
    MLX's quantized embedding cannot hold. The table is then decoded once to f16
    by `llvq-tetra`. The served door never passes it, and the 4B's 2,560 does not
    need it.
    """
    pm = PackedModel(path)
    cfg = {k: v for k, v in pm.config.items() if k != "quantization_config"}
    args = qwen3.ModelArgs.from_dict(cfg)
    model = qwen3.Model(args)
    shared = Shared()
    rots = {key: Rotation(*pm.rotation(key)) for key in pm.rotations}
    for d in pm.records.values():
        if d.get("rotation"):
            rots[d["rotation"]].users += 1
    for r in rots.values():
        r.users -= 1
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
    for layer in model.model.layers:
        if "norm" in ABLATE:
            layer.input_layernorm = nn.Identity()
            layer.post_attention_layernorm = nn.Identity()
        if "qknorm" in ABLATE:
            layer.self_attn.q_norm = nn.Identity()
            layer.self_attn.k_norm = nn.Identity()
        att = layer.self_attn
        if type(att.q_proj).__name__ == "TetraLinear" and type(att.k_proj).__name__ == "TetraLinear":
            grp = QKFused(att.q_proj, att.k_proj)
            att.q_proj, att.k_proj = QPart(grp), KPart(grp)
        mlp = layer.mlp
        if type(mlp.gate_proj).__name__ == "TetraLinear" and type(mlp.up_proj).__name__ == "TetraLinear":
            layer.mlp = FusedMLP(mlp.gate_proj, mlp.up_proj, mlp.down_proj)
    weights = []
    for name, d in pm.raw.items():
        if d["encoding"] == "f16":
            weights.append((name, mx.array(pm.tensor(name).astype(np.float16))))
    emb = pm.raw["model.embed_tokens.weight"]
    assert emb["encoding"] == "quant" and emb["bits"] == 4, emb
    if emb["dims"][1] % emb["group"]:
        if not ragged_embedding:
            raise ValueError(f"embedding width {emb['dims'][1]} is not a multiple of its group {emb['group']}: "
                             f"MLX's quantized embedding cannot hold it")
        qe = nn.Embedding(emb["dims"][0], emb["dims"][1])
        qe.weight = mx.array(pm.dequantize_raw("model.embed_tokens.weight").astype(np.float16))
    else:
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

    head = "\n\n".join(pq.read_table(wiki_parquet()).column("text").to_pylist())[: n * 12]
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


class ChatFormat:
    """Qwen3's turn format with reasoning off, as `llvq_llm::chatfmt` writes it.

    `<|im_start|>{role}\n{text}<|im_end|>\n` a message, and the assistant's
    opening pre-filled with an empty reasoning block.
    """

    def __init__(self, path: str):
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(Path(path) / "tokenizer.json"))
        sid = self.tok.token_to_id
        self.im_start, self.im_end, self.eot = sid("<|im_start|>"), sid("<|im_end|>"), sid("<|endoftext|>")
        self.think = (sid("<think>"), sid("</think>"))
        self.nl = self.tok.encode("\n", add_special_tokens=False).ids

    def turn(self, line: str) -> list[int]:
        enc = lambda t: self.tok.encode(t, add_special_tokens=False).ids
        return ([self.im_start] + enc("user\n" + line) + [self.im_end] + self.nl
                + [self.im_start] + enc("assistant\n")
                + [self.think[0]] + self.nl + self.nl + [self.think[1]] + self.nl + self.nl)


def chat(model, path: str, temp: float, max_new: int) -> None:
    """A terminal conversation in `ChatFormat`, the cache kept across turns."""
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    fmt = ChatFormat(path)
    tok, im_end, eot, nl = fmt.tok, fmt.im_end, fmt.eot, fmt.nl
    sampler = make_sampler(temp=temp, top_p=0.9 if temp > 0 else 0.0)
    cache = make_prompt_cache(model)
    print(f"  temp {temp:.2f}, at most {max_new} new tokens a turn, reasoning off; /reset, /bye")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if line == "/bye":
            break
        if line == "/reset":
            cache = make_prompt_cache(model)
            continue
        if not line:
            continue
        ids = fmt.turn(line)
        held = cache[0].offset
        t0 = time.perf_counter()
        t1, out, shown = None, [], ""
        for (tid, _), _ in zip(generate_step(mx.array(ids), model, max_tokens=max_new, sampler=sampler,
                                             prompt_cache=cache), range(max_new)):
            tid = int(tid)
            if t1 is None:
                t1 = time.perf_counter()
            if tid in (im_end, eot):
                break
            out.append(tid)
            text = tok.decode(out)
            print(text[len(shown):], end="", flush=True)
            shown = text
        t2 = time.perf_counter()
        # generate_step runs one token ahead, so the cache may hold more than the
        # turn: cut it back to the prompt and the shown tokens, then close the turn.
        extra = cache[0].offset - (held + len(ids) + len(out))
        if extra > 0:
            trim_prompt_cache(cache, extra)
        mx.eval(model(mx.array([[im_end] + nl]), cache=cache))
        rate = (len(out) - 1) / (t2 - t1) if len(out) > 1 else 0.0
        print(f"\n  {len(out)} tokens, {rate:.1f} tok/s after a {len(ids)}-token prompt read in "
              f"{(t1 - t0):.2f} s; this REPL's rate, not a measurement")


def time_rounds(model, ids: list[int], n_new: int, timed: int, label: str) -> None:
    """`metalrun time`'s protocol: DISCARDED rounds, then `timed` ones, a fresh cache each, medians [min-max]."""
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
    print(f"{label} door {med(doors):7.2f} tok/s [{min(doors):.2f}-{max(doors):.2f}] | "
          f"decode {med(decodes):7.2f} tok/s [{min(decodes):.2f}-{max(decodes):.2f}] | "
          f"{1e3 / med(decodes):.2f} ms a decode token | first token {med(firsts) * 1e3:.1f} ms "
          f"[{min(firsts) * 1e3:.1f}-{max(firsts) * 1e3:.1f}]")
    print(f"peak memory {mx.get_peak_memory() / 1e9:.2f} GB")
    print(f"tokens sha256 {sha_u32(ref)}")


def main():
    path, mode = sys.argv[1], sys.argv[2]
    print(f"mlx {mx.__version__} | {mx.default_device()} | packed {path}")
    print("measurement door: no served config, f16 activations, not bit-exact with the Rust kernels; "
          "the served door is serve.py")
    t = time.perf_counter()
    model, pm, tetra_bytes = build(path)
    print(f"built in {time.perf_counter() - t:.1f} s | Tetra resident {tetra_bytes / 1e9:.3f} GB | "
          f"active {mx.get_active_memory() / 1e9:.2f} GB")
    if mode == "chat":
        temp = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
        max_new = int(sys.argv[4]) if len(sys.argv) > 4 else 512
        chat(model, path, temp, max_new)
        return
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
    time_rounds(model, prompt_ids(path, n_prompt), n_new, timed, "llvq-mlx")


if __name__ == "__main__":
    main()
