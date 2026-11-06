"""The Metal kernels through `torch.mps.compile_shader`: nothing to compile on the user's machine.

The shaders are the served ones, the copies `bin/tetratables` ships and `metal.shader_source`
checks by digest. PyTorch hands their source to the Metal compiler of the system at the first
call, the way the Objective-C++ binding of `csrc/tetra_decode.mm` did, but with no extension to
build: no ninja, no command line tools.

`compile_shader` cannot size threadgroup memory at dispatch time, and the served kernels take
their activation tile as a `[[threadgroup(0)]]` argument. So each kernel used here is a variant
of a served one, made by editing its text: the argument line goes, and the same buffer is
declared in the body with its size fixed at compile time. Every edit must match exactly once,
so a served file that changes upstream is refused here rather than half-edited. The arithmetic
of a variant is the served text itself, and the gates compare it with the served kernel bit for
bit.

The rows kernels are the exception, and the deviation file of the Mac block says so: their outer
loops are new, around the served helpers, so that one dispatch decodes a block once and applies
it to several activation rows. Each row's arithmetic is the single-row kernel's, in the same
order, which gate G-D checks bit for bit.
"""

from __future__ import annotations

import weakref
from functools import lru_cache

import numpy as np
import torch

from . import metal

# Activation rows one dispatch takes. Four, as `llvq_cuda::tile::PREFILL_ROWS`: four tiles of
# 64 blocks are 24,576 B of the 32,768 an Apple threadgroup has.
ROWS = 4

# Columns the 4-bit rows kernel stages per row. Any multiple of 256 walks the words in the order
# the served untiled kernel does, so the result does not depend on it (`tv_q4_metal_tiled`).
Q4_ROWS_TILE = 1024

THREADGROUP = 256


def kernel_text(src: str, name: str) -> str:
    start = src.index(f"kernel void {name}(")
    return src[start : src.index("\n}\n", start) + 3]


def helper_text(src: str, signature: str) -> str:
    start = src.index(signature)
    return src[start : src.index("\n}\n", start) + 3]


def edit(text: str, what: str, edits: list[tuple[str, str]]) -> str:
    for old, new in edits:
        n = text.count(old)
        if n != 1:
            raise ValueError(f"{what}: {old!r} matches {n} times in the served text, want 1")
        text = text.replace(old, new)
    return text


def static_threadgroup(src: str, name: str, new_name: str, decl: str) -> str:
    """`name`, renamed, its one `[[threadgroup(0)]]` argument moved into the body as `decl`."""
    text = kernel_text(src, name)
    lines = [line for line in text.split("\n") if "[[threadgroup(0)]]" in line]
    if len(lines) != 1:
        raise ValueError(f"{name}: {len(lines)} threadgroup arguments, want 1")
    return edit(
        text,
        name,
        [
            (f"kernel void {name}(", f"kernel void {new_name}("),
            (lines[0] + "\n", ""),
            ("\n{\n", "\n{\n    " + decl + "\n"),
        ],
    )


def _library(source: str):
    if not torch.backends.mps.is_available():
        raise RuntimeError("these kernels run on Metal, and no MPS device is available")
    return torch.mps.compile_shader(source)


# ---------------------------------------------------------------------------------------------
# The Tetra matvec
# ---------------------------------------------------------------------------------------------


TETRA_ROWS_KERNEL = """
kernel void tv_tetra48_rows_cs(const device uint*   words          [[buffer(0)]],
                               constant uint&       row_stride_u32 [[buffer(1)]],
                               const device uint*   rows           [[buffer(2)]],
                               const device uchar*  prefixes       [[buffer(3)]],
                               const device ushort* branches       [[buffer(4)]],
                               const device uchar*  suffixes       [[buffer(5)]],
                               const device float*  gscale         [[buffer(6)]],
                               const device float*  invnorm        [[buffer(7)]],
                               const device float*  rscale         [[buffer(8)]],
                               const device half*   tail           [[buffer(9)]],
                               const device float*  x              [[buffer(10)]],
                               device float*        y              [[buffer(11)]],
                               constant uint&       nblocks        [[buffer(12)]],
                               constant uint&       tail_w         [[buffer(13)]],
                               constant uint&       nrows          [[buffer(14)]],
                               constant uint&       x_stride       [[buffer(15)]],
                               constant uint&       d_out          [[buffer(16)]],
                               uint tid  [[thread_position_in_threadgroup]],
                               uint gid  [[thread_position_in_grid]],
                               uint tgs  [[threads_per_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float xs[LLVQ_ROWS * LLVQ_TILE_BLOCKS * 24u];
    uint row = gid >> 5;
    const device uint* wrow = words + row * row_stride_u32;
    F1rTables tab = { rows, prefixes, branches, suffixes };
    float acc[LLVQ_ROWS];
    for (uint r = 0u; r < LLVQ_ROWS; ++r) {
        acc[r] = 0.0f;
    }

    uint ntiles = (nblocks + LLVQ_TILE_BLOCKS - 1u) / LLVQ_TILE_BLOCKS;
    for (uint t = 0u; t < ntiles; ++t) {
        uint jlo = t * LLVQ_TILE_BLOCKS;
        uint jhi = min(jlo + LLVQ_TILE_BLOCKS, nblocks);
        uint n = (jhi - jlo) * 24u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0u; r < nrows; ++r) {
            for (uint i = tid; i < n; i += tgs) {
                xs[r * LLVQ_TILE_BLOCKS * 24u + i] = x[r * x_stride + jlo * 24u + i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = jlo + lane; j < jhi; j += 32u) {
            uint lo, hi16;
            f1r_load(wrow, j, lo, hi16);
            // The decode of `tetra48_dot`, once a block for every row.
            uint q[6];
            f1r_v3_quads(lo, hi16, tab, q);
            uint m = (tetra48_n2(q) >> 4) & (TETRA48_SHELLS - 1u);
            uint g = (hi16 >> 15) & 1u;
            for (uint r = 0u; r < nrows; ++r) {
                acc[r] += tetra48_dot_q(q, m, g,
                                        xs + r * LLVQ_TILE_BLOCKS * 24u + (j - jlo) * 24u,
                                        gscale, invnorm);
            }
        }
    }

    for (uint r = 0u; r < nrows; ++r) {
        float a = warp_sum(acc[r]);
        if (lane == 0u) {
            // The epilogue of `tv_tetra48_metal`, one row at a time.
            float tv = 0.0f;
            const device float* xt = x + r * x_stride + nblocks * 24u;
            for (uint i = 0u; i < tail_w; ++i) {
                tv = fma(float(tail[row * tail_w + i]), xt[i], tv);
            }
            y[r * d_out + row] = fma(a, rscale[row], tv);
        }
    }
}
"""


@lru_cache(maxsize=4)
def tetra_library(tile: int):
    src = metal.tiled_source(tile)
    # `tetra48_dot` with its decode taken out: the same fma chain, the same scaling, read from the
    # served text so the rows kernel cannot drift from the single-row one.
    dot = edit(
        helper_text(src, "inline float tetra48_dot(uint lo,"),
        "tetra48_dot",
        [
            ("inline float tetra48_dot(uint lo,\n                         uint hi16,\n"
             "                         F1rTables t,\n",
             "inline float tetra48_dot_q(thread const uint q[6],\n                           uint m,\n"
             "                           uint g,\n"),
            ("    uint q[6];\n    f1r_v3_quads(lo, hi16, t, q);\n", ""),
            ("    uint m = (tetra48_n2(q) >> 4) & (TETRA48_SHELLS - 1u);\n"
             "    uint g = (hi16 >> 15) & 1u;\n", ""),
        ],
    )
    single = static_threadgroup(
        src, "tv_tetra48_metal", "tv_tetra48_cs", "threadgroup float xs[LLVQ_TILE_BLOCKS * 24u];"
    )
    return _library(f"#define LLVQ_ROWS {ROWS}u\n{src}\n{single}\n{dot}\n{TETRA_ROWS_KERNEL}")


def tetra_matvec(words, gscale, rscale, tail, x, desc: dict, tile: int, rows: bool = True):
    """`y = W x` for a Tetra record, `x` of shape `(n, d_in)` f32 on mps, `y` of `(n, d_out)`."""
    lib = tetra_library(tile)
    tables = metal.shared_tables("mps")
    invnorm = metal.invnorm_on("mps")
    d_out, nblocks, tail_w = desc["d_out"], desc["nblocks"], desc["tail_cols"]
    stride = metal.stride_u32(nblocks)
    x = x.contiguous()
    n = x.shape[0]
    y = torch.empty(n, d_out, device=x.device, dtype=torch.float32)
    if not rows:
        for i in range(n):
            lib.tv_tetra48_cs(
                words, stride, *tables, gscale, invnorm, rscale, tail, x[i], y[i],
                nblocks, tail_w, threads=d_out * 32, group_size=THREADGROUP,
            )
        return y
    for r0 in range(0, n, ROWS):
        r = min(ROWS, n - r0)
        lib.tv_tetra48_rows_cs(
            words, stride, *tables, gscale, invnorm, rscale, tail, x[r0], y[r0],
            nblocks, tail_w, r, x.shape[1], d_out, threads=d_out * 32, group_size=THREADGROUP,
        )
    return y


# ---------------------------------------------------------------------------------------------
# The 4-bit matvec
# ---------------------------------------------------------------------------------------------


Q4_ROWS_KERNEL = """
kernel void tv_q4_rows_cs(const device uint*   wq       [[buffer(0)]],
                          const device ushort* scales   [[buffer(1)]],
                          const device ushort* biases   [[buffer(2)]],
                          const device float*  x        [[buffer(3)]],
                          device float*        y        [[buffer(4)]],
                          constant uint&       d_in     [[buffer(5)]],
                          constant uint&       gpr      [[buffer(6)]],
                          constant uint&       nrows    [[buffer(7)]],
                          constant uint&       x_stride [[buffer(8)]],
                          constant uint&       d_out    [[buffer(9)]],
                          uint tid  [[thread_position_in_threadgroup]],
                          uint gid  [[thread_position_in_grid]],
                          uint tgs  [[threads_per_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float xs[LLVQ_ROWS * LLVQ_Q4_ROWS_TILE];
    uint row = gid >> 5;
    uint nwords = d_in >> 3;
    uint w0 = row * nwords;
    uint s0 = row * gpr;
    float acc[LLVQ_ROWS];
    for (uint r = 0u; r < LLVQ_ROWS; ++r) {
        acc[r] = 0.0f;
    }

    for (uint c0 = 0u; c0 < d_in; c0 += LLVQ_Q4_ROWS_TILE) {
        uint chi = min(c0 + LLVQ_Q4_ROWS_TILE, d_in);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0u; r < nrows; ++r) {
            for (uint i = tid; i < chi - c0; i += tgs) {
                xs[r * LLVQ_Q4_ROWS_TILE + i] = x[r * x_stride + c0 + i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint wi = (c0 >> 3) + lane; wi < (chi >> 3); wi += 32u) {
            uint p = wq[w0 + wi];
            uint c = wi << 3;
            uint g = s0 + c / LLVQ_Q4_GROUP;
            ushort sb = scales[g];
            ushort bb = biases[g];
            for (uint k = 0u; k < 8u; ++k) {
                float w = q4_deq((p >> (4u * k)) & 0xfu, sb, bb);
                for (uint r = 0u; r < nrows; ++r) {
                    acc[r] = fma(w, xs[r * LLVQ_Q4_ROWS_TILE + c - c0 + k], acc[r]);
                }
            }
        }
    }
    for (uint r = 0u; r < nrows; ++r) {
        float a = warp_sum(acc[r]);
        if (lane == 0u) {
            y[r * d_out + row] = a;
        }
    }
}
"""


@lru_cache(maxsize=4)
def q4_library(tile_cols: int):
    if tile_cols % 256:
        raise ValueError(f"tile_cols {tile_cols} is not a multiple of 256: the lanes would interleave differently")
    src = metal.q4_source()
    single = static_threadgroup(
        src, "tv_q4_metal_tiled", "tv_q4_tiled_cs", "threadgroup float xs[LLVQ_Q4_TILE_COLS];"
    )
    defines = (
        f"#define LLVQ_ROWS {ROWS}u\n#define LLVQ_Q4_TILE_COLS {tile_cols}u\n"
        f"#define LLVQ_Q4_ROWS_TILE {Q4_ROWS_TILE}u\n"
    )
    return _library(defines + src + "\n" + single + "\n" + Q4_ROWS_KERNEL)


def q4_matvec(wq, scales, biases, x, desc: dict, tile_cols: int, rows: bool = True):
    """`y = W x` for an int4 g128 record, `x` of shape `(n, d_in)` f32 on mps."""
    lib = q4_library(tile_cols)
    d_out, d_in, gpr = desc["d_out"], desc["d_in"], desc["groups_per_row"]
    x = x.contiguous()
    n = x.shape[0]
    y = torch.empty(n, d_out, device=x.device, dtype=torch.float32)
    if not rows:
        for i in range(n):
            lib.tv_q4_tiled_cs(
                wq, scales, biases, x[i], y[i], d_in, gpr, tile_cols,
                threads=d_out * 32, group_size=THREADGROUP,
            )
        return y
    for r0 in range(0, n, ROWS):
        r = min(ROWS, n - r0)
        lib.tv_q4_rows_cs(
            wq, scales, biases, x[r0], y[r0], d_in, gpr, r, d_in, d_out,
            threads=d_out * 32, group_size=THREADGROUP,
        )
    return y


# ---------------------------------------------------------------------------------------------
# The 4-bit vocabulary
# ---------------------------------------------------------------------------------------------


@lru_cache(maxsize=4)
def vocab_library(d: int):
    src = metal.shader_source("emb_q4.metal")
    # The served output layer reads the activation in half, as the served engine carries it. This
    # path carries f32, so the variant reads f32: one type in the signature, nothing else.
    head = static_threadgroup(src, "tv_emb_q4_metal", "tv_emb_q4_f32_cs", "threadgroup float xs[LLVQ_EMB_D];")
    head = edit(head, "tv_emb_q4_metal", [("const device half* x ", "const device float* x ")])
    return _library(f"#define LLVQ_EMB_D {d}u\n{src}\n{head}")


def vocab_gather(wq, scales, biases, ids, d: int, gpr: int):
    """The rows of the table for `ids`, f32, exactly as the dense path dequantizes them."""
    lib = vocab_library(d)
    flat = ids.reshape(-1).to(device=wq.device, dtype=torch.int32).contiguous()
    y = torch.empty(flat.numel(), d, device=wq.device, dtype=torch.float32)
    lib.emb_q4_gather_metal(
        wq, scales, biases, flat, y, d, gpr, 0, threads=flat.numel() * THREADGROUP, group_size=THREADGROUP
    )
    return y.reshape(*ids.shape, d)


def vocab_logits(wq, scales, biases, h, vocab: int, d: int, gpr: int):
    """`h · Wᵀ` against the 4-bit table, `h` of shape `(n, d)` f32 on mps, one dispatch a row."""
    lib = vocab_library(d)
    h = h.contiguous()
    y = torch.empty(h.shape[0], vocab, device=h.device, dtype=torch.float32)
    for i in range(h.shape[0]):
        lib.tv_emb_q4_f32_cs(
            wq, scales, biases, h, y, d, gpr, i * d, i * vocab, threads=vocab * 32, group_size=THREADGROUP
        )
    return y


# ---------------------------------------------------------------------------------------------
# The rotation
# ---------------------------------------------------------------------------------------------


ROT_KMAX = 32


@lru_cache(maxsize=1)
def rotation_library():
    src = metal.shader_source("llvq_rot.metal")
    load = edit(
        helper_text(src, "inline void rot_load(const device ushort* xin,"),
        "rot_load",
        [
            ("inline void rot_load(const device ushort* xin,", "inline void rot_load_f32(const device float* xin,"),
            ("float v = rot_h2f(xin[x_off + i]);", "float v = xin[x_off + i];"),
        ],
    )
    rows = edit(
        kernel_text(src, "rot_apply_rows_metal"),
        "rot_apply_rows_metal",
        [
            ("kernel void rot_apply_rows_metal(const device ushort* xin ",
             "kernel void rot_apply_rows_f32(const device float* xin "),
            ("rot_load(", "rot_load_f32("),
        ],
    )
    return _library(f"{src}\n{load}\n{rows}")


class RotationKernel:
    """`Q x` on the device for many rows, one dispatch, kept for the other projections that share it.

    q, k and v rotate the same activation, and so do gate and up. The rotation is kept for the next
    caller only if that caller hands over the same tensor object, unchanged since (its version
    counter), and it is dropped once all `consumers` have read it.

    The tensor object and not its storage address: on 2026-10-09 the address was the key, and the
    MPS allocator gave the next token's activation the same address, so a new activation received
    the rotation of an old one and the tokens diverged at the third (proofs/...-ECARTS.md, É3).
    """

    def __init__(self, signs: np.ndarray, small: np.ndarray, device: str = "mps", consumers: int = 1):
        n = signs.shape[0]
        self.n, self.m = n, n & (-n)
        self.k = n // self.m
        if self.k > ROT_KMAX:
            raise ValueError(f"the mix is {self.k} by {self.k}, the kernel takes {ROT_KMAX} at most")
        bits = np.zeros(-(-n // 32), dtype=np.uint32)
        neg = np.flatnonzero(np.asarray(signs) < 0)
        np.bitwise_or.at(bits, neg >> 5, (np.uint32(1) << (neg & 31).astype(np.uint32)))
        padded = np.zeros((ROT_KMAX, ROT_KMAX), dtype=np.float32)
        padded[: self.k, : self.k] = np.asarray(small, dtype=np.float32)
        self.signbits = torch.from_numpy(bits).to(device)
        self.small = torch.from_numpy(padded.reshape(-1)).to(device)
        self.inv = float(np.float32(1.0 / np.sqrt(np.float64(self.m))))
        self.consumers = consumers
        self._kept = None  # (weak reference to the input, its version, the rotation, reads so far)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self._kept is not None:
            ref, version, out, reads = self._kept
            if ref() is x and x._version == version:
                reads += 1
                self._kept = None if reads >= self.consumers else (ref, version, out, reads)
                return out
            self._kept = None
        flat = x.reshape(-1, self.n).to(torch.float32).contiguous()
        rows = flat.shape[0]
        out = torch.empty_like(flat)
        scratch = torch.empty_like(flat)
        rotation_library().rot_apply_rows_f32(
            flat, self.signbits, self.small, out, scratch, self.n, self.m, self.k, self.inv, 0, self.n,
            threads=rows * THREADGROUP, group_size=THREADGROUP,
        )
        if self.consumers > 1:
            self._kept = (weakref.ref(x), x._version, out, 1)
        return out
