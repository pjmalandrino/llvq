"""The served kernels as torch ops vLLM can compile around and capture.

vLLM runs the model under `torch.compile` with the token count dynamic, then
captures CUDA graphs per batch size. Two consequences shape this file.

* **The token count must not branch in traced code.** A Python `if M == 1` in a
  layer's `apply` becomes a guard on a dynamic size. So every op that chooses a
  path by M is one opaque op, `llvq_vllm::*`, defined here with a fake
  implementation that only gives the output shape. Dynamo sees a call; the
  choice happens inside, at run time, and at capture time for each size.
* **Nothing in an op may synchronize with the host or read a device value back.**
  The ops below allocate their outputs and launch; they never call `.item()`,
  never move a tensor to the CPU, and never size anything from a device value.

The path an op takes for M rows:

| M | Tetra | int4 g128 | tied head |
|---|---|---|---|
| up to `rows_max` | `tv_tetra48`, once a row | `tv_q4`, once a row | `tv_emb_q4`, once a row |
| above | `tetra_dequant`, a GEMM | `int4_dequant`, a GEMM | the same, by vocabulary chunks |

The row kernels are the served kernels, bit for bit our engine's arithmetic. The
dense path is f16 weights against f16 activations with cuBLAS, like vLLM's own
FP16 arm, and is gated to a tolerance, never to identity.

On a CPU each op runs a reference in torch: the same branches, f32 in place of
the row kernels. That is what the tests run on a Mac.
"""

from __future__ import annotations

import torch

from . import buffers

LIB = "llvq_vllm"
# Threads a block for the matvecs: eight rows a block, as `fused_cuda` launches.
CUDA_THREADS = 256
# Vocabulary rows dequantized at once by the head's dense path: 16,384 rows of
# 2,560 in f16 are 84 MB of scratch, against 778 MB for the whole table.
HEAD_CHUNK = 16384

_lib = torch.library.Library(LIB, "DEF")
_lib.define("rotate(Tensor x, Tensor signbits, Tensor small, int n, int m, int k, float inv, "
            "int threads) -> Tensor")
_lib.define("tetra_mm(Tensor x, Tensor words, Tensor gscale, Tensor rscale, Tensor tail, "
            "int d_out, int nblocks, int tail_w, int stride_u32, int tile, int rows_max) -> Tensor")
_lib.define("q4_mm(Tensor x, Tensor wq, Tensor scales, Tensor biases, int d_out, int d_in, "
            "int gpr, int rows_max) -> Tensor")
_lib.define("emb_q4_gather(Tensor ids, Tensor wq, Tensor scales, Tensor biases, int d, "
            "int gpr) -> Tensor")
_lib.define("emb_q4_mm(Tensor x, Tensor wq, Tensor scales, Tensor biases, int vocab, int d, "
            "int gpr, int rows_max) -> Tensor")


def _cuda_ext():
    """The extension that registers `torch.ops.llvq_cuda`, built once a process."""
    from llvq_tetra import metal

    metal.extension_for("cuda")
    return metal


def prepare(device: torch.device) -> None:
    """Build the extension and move every shared table to `device`, at load time.

    Each of these is cached on first use, and a first use is a host-to-device
    copy. vLLM captures a CUDA graph per batch size with no warmup run by
    default, so a first use inside a capture would be a copy the capture
    forbids. Called from `process_weights_after_loading`, it happens before.
    """
    buffers.decode_tables(str(device))
    if device.type == "cuda":
        metal = _cuda_ext()
        metal.shared_tables("cuda")
        metal.invnorm_on("cuda")


# --- rotate ------------------------------------------------------------------


def _rb(signbits, small, n, m, k, inv) -> buffers.RotationBuffers:
    return buffers.RotationBuffers("op", signbits, small, n, m, k, inv)


def _rotate_cuda(x, signbits, small, n, m, k, inv, threads):
    _cuda_ext()
    x16 = x.reshape(-1, n).to(torch.float16).contiguous()
    return torch.ops.llvq_cuda.rot_apply_rows(x16, signbits, small, n, m, k, inv, threads)


def _rotate_cpu(x, signbits, small, n, m, k, inv, threads):
    # The kernel widens an f16 activation; the reference does the same rounding
    # first, so a CPU test sees what the card would be handed.
    x16 = x.reshape(-1, n).to(torch.float16)
    return buffers.rotate_reference(x16, _rb(signbits, small, n, m, k, inv))


# --- the Tetra matvec ------------------------------------------------------


def _tetra_rows_cuda(x, words, gscale, rscale, tail, d_out, nblocks, tail_w, stride_u32, tile):
    metal = _cuda_ext()
    tables = metal.shared_tables("cuda")
    invnorm = metal.invnorm_on("cuda")
    x = x.to(torch.float32).contiguous()
    ys = [
        torch.ops.llvq_cuda.tv_tetra48(words, *tables, gscale, invnorm, rscale, tail, x[i],
                                       d_out, nblocks, tail_w, stride_u32, tile, CUDA_THREADS)
        for i in range(x.shape[0])
    ]
    return torch.stack(ys) if ys else x.new_empty((0, d_out), dtype=torch.float16)


def _tetra_dense(x, words, gscale, rscale, tail, d_out, nblocks, tail_w):
    w = buffers.tetra_dequant(words, gscale, rscale, tail, d_out, nblocks, tail_w, torch.float16)
    return torch.matmul(x.to(torch.float16), w.t())


def _tetra_mm_cuda(x, words, gscale, rscale, tail, d_out, nblocks, tail_w, stride_u32, tile,
                   rows_max):
    if x.shape[0] <= rows_max:
        return _tetra_rows_cuda(x, words, gscale, rscale, tail, d_out, nblocks, tail_w,
                                stride_u32, tile)
    return _tetra_dense(x, words, gscale, rscale, tail, d_out, nblocks, tail_w)


def _tetra_mm_cpu(x, words, gscale, rscale, tail, d_out, nblocks, tail_w, stride_u32, tile,
                  rows_max):
    if x.shape[0] <= rows_max:
        w = buffers.tetra_dequant(words, gscale, rscale, tail, d_out, nblocks, tail_w,
                                  torch.float32)
        return (x.to(torch.float32) @ w.t()).to(torch.float16)
    return _tetra_dense(x, words, gscale, rscale, tail, d_out, nblocks, tail_w)


# --- the int4 g128 matvec ----------------------------------------------------


def _q4_mm_cuda(x, wq, scales, biases, d_out, d_in, gpr, rows_max):
    if x.shape[0] <= rows_max:
        _cuda_ext()
        x32 = x.to(torch.float32).contiguous()
        ys = [torch.ops.llvq_cuda.tv_q4(wq, scales, biases, x32[i], d_out, d_in, gpr,
                                        CUDA_THREADS) for i in range(x32.shape[0])]
        return torch.stack(ys) if ys else x.new_empty((0, d_out), dtype=torch.float16)
    w = buffers.int4_dequant(wq, scales, biases, d_out, d_in, 128, torch.float16)
    return torch.matmul(x.to(torch.float16), w.t())


def _q4_mm_cpu(x, wq, scales, biases, d_out, d_in, gpr, rows_max):
    dtype = torch.float32 if x.shape[0] <= rows_max else torch.float16
    w = buffers.int4_dequant(wq, scales, biases, d_out, d_in, 128, dtype)
    return (x.to(dtype) @ w.t()).to(torch.float16)


# --- the int4 g64 vocabulary -------------------------------------------------


def _emb_q4_gather(ids, wq, scales, biases, d, gpr):
    flat = ids.reshape(-1).to(torch.int64)
    rows = buffers.int4_rows(wq, scales, biases, flat, d, 64, torch.float16)
    return rows.reshape(*ids.shape, d)


def _emb_q4_dense(x, wq, scales, biases, vocab, d, gpr, dtype):
    x = x.to(dtype)
    out = torch.empty((x.shape[0], vocab), dtype=dtype, device=x.device)
    q = wq.reshape(vocab, d // 2)
    s, b = scales.reshape(vocab, gpr), biases.reshape(vocab, gpr)
    for r0 in range(0, vocab, HEAD_CHUNK):
        r1 = min(vocab, r0 + HEAD_CHUNK)
        w = buffers.int4_dequant(q[r0:r1], s[r0:r1], b[r0:r1], r1 - r0, d, 64, dtype)
        out[:, r0:r1] = x @ w.t()
    return out.to(torch.float16)


def _emb_q4_mm_cuda(x, wq, scales, biases, vocab, d, gpr, rows_max):
    if x.shape[0] <= rows_max:
        _cuda_ext()
        x16 = x.to(torch.float16).contiguous()
        ys = [torch.ops.llvq_cuda.tv_emb_q4(wq, scales, biases, x16[i], vocab, d, gpr,
                                            CUDA_THREADS) for i in range(x16.shape[0])]
        return torch.stack(ys) if ys else x.new_empty((0, vocab), dtype=torch.float16)
    return _emb_q4_dense(x, wq, scales, biases, vocab, d, gpr, torch.float16)


def _emb_q4_mm_cpu(x, wq, scales, biases, vocab, d, gpr, rows_max):
    dtype = torch.float32 if x.shape[0] <= rows_max else torch.float16
    return _emb_q4_dense(x, wq, scales, biases, vocab, d, gpr, dtype)


for _name, _cuda, _cpu in (
    ("rotate", _rotate_cuda, _rotate_cpu),
    ("tetra_mm", _tetra_mm_cuda, _tetra_mm_cpu),
    ("q4_mm", _q4_mm_cuda, _q4_mm_cpu),
    ("emb_q4_gather", _emb_q4_gather, _emb_q4_gather),
    ("emb_q4_mm", _emb_q4_mm_cuda, _emb_q4_mm_cpu),
):
    _lib.impl(_name, _cuda, "CUDA")
    _lib.impl(_name, _cpu, "CPU")


# --- shapes, for Dynamo --------------------------------------------------------


@torch.library.register_fake(f"{LIB}::rotate")
def _(x, signbits, small, n, m, k, inv, threads):
    return x.new_empty((x.numel() // n, n), dtype=torch.float32)


@torch.library.register_fake(f"{LIB}::tetra_mm")
def _(x, words, gscale, rscale, tail, d_out, nblocks, tail_w, stride_u32, tile, rows_max):
    return x.new_empty((x.shape[0], d_out), dtype=torch.float16)


@torch.library.register_fake(f"{LIB}::q4_mm")
def _(x, wq, scales, biases, d_out, d_in, gpr, rows_max):
    return x.new_empty((x.shape[0], d_out), dtype=torch.float16)


@torch.library.register_fake(f"{LIB}::emb_q4_gather")
def _(ids, wq, scales, biases, d, gpr):
    return wq.new_empty((*ids.shape, d), dtype=torch.float16)


@torch.library.register_fake(f"{LIB}::emb_q4_mm")
def _(x, wq, scales, biases, vocab, d, gpr, rows_max):
    return x.new_empty((x.shape[0], vocab), dtype=torch.float16)
