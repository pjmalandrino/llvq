"""The served Tetra matvec in the forward pass: weights compressed in memory.

M1 of `docs/plan-transformers.md`. Stage 1 loads the packed model and gives the
right tokens with the weights **dense**, 8.05 GB at f16 on the 4B against a 1.42 GB
file. This module holds each Tetra record as the GPU reads it, dispatches
`tv_tetra48_metal`, and never writes the matrix.

## What lives on the device, per projection

The `tetra48` words, the two gain centroids in f32, the row scales in f32, the tail
in f16. That is what the kernel binds, and the widths are the kernel's, not the
file's: the file holds the tail in f32 and the row scales in f64. So this path is
**not** bit-identical to a dense reconstruction, by construction, and the gate is
the tokens rather than a widened tolerance.

Shared across projections: the four decode tables and `invnorm`, 32 entries of
`1/sqrt(16m)` with `invnorm[0] = 0`, exactly as `llvq_llm::fused` builds them.

## The rotation

In torch, f32, on the device: the sign flip, the butterfly, then the small mix.
The weights are stored in the rotated basis, so the activation has to be rotated
before the matvec; un-rotating the weights instead is what stage 1 did, and it needs
them materialized.

`rot_apply_metal` is the served path's own rotation and reads f16. Binding it is a
second op with a gate of its own, and swapping it in here is a drop-in. What this
costs meanwhile is redundant work: `q`, `k` and `v` share a rotation under
`LLVQ_ROT_SHARE=1` and each rotates its own copy of the same vector.

## The prefill

One dispatch per token. The kernel takes one activation vector, and a dense
fallback for `T > 1` would need the materialized weight, which is the thing this
removes. So a prompt of T tokens costs T dispatches a projection, and no number
about speed comes out of this stage.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from . import metal
from .modules import LlvqLinear

TETRA48_SHELLS = 32


def invnorm_table() -> np.ndarray:
    """`1/sqrt(16m)` in f32, `invnorm[0] = 0`, computed in f64 and narrowed once.

    The same table `llvq_llm::fused::tetra48_tables` builds, and the same order of
    operations: the division in f64, then one narrowing.
    """
    out = np.zeros(TETRA48_SHELLS, dtype=np.float32)
    for m in range(1, TETRA48_SHELLS):
        out[m] = np.float32(1.0 / np.sqrt(np.float64(16 * m)))
    return out


def wht_torch(v: torch.Tensor) -> torch.Tensor:
    """The scaled Walsh-Hadamard transform along the last axis, in torch.

    The same butterfly as `llvqhf.dequant.wht_rows`, in f32 on the device. It is not
    the f64 host transform and does not claim to be: this path feeds a kernel whose
    own arithmetic is f32.
    """
    n = v.shape[-1]
    if n & (n - 1):
        raise ValueError(f"the transform needs a power of two, got {n}")
    length = 1
    while length < n:
        x = v.reshape(*v.shape[:-1], n // (2 * length), 2, length)
        a, b = x[..., 0, :], x[..., 1, :]
        v = torch.stack((a + b, a - b), dim=-2).reshape(*v.shape)
        length <<= 1
    return v * (1.0 / np.sqrt(float(n)))


class Rotation(nn.Module):
    """`x ← Q x` on the device, the transform the codes were written under."""

    def __init__(self, signs: np.ndarray, small: np.ndarray, device, dtype=torch.float32):
        super().__init__()
        n = signs.shape[0]
        self.n = n
        self.m = n & (-n)
        self.k = n // self.m
        if small.shape != (self.k, self.k):
            raise ValueError(f"the mix is {small.shape}, want {self.k} by {self.k}")
        self.register_buffer("signs", torch.from_numpy(signs.astype(np.float32)).to(device))
        self.register_buffer("small", torch.from_numpy(small.astype(np.float32)).to(device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`Q x` for one or many rows, `(…, n)` in and out.

        `Q = (Q_odd ⊗ H_m) D`: the sign flip, the transform per group, then the mix
        across groups.
        """
        v = (x * self.signs).reshape(*x.shape[:-1], self.k, self.m)
        v = wht_torch(v)
        if self.k > 1:
            v = torch.einsum("gt,...tm->...gm", self.small, v)
        return v.reshape(*x.shape)


class FusedTetraLinear(LlvqLinear):
    """One Tetra record, resident compressed, multiplied by the served kernel."""

    def __init__(self, desc: dict, bias: torch.Tensor | None = None):
        super().__init__(desc["d_out"], desc["d_in"], bias)
        self.desc = desc
        self.rotation: Rotation | None = None
        self._tile = 0
        for name in ("words", "gscale", "rscale", "tail"):
            setattr(self, f"_{name}", None)

    @classmethod
    def from_loaded(cls, loaded, rotation: Rotation | None, tile: int, device: str):
        """Build the resident form from a `TetraLinear` the loader just filled.

        The transcode from the disk stream to the served layout happens here, once
        per projection, and the loaded buffers are dropped with the module.
        """
        from .tetra import split_stream

        # Loading the extension is what registers `torch.ops.llvq`, and it is
        # cached: a module that arms without it would fail at its first forward
        # with a missing attribute rather than here.
        metal._extension()
        d = loaded.desc
        n = d["d_out"] * d["nblocks"]
        codes = loaded.codes.detach().cpu().numpy()
        labels, gains = split_stream(codes, n, d["index_bits"], d["gain_bits"])
        words = metal.to_tetra48(labels, gains, d["d_out"], d["nblocks"])
        centroids = loaded.centroids.detach().cpu().numpy()
        if centroids.shape != (2,):
            raise ValueError(f"{d['prefix']}: {centroids.shape} centroids, the kernel binds two")
        tail = (loaded.tail.detach().cpu().numpy().astype(np.float16).reshape(-1)
                if d["tail_cols"] else np.zeros(1, dtype=np.float16))
        out = cls(d, loaded.bias.detach() if loaded.bias is not None else None)
        out.arm(
            {
                "words": torch.from_numpy(words.reshape(-1)).to(device),
                "gscale": torch.from_numpy(centroids.astype(np.float32)).to(device),
                "rscale": torch.from_numpy(
                    loaded.row_scales.detach().cpu().numpy().astype(np.float32)
                ).to(device),
                "tail": torch.from_numpy(tail).to(device),
            },
            rotation,
            tile,
        )
        return out

    def arm(self, buffers: dict, rotation: Rotation | None, tile: int) -> None:
        """Take the device buffers this projection will be dispatched with."""
        self._words = buffers["words"]
        self._gscale = buffers["gscale"]
        self._rscale = buffers["rscale"]
        self._tail = buffers["tail"]
        self.rotation = rotation
        self._tile = tile

    def resident_bytes(self) -> int:
        return sum(t.numel() * t.element_size()
                   for t in (self._words, self._gscale, self._rscale, self._tail))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._words is None:
            raise RuntimeError("this projection was never armed with its device buffers")
        d = self.desc
        shape = x.shape
        flat = x.reshape(-1, d["d_in"]).to(torch.float32)
        if self.rotation is not None:
            flat = self.rotation(flat)
        flat = flat.contiguous()
        # One dispatch a token: the kernel takes a vector, and a batched fallback
        # would need the weight materialized.
        out = [
            torch.ops.llvq.tv_tetra48(
                self._words, *metal.shared_tables(x.device.type),
                self._gscale, metal.invnorm_on(x.device.type), self._rscale, self._tail,
                flat[i], d["d_out"], d["nblocks"], d["tail_cols"],
                metal.stride_u32(d["nblocks"]), self._tile, metal.tiled_source(self._tile),
            )
            for i in range(flat.shape[0])
        ]
        y = torch.stack(out).reshape(*shape[:-1], d["d_out"])
        if self.bias is not None:
            y = y + self.bias
        return y.to(x.dtype)
