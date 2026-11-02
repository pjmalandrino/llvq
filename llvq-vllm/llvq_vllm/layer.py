"""One layer, armed and run, with no vLLM in sight.

`vllm_glue` is a thin adapter over this file, so the forward a vLLM layer runs
is the forward the Mac tests run, on the CPU references of `ops.py`.

## What a layer holds

The device buffers are registered on the holder module as non-persistent
buffers, `llvq_<i>_words` and the like, so vLLM's memory profile counts them and
`torch.compile` reads them as module state. Beside them sits a tuple of frozen
metadata, plain ints and buffer names, which is all `forward` branches on.

## One rotation per key

The Tetra shards of a layer that share a rotation key read one rotated
activation. On the sealed 4B that is q with k, and gate with up: the
`LLVQ_ROT_SHARE=1` of the served config, which vLLM's fused layers make free,
since q, k and v are one call. The int4 shards read the activation as it comes,
in the natural basis.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from . import ops  # noqa: F401  (defines torch.ops.llvq_vllm)
from .buffers import RotationBuffers, tetra48_words
from .plan import INT4, TETRA, LayerPlan, ParamSpec
from llvq_tetra.metal import stride_u32


# The buffer names are built at arm time and carried as strings. Building them
# in `forward` would format an int inside traced code, and torch 2.11's Dynamo,
# the image's, treats such an int as symbolic and refuses the f-string.


@dataclass(frozen=True)
class TetraMeta:
    index: int
    d_out: int
    nblocks: int
    tail_w: int
    stride: int
    rotation: str
    bufs: tuple[str, str, str, str]  # words, gscale, rscale, tail


@dataclass(frozen=True)
class Int4Meta:
    index: int
    d_out: int
    d_in: int
    gpr: int
    bufs: tuple[str, str, str]  # qweight, scales, biases


@dataclass(frozen=True)
class RotMeta:
    key: str
    n: int
    m: int
    k: int
    inv: float
    threads: int
    bufs: tuple[str, str]  # sign bits, Q_odd


@dataclass(frozen=True)
class Armed:
    """Everything `forward` needs that is not a tensor."""

    shards: tuple
    rotations: tuple[RotMeta, ...]
    d_in: int
    d_out: int
    tile: int
    rows_max: int


def _buf(i: int, name: str) -> str:
    return f"llvq_{i}_{name}"


def _rot_buf(key: str, name: str) -> str:
    return f"llvq_r_{key}_{name}"


def slot_view(flat: torch.Tensor, spec: ParamSpec, index: int) -> torch.Tensor:
    """One shard's tensor inside a layer's flat parameter, in its checkpoint shape."""
    slot = spec.slots[index]
    return flat.reshape(-1)[slot.offset: slot.offset + slot.numel].reshape(slot.shape)


def arm(holder: nn.Module, plan: LayerPlan, raw: dict[str, torch.Tensor],
        specs: list[ParamSpec], rotations: dict[str, RotationBuffers], tile: int,
        rows_max: int) -> Armed:
    """Build the served buffers from the loaded tensors and register them on `holder`.

    `raw` maps a spec name to its loaded flat tensor, on the device the layer
    serves from. Widths are the kernels', not the file's: the gain centroids and
    the row scales narrowed to f32, the tail to f16, as `FusedTetraLinear` does.
    """
    by_name = {s.name: s for s in specs}
    metas = []
    for s in plan.shards:
        i = s.index
        if s.kind == TETRA:
            codes = slot_view(raw["codes"], by_name["codes"], i)
            words = tetra48_words(codes.reshape(-1), s.d_out, s.nblocks)
            gscale = slot_view(raw["centroids"], by_name["centroids"], i).to(torch.float32)
            rscale = slot_view(raw["row_scales"], by_name["row_scales"], i).to(torch.float32)
            if s.tail_cols:
                tail = slot_view(raw["tail"], by_name["tail"], i).to(torch.float16).reshape(-1)
            else:
                tail = torch.zeros(1, dtype=torch.float16, device=codes.device)
            names = tuple(_buf(i, n) for n in ("words", "gscale", "rscale", "tail"))
            for name, t in zip(names, (words, gscale, rscale, tail)):
                holder.register_buffer(name, t.contiguous(), persistent=False)
            metas.append(TetraMeta(i, s.d_out, s.nblocks, s.tail_cols, stride_u32(s.nblocks),
                                   s.rotation, names))
        elif s.kind == INT4:
            names = tuple(_buf(i, n) for n in ("qweight", "scales", "biases"))
            for spec_name, name in zip(("qweight", "scales", "biases"), names):
                t = slot_view(raw[spec_name], by_name[spec_name], i)
                holder.register_buffer(name, t.reshape(-1).contiguous().clone(), persistent=False)
            metas.append(Int4Meta(i, s.d_out, s.d_in, s.gpr, names))
        else:
            raise ValueError(f"{plan.prefix}: unknown shard kind {s.kind!r}")
    rots = []
    for key in plan.rotations:
        rb = rotations[key]
        if rb.n != plan.d_in:
            raise ValueError(f"{plan.prefix}: rotation {key} is {rb.n} wide for d_in {plan.d_in}")
        names = (_rot_buf(key, "bits"), _rot_buf(key, "small"))
        holder.register_buffer(names[0], rb.signbits, persistent=False)
        holder.register_buffer(names[1], rb.small, persistent=False)
        rots.append(RotMeta(key, rb.n, rb.m, rb.k, rb.inv, rb.threads, names))
    return Armed(tuple(metas), tuple(rots), plan.d_in, plan.d_out, tile, rows_max)


def arm_loaded(holder: nn.Module, plan: LayerPlan, specs: list[ParamSpec],
               rotations: dict[str, RotationBuffers], tile: int, rows_max: int,
               prefix: str) -> Armed:
    """Arm `holder` from the parameters its loaders filled, then drop them.

    What `process_weights_after_loading` does, without vLLM. A hole in any
    parameter is refused before a buffer is built. The loaded tensors are spent
    afterwards: the served buffers hold the same information in the kernels'
    widths. Rotation tables are not dropped, since another layer may read them
    from this owner after it.
    """
    from .params import loaded_flats

    raw = loaded_flats(holder, specs, prefix)
    armed = arm(holder, plan, raw, specs, rotations, tile, rows_max)
    for spec in specs:
        delattr(holder, spec.name)
    return armed


def forward(holder: nn.Module, armed: Armed, x: torch.Tensor,
            bias: torch.Tensor | None = None) -> torch.Tensor:
    """`y = W x` for every row of `x`, `[..., d_in]` in, `[..., d_out]` f16 out."""
    lead = x.shape[:-1]
    flat = x.reshape(-1, armed.d_in)
    rotated = {}
    for r in armed.rotations:
        bits, small = (getattr(holder, n) for n in r.bufs)
        rotated[r.key] = torch.ops.llvq_vllm.rotate(flat, bits, small, r.n, r.m, r.k, r.inv,
                                                    r.threads)
    outs = []
    for m in armed.shards:
        if isinstance(m, TetraMeta):
            words, gscale, rscale, tail = (getattr(holder, n) for n in m.bufs)
            outs.append(torch.ops.llvq_vllm.tetra_mm(
                rotated[m.rotation], words, gscale, rscale, tail, m.d_out, m.nblocks, m.tail_w,
                m.stride, armed.tile, armed.rows_max))
        else:
            wq, scales, biases = (getattr(holder, n) for n in m.bufs)
            outs.append(torch.ops.llvq_vllm.q4_mm(flat, wq, scales, biases, m.d_out, m.d_in,
                                                  m.gpr, armed.rows_max))
    y = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)
    if bias is not None:
        y = y + bias.to(y.dtype)
    return y.reshape(*lead, armed.d_out)


def resident_bytes(holder: nn.Module) -> int:
    """Device bytes of the served buffers on `holder`, measured and not computed."""
    return sum(b.numel() * b.element_size()
               for name, b in holder.named_buffers(recurse=False) if name.startswith("llvq_"))
