"""The served Tetra shader as a torch op on Metal.

Stage 2 of `docs/plan-transformers.md`. The op is the shader's own decoder entry
point, `tetra48_probe`, built once into a torch extension:

    from llvqhf.metal import tetra_decode
    points = tetra_decode(codes, d_out, nblocks)   # int8 [d_out·nblocks, 24]

## What it computes, and what it deliberately does not

Points, and nothing else. The gain, the row scale, the tail and the un-rotation
stay on the host, because that chain is f64 by the format's design and Metal has
no f64. An op that returned weights would have to be judged against a reference
it cannot equal, and the stage 2 prereg says so rather than widening a tolerance.

## The two layouts

`llvq-artifact` packs a block MSB-first on disk; the shader reads the served
`tetra48` layout, little-endian and row-strided with a pad
(`llvq-artifact/src/tetra48.rs`). [`to_tetra48`] is the transcode between them,
and it is covered by the gate: get it wrong and the points differ.

## The build

`torch.utils.cpp_extension.load` compiles `csrc/tetra_decode.mm` once and caches
it under `~/.cache/torch_extensions`. It needs the command line tools, which is
what a Metal kernel costs on a Mac. The MSL source is read from the package's
data directory, a copy of the repository's served shader whose sha256 the tables
file records.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

DATA = Path(__file__).parent / "data"
SHADER = DATA / "llvq_tetra48.metal"
CONSTANTS = DATA / "tetra-tables.json"
CSRC = Path(__file__).parent / "csrc" / "tetra_decode.mm"
CSRC_CUDA = Path(__file__).parent / "csrc" / "tetra_cuda.cu"


def shader_source(name: str = "llvq_tetra48.metal") -> str:
    """One of the shipped shaders, with its digest checked against the tables file.

    The shaders ship as copies of the repository's. Checking here means a copy
    that drifted is refused at the first dispatch rather than computing plausible
    wrong numbers.
    """
    import hashlib

    shaders = json.loads(CONSTANTS.read_text())["shaders"]
    if name not in shaders:
        raise KeyError(f"{name} is not one of the shipped shaders, {sorted(shaders)}")
    raw = (DATA / name).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != shaders[name]["sha256"]:
        raise ValueError(
            f"{name} is not the shader the tables were dumped with:\n"
            f"  recorded {shaders[name]['sha256']}\n  shipped  {got}"
        )
    return raw.decode()


@lru_cache(maxsize=1)
def q4_source() -> str:
    """The int4 shader, which needs no define: its group size is in the file."""
    return shader_source("tv_q4_h.metal")


def extension_for(device: str):
    """The extension that registers the op namespace of `device`.

    Metal and CUDA are two files and two namespaces, `torch.ops.llvq` and
    `torch.ops.llvq_cuda`, because they are two kernels with two signatures: the
    CUDA one writes f16 and takes its tile through a compile flag where Metal
    takes the source. One name for both would have hidden that.
    """
    if device == "cuda":
        return _extension_cuda()
    return _extension()


@lru_cache(maxsize=1)
def _extension_cuda():
    """Compile `csrc/tetra_cuda.cu` against the served kernel, with nvcc.

    The include path is the repository's `llvq-llm/kernels`, and that file's own
    `#include "../../llvq-cuda/kernels/..."` resolve from there, so the tree has
    to be the repository's shape. `TILE_BLOCKS` is the kernel's define, not
    Metal's `LLVQ_TILE_BLOCKS`, and the shared bytes the caller passes have to
    match it.
    """
    from torch.utils.cpp_extension import load

    if not torch.cuda.is_available():
        raise RuntimeError("this op runs on CUDA, and no device is available")
    root = Path(os.environ.get("LLVQ_REPO", Path(__file__).resolve().parents[2]))
    kernels = root / "llvq-llm" / "kernels"
    if not (kernels / "tv_tetra48_h.cu").exists():
        raise FileNotFoundError(
            f"{kernels / 'tv_tetra48_h.cu'} is missing. Set LLVQ_REPO to the repository "
            "root: the served kernel is included from it rather than copied"
        )
    tile = int(os.environ.get("LLVQ_HF_TILE", "64"))
    return load(
        name="llvqhf_cuda",
        sources=[str(CSRC_CUDA)],
        extra_include_paths=[str(kernels)],
        # C++20 on both, for the reason the Metal arm already met and this one
        # repeated anyway: torch's headers use `requires` clauses and refuse to
        # parse under c++17, with `#error C++20 or later compatible compiler`.
        extra_cflags=["-std=c++20"],
        extra_cuda_cflags=["-std=c++20", f"-DTILE_BLOCKS={tile}u", "--fmad=true"],
        is_python_module=False,
        verbose=False,
    )


@lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import load

    if not torch.backends.mps.is_available():
        raise RuntimeError("this op runs on Metal, and no MPS device is available")
    return load(
        name="llvqhf_metal",
        sources=[str(CSRC)],
        # C++20: torch 2.14 headers use `requires` clauses, so c++17 does not
        # parse them.
        extra_cflags=["-std=c++20", "-ObjC++", "-fobjc-arc"],
        extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
        # Not a Python module: the op is registered with TORCH_LIBRARY, so torch
        # loads the library and `torch.ops.llvq` appears. Asking for a Python
        # module would look for a `PyInit_` this file has no reason to define.
        is_python_module=False,
        verbose=False,
    )


@lru_cache(maxsize=4)
def tiled_source(tile: int) -> str:
    """The shader with `LLVQ_TILE_BLOCKS` defined, as `llvq_llm::fused_metal` compiles it.

    One number, passed once: the same `tile` sizes the threadgroup buffer on the
    host side, so the define and the allocation cannot disagree.
    """
    if tile <= 0:
        raise ValueError(f"the tile must be positive, got {tile}")
    return f"#define LLVQ_TILE_BLOCKS {tile}u\n{shader_source()}"


@lru_cache(maxsize=2)
def invnorm_on(device: str) -> "torch.Tensor":
    """`1/sqrt(16m)` in f32, 32 entries, `invnorm[0] = 0`, on the device."""
    from .fused import invnorm_table

    return torch.from_numpy(invnorm_table()).to(device)


def shared_tables(device: str):
    """The four decoder tables as byte blobs, loaded once per device."""
    return _tables_on_device(device)


@lru_cache(maxsize=4)
def _tables_on_device(device: str):
    """The four decoder tables as byte blobs on the device, loaded once.

    Byte blobs and not typed tensors: the shader reads `rows` as `uint*` and
    `branches` as `ushort*`, and the layout is decided once in
    `llvq_llm::hfpack::tetra_tables`, not twice.
    """
    from safetensors.numpy import load_file

    t = load_file(DATA / "tetra-tables.safetensors")
    def blob(a: np.ndarray, dtype: str) -> torch.Tensor:
        return torch.from_numpy(
            np.ascontiguousarray(a, dtype=dtype).view(np.uint8).reshape(-1).copy()
        ).to(device)

    return (
        blob(t["rows"], "<u4"),
        blob(t["prefixes_flat"], "<u1"),
        blob(t["branches_u16"], "<u2"),
        blob(t["suffixes_flat"], "<u1"),
    )


def stride_u32(nblocks: int) -> int:
    """`round_up(6·nblocks, 8) / 4`, the served row stride.

    Rounding to eight bytes and not four is what covers the last block's two-word
    read window (`llvq_artifact::tetra48::stride_u32`).
    """
    return -(-(6 * nblocks) // 8) * 2


def to_tetra48(labels: np.ndarray, gains: np.ndarray, d_out: int, nblocks: int) -> np.ndarray:
    """The served layout from a record's labels and gains.

    Two conventions meet here and neither is the other's byte order:

    * the disk packs 47 index bits then one gain bit, MSB-first, so the six bytes
      of a block read big-endian give `label << 1 | gain`;
    * the Tetra word is `label | gain << 47` (`llvq-search/src/tetra/word.rs`),
      and the shader reads it little-endian at bit offset `48·j` of its row
      (`f1r_load`).

    So a byte reversal is not the transcode. The gain bit has to move from the
    bottom of the disk value to bit 47 of the word, and reversing bytes alone
    decodes plausible wrong points, which is what the first attempt at this
    function did.
    """
    n = d_out * nblocks
    if labels.size != n or gains.size != n:
        raise ValueError(f"{labels.size} labels and {gains.size} gains for {n} blocks")
    if labels.size and int(labels.max()) >> 47:
        raise ValueError("a label above 47 bits would collide with the gain bit at 47")
    if gains.size and int(gains.max()) > 1:
        raise ValueError("a Tetra word carries one gain bit, so a gain above 1 is not one")
    word = labels.astype(np.uint64) | (gains.astype(np.uint64) << np.uint64(47))
    little = np.empty((n, 6), dtype=np.uint8)
    for i in range(6):
        little[:, i] = ((word >> np.uint64(8 * i)) & np.uint64(0xFF)).astype(np.uint8)
    stride = stride_u32(nblocks)
    out = np.zeros((d_out, stride * 4), dtype=np.uint8)
    out[:, : nblocks * 6] = little.reshape(d_out, nblocks * 6)
    return out


def tetra_decode(codes: np.ndarray, d_out: int, nblocks: int, index_bits: int = 47,
                 gain_bits: int = 1, device: str = "mps") -> torch.Tensor:
    """`(d_out·nblocks, 24)` int8 points in artifact order, decoded on Metal.

    `codes` is a record's stream as the file stores it. The cast to int8 is exact:
    the shader writes integers and the codebook's largest coordinate is 10.
    """
    from .tetra import split_stream

    ext = _extension()  # noqa: F841  (registers torch.ops.llvq)
    rows, prefixes, branches, suffixes = _tables_on_device(device)
    labels, gains = split_stream(codes, d_out * nblocks, index_bits, gain_bits)
    words = torch.from_numpy(to_tetra48(labels, gains, d_out, nblocks)).to(device)
    points, _shell = torch.ops.llvq.tetra_decode(
        words.reshape(-1), rows, prefixes, branches, suffixes,
        d_out, nblocks, stride_u32(nblocks), shader_source(),
    )
    return points.to(torch.int8)
