"""The fused arm: the rotation in torch, the tables, and the kernel's refusals.

The gate over the 4B is the tokens, and it needs the 1.4 GB object. What runs here
is everything that does not: the invnorm table against its definition, the torch
transform against the f64 host one, and the guards the kernel has no room to carry
itself.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from llvqhf.dequant import wht_rows
from llvqhf.fused import Rotation, invnorm_table, wht_torch

FIXTURE = Path(__file__).parent / "fixtures" / "tiny"

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="the fused arm runs on Metal, and this host has no MPS device",
)


def test_invnorm_is_one_over_the_root_of_sixteen_m():
    t = invnorm_table()
    assert t.shape == (32,) and t.dtype == np.float32
    assert t[0] == 0.0, "the origin block has no shell and the kernel masks to it"
    for m in (1, 4, 26, 31):
        assert t[m] == np.float32(1.0 / np.sqrt(np.float64(16 * m)))


def test_the_torch_transform_is_the_host_one():
    """f32 against the f64 butterfly: the same transform, to f32 precision."""
    for n in (8, 64, 512):
        v = np.random.default_rng(n).standard_normal((3, n))
        want = v.copy()
        wht_rows(want)
        # `.float()` before the device, not after: MPS refuses an f64 tensor
        # outright, which is the same wall that shaped this whole stage.
        got = wht_torch(torch.from_numpy(v).float().to("mps")).cpu().numpy()
        assert np.allclose(got, want, atol=1e-5), n


def test_the_rotation_preserves_norms():
    """`Q` is orthogonal, so `‖Qx‖ = ‖x‖`. n = 40 exercises the odd factor."""
    n, k = 40, 5
    rng = np.random.default_rng(7)
    signs = rng.choice([-1.0, 1.0], size=n)
    q, _ = np.linalg.qr(rng.standard_normal((k, k)))
    rot = Rotation(signs, np.ascontiguousarray(q), "mps")
    x = torch.from_numpy(rng.standard_normal((4, n)).astype(np.float32)).to("mps")
    y = rot(x)
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-4)


def test_a_d_out_that_is_not_a_multiple_of_eight_is_refused():
    """The kernel carries no row guard: 32 threads a row, 256 a threadgroup."""
    from llvqhf import metal

    metal._extension()
    tables = metal.shared_tables("mps")
    d_out, nblocks = 4, 3
    stride = metal.stride_u32(nblocks)
    with pytest.raises(RuntimeError, match="not a multiple of 8"):
        torch.ops.llvq.tv_tetra48(
            torch.zeros(d_out * stride * 4, dtype=torch.uint8, device="mps"), *tables,
            torch.zeros(2, device="mps"), metal.invnorm_on("mps"),
            torch.zeros(d_out, device="mps"),
            torch.zeros(1, dtype=torch.float16, device="mps"),
            torch.zeros(nblocks * 24, device="mps"),
            d_out, nblocks, 0, stride, 64, metal.tiled_source(64),
        )


def test_a_stream_of_the_wrong_length_is_refused():
    from llvqhf import metal

    metal._extension()
    tables = metal.shared_tables("mps")
    d_out, nblocks = 8, 3
    stride = metal.stride_u32(nblocks)
    with pytest.raises(RuntimeError, match="which wants"):
        torch.ops.llvq.tv_tetra48(
            torch.zeros(d_out * stride * 4 - 8, dtype=torch.uint8, device="mps"), *tables,
            torch.zeros(2, device="mps"), metal.invnorm_on("mps"),
            torch.zeros(d_out, device="mps"),
            torch.zeros(1, dtype=torch.float16, device="mps"),
            torch.zeros(nblocks * 24, device="mps"),
            d_out, nblocks, 0, stride, 64, metal.tiled_source(64),
        )


def test_a_tail_in_f32_is_refused():
    """The kernel binds the tail as `half`. An f32 buffer would be read at twice
    its stride and give plausible wrong sums."""
    from llvqhf import metal

    metal._extension()
    tables = metal.shared_tables("mps")
    d_out, nblocks = 8, 3
    stride = metal.stride_u32(nblocks)
    with pytest.raises(RuntimeError, match="reads the tail as half"):
        torch.ops.llvq.tv_tetra48(
            torch.zeros(d_out * stride * 4, dtype=torch.uint8, device="mps"), *tables,
            torch.zeros(2, device="mps"), metal.invnorm_on("mps"),
            torch.zeros(d_out, device="mps"),
            torch.zeros(d_out * 8, device="mps"),
            torch.zeros(nblocks * 24 + 8, device="mps"),
            d_out, nblocks, 8, stride, 64, metal.tiled_source(64),
        )


def test_the_tile_is_defined_in_the_source_the_op_compiles():
    from llvqhf import metal

    assert "#define LLVQ_TILE_BLOCKS 64u" in metal.tiled_source(64)
    assert "#define LLVQ_TILE_BLOCKS 32u" in metal.tiled_source(32)
    with pytest.raises(ValueError, match="must be positive"):
        metal.tiled_source(0)
