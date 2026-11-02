"""Every layer of the fixture, armed and run on the CPU references, against the f64 dense.

Both paths of each op run: the row path (up to `rows_max` tokens, the served
kernels on a card) and the dense path (dequantize, then an f16 GEMM). Each is
held to the per-row bar of the CUDA stage, 1e-2, which the defects this repo
has met clear by far: a dropped tail costs 3.5 to 8.8 % a row, a wrong rotation
the whole row.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import LAYERS, Source, build_layer, dense_of, mixed_source, row_error
from llvq_vllm.layer import forward, resident_bytes

ROWS_MAX = 4


def activation(m: int, d_in: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(m, d_in, generator=g).to(torch.float16)


@pytest.mark.parametrize("prefix", LAYERS)
@pytest.mark.parametrize("m", [1, ROWS_MAX, ROWS_MAX + 3])
def test_a_fixture_layer_is_its_dense_product(mini, mini_qc, prefix, m):
    src = Source(mini, mini_qc)
    holder, armed, plan = build_layer(src, prefix, rows_max=ROWS_MAX)
    x = activation(m, plan.d_in, seed=m)
    y = forward(holder, armed, x)
    assert y.dtype == torch.float16 and tuple(y.shape) == (m, plan.d_out)
    ref = x.double().numpy() @ dense_of(src, plan).astype(np.float64).T
    assert row_error(y, ref) < 1e-2


@pytest.mark.parametrize("m", [1, ROWS_MAX + 3])
def test_the_served_qkv_layout_is_its_dense_product(mini, mini_qc, m):
    """q and k Tetra on one rotation, v int4 on the raw activation, in that column order."""
    src = mixed_source(mini, mini_qc)
    holder, armed, plan = build_layer(src, "model.layers.0.self_attn.qkv_proj",
                                      rows_max=ROWS_MAX)
    assert [s.kind for s in plan.shards] == ["tetra", "tetra", "int4g128"]
    assert len(armed.rotations) == 1
    x = activation(m, plan.d_in, seed=10 + m)
    y = forward(holder, armed, x)
    ref = x.double().numpy() @ dense_of(src, plan).astype(np.float64).T
    assert row_error(y, ref) < 1e-2
    # The int4 columns are where v sits, after q and k, and are v's own product.
    v = slice(128 + 64, 128 + 64 + 64)
    assert row_error(y[:, v], ref[:, v]) < 1e-2


def test_the_row_path_and_the_dense_path_agree(mini, mini_qc):
    src = Source(mini, mini_qc)
    holder, armed, plan = build_layer(src, "model.layers.0.mlp.gate_up_proj", rows_max=ROWS_MAX)
    x = activation(ROWS_MAX + 1, plan.d_in, seed=3)
    dense = forward(holder, armed, x)  # M > rows_max: the dense path
    rows = torch.cat([forward(holder, armed, x[i:i + 1]) for i in range(x.shape[0])])
    assert row_error(dense, rows.double().numpy()) < 1e-2


def test_leading_dimensions_and_bias_pass_through(mini, mini_qc):
    src = Source(mini, mini_qc)
    holder, armed, plan = build_layer(src, "model.layers.0.self_attn.o_proj")
    x = activation(6, plan.d_in, seed=4).reshape(2, 3, plan.d_in)
    bias = torch.linspace(-1, 1, plan.d_out)
    y = forward(holder, armed, x, bias)
    assert tuple(y.shape) == (2, 3, plan.d_out)
    y0 = forward(holder, armed, x)
    # One f16 addition after the product, as vLLM's own linear layers add it.
    assert torch.equal(y, y0 + bias.half())


def test_the_resident_buffers_are_the_served_widths(mini, mini_qc):
    """6 bytes a block padded to 8-byte rows, f32 scales, an f16 tail, per Tetra shard."""
    src = Source(mini, mini_qc)
    holder, armed, plan = build_layer(src, "model.layers.0.mlp.gate_up_proj")
    stride_bytes = -(-(6 * 5) // 8) * 8  # 5 blocks a row
    per_shard = 256 * stride_bytes + 2 * 4 + 256 * 4 + 256 * 16 * 2
    rotation = 5 * 4 + 32 * 32 * 4  # 136 sign bits in 5 words, Q_odd padded
    assert resident_bytes(holder) == 2 * per_shard + rotation
    assert not [n for n, _ in holder.named_parameters()], "the loaded tensors are spent"
