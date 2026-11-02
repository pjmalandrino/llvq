"""The served buffers against the numpy reference of `llvq-tetra`.

The transcode and the decode are integer and are held to identity. The rotation
and the dequantization are f32 and are held to the per-row bar of the CUDA
stage, 1e-2, against the f64 reconstruction that matched `decode_matrix` digest
for digest.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import row_error
from llvq_tetra import metal
from llvq_tetra.dequant import dequantize_int4
from llvq_tetra.fused import Rotation
from llvq_tetra.tetra import TetraTables, split_stream
from llvq_vllm import buffers as B

TETRA_RECORDS = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
]


def record(pm, name):
    d = pm.records[name]
    return d, pm.tensor(f"{d['prefix']}.codes")


@pytest.mark.parametrize("name", TETRA_RECORDS)
def test_the_device_transcode_is_the_numpy_one(mini, name):
    d, codes = record(mini, name)
    labels, gains = split_stream(codes, d["d_out"] * d["nblocks"], 47, 1)
    want = metal.to_tetra48(labels, gains, d["d_out"], d["nblocks"]).reshape(-1)
    got = B.tetra48_words(torch.from_numpy(codes.copy()), d["d_out"], d["nblocks"])
    assert np.array_equal(got.numpy(), want)


@pytest.mark.parametrize("name", TETRA_RECORDS)
def test_reading_the_served_words_gives_back_the_labels(mini, name):
    d, codes = record(mini, name)
    labels, gains = split_stream(codes, d["d_out"] * d["nblocks"], 47, 1)
    words = B.tetra48_words(torch.from_numpy(codes.copy()), d["d_out"], d["nblocks"])
    got_l, got_g = B.words_to_labels(words, d["d_out"], d["nblocks"])
    assert np.array_equal(got_l.numpy().astype(np.uint64), labels)
    assert np.array_equal(got_g.numpy(), gains.astype(np.int64))


@pytest.mark.parametrize("name", TETRA_RECORDS)
def test_the_torch_decode_is_the_numpy_decode(mini, name):
    d, codes = record(mini, name)
    labels, _ = split_stream(codes, d["d_out"] * d["nblocks"], 47, 1)
    want = TetraTables.load().decode(labels)
    got = B.decode_points(torch.from_numpy(labels.astype(np.int64)), B.decode_tables("cpu"))
    assert np.array_equal(got.numpy(), want)


@pytest.mark.parametrize("key", ["136_0000000000005eed", "128_0000000000005eed"])
def test_the_kernel_rotation_is_the_rotation_stage_4_served(mini, key):
    """`rotate_reference` against `llvq_tetra.fused.Rotation`, which gave 256 ids of 256."""
    signs, small = mini.rotation(key)
    rb = B.rotation_buffers(key, signs, small, "cpu")
    x = torch.randn(5, rb.n, generator=torch.Generator().manual_seed(1))
    want = Rotation(signs, small, "cpu")(x)
    got = B.rotate_reference(x, rb)
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-5)


def test_the_rotation_tables_follow_the_rust_builder(mini):
    """`RotationTables::build`: a set bit is a negative sign, `Q_odd` padded, `1/√m` from f64."""
    signs, small = mini.rotation("136_0000000000005eed")
    rb = B.rotation_buffers("k", signs, small, "cpu")
    assert (rb.n, rb.m, rb.k) == (136, 8, 17)
    bits = rb.signbits.numpy().view(np.uint32)
    for i, s in enumerate(signs):
        assert bool((bits[i >> 5] >> (i & 31)) & 1) == (s < 0)
    padded = rb.small.numpy().reshape(B.ROT_KMAX, B.ROT_KMAX)
    assert np.array_equal(padded[:17, :17], small.astype(np.float32))
    assert not padded[17:, :].any() and not padded[:, 17:].any()
    assert rb.inv == float(np.float32(1.0 / np.sqrt(8.0)))
    assert rb.threads == 256


def test_a_rotation_past_kmax_is_refused():
    n = 512 * 33
    with pytest.raises(ValueError, match="above KMAX"):
        B.rotation_buffers("wide", np.ones(n), np.eye(33), "cpu")


@pytest.mark.parametrize("name", TETRA_RECORDS)
def test_rotated_weights_times_rotated_activation_is_the_dense_product(mini, name):
    """`W x = W' · rot(x)`: the identity the fused path rests on, end to end."""
    d, codes = record(mini, name)
    words = B.tetra48_words(torch.from_numpy(codes.copy()), d["d_out"], d["nblocks"])
    prefix = d["prefix"]
    gscale = torch.from_numpy(mini.tensor(f"{prefix}.centroids").astype(np.float32))
    rscale = torch.from_numpy(mini.tensor(f"{prefix}.row_scales").astype(np.float32))
    tail = torch.from_numpy(mini.tensor(f"{prefix}.tail").astype(np.float16))
    w_rot = B.tetra_dequant(words, gscale, rscale, tail, d["d_out"], d["nblocks"],
                            d["tail_cols"], torch.float32)
    rb = B.rotation_buffers(d["rotation"], *mini.rotation(d["rotation"]), "cpu")
    x = torch.randn(3, d["d_in"], generator=torch.Generator().manual_seed(2))
    y = B.rotate_reference(x, rb) @ w_rot.t()
    ref = x.double().numpy() @ mini.dequantize(name).astype(np.float64).T
    assert row_error(y, ref) < 1e-2


def test_the_int4_dequant_is_the_numpy_one(mini):
    d = mini.records["model.layers.0.mlp.down_proj.weight"]
    q, s, b = (mini.tensor(f"{d['prefix']}.{n}") for n in ("qweight", "scales", "biases"))
    want = dequantize_int4(q, s, b, d["d_out"], d["d_in"], 128)
    got = B.int4_dequant(*(torch.from_numpy(a.copy()) for a in (q, s, b)), d["d_out"], d["d_in"],
                         128, torch.float32)
    assert np.array_equal(got.numpy(), want)


def test_gathered_vocabulary_rows_are_the_dequantized_table_in_f16(mini):
    table = mini.dequantize_raw("model.embed_tokens.weight")
    q, s, b = (torch.from_numpy(mini.tensor(f"model.embed_tokens.{n}").copy())
               for n in ("qweight", "scales", "biases"))
    ids = torch.tensor([0, 7, 63, 7])
    got = B.int4_rows(q, s, b, ids, table.shape[1], 64)
    assert np.array_equal(got.numpy(), table[ids.numpy()].astype(np.float16))
