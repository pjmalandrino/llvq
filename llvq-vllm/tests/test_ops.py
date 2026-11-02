"""The ops are what `torch.compile` and CUDA graph capture need them to be.

vLLM compiles the model with the token count dynamic and refuses a recompile, so
three properties are held here on the CPU implementations:

* each op's fake implementation agrees with the real one on shape, dtype and
  strides, and its schema matches what it does (`torch.library.opcheck`);
* a layer's forward traces into one graph, no break, with M symbolic;
* calling it at another M reuses that graph instead of recompiling.

What only a card can say, that the CUDA kernels capture into a graph and replay
the same bits, is a gate of the prereg, not of this file.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import Source, build_layer
from llvq_vllm import buffers as B
from llvq_vllm.layer import _buf, _rot_buf, forward

OPS = torch.ops.llvq_vllm


@pytest.fixture(scope="module")
def gate_up(mini, mini_qc):
    return build_layer(Source(mini, mini_qc), "model.layers.0.mlp.gate_up_proj", rows_max=2)


@pytest.fixture(scope="module")
def down(mini, mini_qc):
    return build_layer(Source(mini, mini_qc), "model.layers.0.mlp.down_proj", rows_max=2)


def tetra_args(holder, armed, x, shard=0):
    m = armed.shards[shard]
    r = armed.rotations[0]
    rot = (x, getattr(holder, _rot_buf(r.key, "bits")), getattr(holder, _rot_buf(r.key, "small")),
           r.n, r.m, r.k, r.inv, r.threads)
    mm = (OPS.rotate(*rot), getattr(holder, _buf(m.index, "words")),
          getattr(holder, _buf(m.index, "gscale")), getattr(holder, _buf(m.index, "rscale")),
          getattr(holder, _buf(m.index, "tail")), m.d_out, m.nblocks, m.tail_w, m.stride,
          armed.tile, armed.rows_max)
    return rot, mm


@pytest.mark.parametrize("m", [1, 5])
def test_the_rotation_and_tetra_ops_pass_opcheck(gate_up, m):
    holder, armed, plan = gate_up
    x = torch.randn(m, plan.d_in).half()
    rot, mm = tetra_args(holder, armed, x)
    torch.library.opcheck(OPS.rotate.default, rot)
    torch.library.opcheck(OPS.tetra_mm.default, mm)


@pytest.mark.parametrize("m", [1, 5])
def test_the_int4_op_passes_opcheck(down, m):
    holder, armed, plan = down
    s = armed.shards[0]
    args = (torch.randn(m, plan.d_in).half(), getattr(holder, _buf(0, "qweight")),
            getattr(holder, _buf(0, "scales")), getattr(holder, _buf(0, "biases")), s.d_out,
            s.d_in, s.gpr, armed.rows_max)
    torch.library.opcheck(OPS.q4_mm.default, args)


@pytest.mark.parametrize("m", [1, 5])
def test_the_vocabulary_ops_pass_opcheck(mini, m):
    q, s, b = (torch.from_numpy(np.array(mini.tensor(f"model.embed_tokens.{n}")))
               for n in ("qweight", "scales", "biases"))
    vocab, d = 64, 136
    gpr = -(-d // 64)
    torch.library.opcheck(OPS.emb_q4_gather.default, (torch.arange(m), q, s, b, d, gpr))
    torch.library.opcheck(OPS.emb_q4_mm.default,
                          (torch.randn(m, d).half(), q, s, b, vocab, d, gpr, 2))


def test_the_head_is_the_table_times_the_hidden_state(mini):
    table = mini.dequantize_raw("model.embed_tokens.weight").astype(np.float64)
    q, s, b = (torch.from_numpy(np.array(mini.tensor(f"model.embed_tokens.{n}")))
               for n in ("qweight", "scales", "biases"))
    h = torch.randn(3, 136).half()
    ref = h.double().numpy() @ table.T
    for rows_max in (1, 4):  # the dense path, then the row path
        y = OPS.emb_q4_mm(h, q, s, b, 64, 136, 3, rows_max)
        err = np.linalg.norm(y.double().numpy() - ref, axis=1) / np.linalg.norm(ref, axis=1)
        assert err.max() < 1e-2


def test_a_layer_traces_into_one_graph_and_does_not_recompile_with_m(gate_up):
    holder, armed, plan = gate_up
    torch._dynamo.reset()
    fn = torch.compile(lambda x: forward(holder, armed, x), backend="aot_eager",
                       fullgraph=True, dynamic=True)
    a = torch.randn(3, plan.d_in).half()
    b = torch.randn(9, plan.d_in).half()
    want_a, want_b = forward(holder, armed, a), forward(holder, armed, b)
    assert torch.equal(fn(a), want_a)
    with torch.compiler.set_stance("fail_on_recompile"):
        assert torch.equal(fn(b), want_b)


def test_prepare_caches_the_tables_before_any_forward():
    """What a capture must not do itself: the first, host-to-device, read of a table."""
    from llvq_vllm import ops

    B.decode_tables.cache_clear()
    ops.prepare(torch.device("cpu"))
    assert B.decode_tables.cache_info().currsize == 1
    before = B.decode_tables.cache_info().hits
    B.decode_tables("cpu")
    assert B.decode_tables.cache_info().hits == before + 1


def test_rotate_takes_f16_whatever_it_is_handed(gate_up):
    """The kernel reads f16, so the CPU reference rounds an f32 activation the same way."""
    holder, armed, plan = gate_up
    x = torch.randn(2, plan.d_in)
    rot, _ = tetra_args(holder, armed, x)
    rb = B.RotationBuffers("k", *rot[1:3], *rot[3:7])
    assert torch.equal(OPS.rotate(*rot), B.rotate_reference(x.half(), rb))
