"""`from_pretrained` on a whole model, which no test did before 2026-10-02.

The other fixture, `tiny`, describes nothing: a 4 by 88 record beside a config
that says `hidden_size` 96. It tests the packer field by field and it is not
touched. But it means every test here read tensors and none ever loaded a model,
and the registration defect of `__init__.py` lived in exactly that hole: four
stages of measurement, all through our own scripts, each importing `.quantizer`
by hand.

So `fixtures/mini` is a coherent one-layer Qwen3, 148 KB, written by
`the_mini_fixture_describes_a_whole_qwen3_layer` in `llvq-llm/tests/hfpack.rs`.
Its weights are random, so the logits mean nothing and nothing here asserts on
their values. What it can say is what a maintainer's CI needs to see: the file
loads with no key missing and none unexpected, every projection is replaced, and
a forward pass gives finite numbers of the right shape, on the CPU, with no
kernel compiled.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

# At module level, where a user would put it, and the first version of this file
# left it out. Run alone, the file then loaded the fixture with the method
# unregistered: `transformers` skipped the quantization, reinitialized every
# dense weight it found MISSING, and THE FORWARD PASS STILL PASSED, on random
# numbers. So a forward pass that runs proves nothing here. The gate is
# `missing_keys` and `unexpected_keys`.
import llvq_tetra  # noqa: E402, F401

MINI = Path(__file__).parent / "fixtures" / "mini"

HIDDEN, HEAD_DIM, Q_HEADS, KV_HEADS, INTER, VOCAB = 136, 32, 4, 2, 256, 64
PROJECTIONS = 7


@pytest.fixture(scope="module")
def loaded():
    if not MINI.exists():
        pytest.fail(
            f"{MINI} is missing; write it with "
            "`LLVQ_HF_MINI_FIXTURE=../llvq-tetra/tests/fixtures/mini "
            "cargo test -p llvq-llm --test hfpack`"
        )
    from transformers import AutoModelForCausalLM

    model, info = AutoModelForCausalLM.from_pretrained(
        MINI, dtype=torch.float32, output_loading_info=True
    )
    return model, info


def test_nothing_is_missing_and_nothing_is_unexpected(loaded):
    """The assertion the soft failure of 2026-10-01 would have tripped.

    With the method unregistered, `transformers` only warns, loads the model as
    dense, and reports every record UNEXPECTED and every dense weight MISSING.
    So these two lists are the gate, not the absence of an exception.
    """
    _, info = loaded
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        assert not info[key], f"{key}: {sorted(info[key])}"


def test_every_projection_was_replaced(loaded):
    from llvq_tetra.modules import Int4Linear, TetraLinear

    model, _ = loaded
    tetra = [n for n, m in model.named_modules() if isinstance(m, TetraLinear)]
    int4 = [n for n, m in model.named_modules() if isinstance(m, Int4Linear)
            and not isinstance(m, TetraLinear)]
    assert len(tetra) == 6, tetra
    assert len(int4) == 1, int4
    assert int4[0].endswith("mlp.down_proj"), int4
    assert len(tetra) + len(int4) == PROJECTIONS


def test_the_shapes_are_the_architectures(loaded):
    model, _ = loaded
    sa = model.model.layers[0].self_attn
    mlp = model.model.layers[0].mlp
    for mod, d_out, d_in in [
        (sa.q_proj, Q_HEADS * HEAD_DIM, HIDDEN),
        (sa.k_proj, KV_HEADS * HEAD_DIM, HIDDEN),
        (sa.v_proj, KV_HEADS * HEAD_DIM, HIDDEN),
        (sa.o_proj, HIDDEN, Q_HEADS * HEAD_DIM),
        (mlp.gate_proj, INTER, HIDDEN),
        (mlp.up_proj, INTER, HIDDEN),
        (mlp.down_proj, HIDDEN, INTER),
    ]:
        assert mod.weight.shape == (d_out, d_in), (type(mod).__name__, mod.weight.shape)


def test_the_weights_are_not_left_at_zero(loaded):
    """A materialize that silently did nothing would pass every test above."""
    model, _ = loaded
    w = model.model.layers[0].self_attn.q_proj.weight
    assert torch.isfinite(w).all()
    assert w.abs().max() > 0, "the dequantized weight is all zeros"
    assert w.count_nonzero() > w.numel() // 2, "most of the weight is zero"


def test_a_forward_pass_runs_on_the_cpu(loaded):
    model, _ = loaded
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        out = model(ids).logits
    assert out.shape == (1, 5, VOCAB), out.shape
    assert torch.isfinite(out).all(), "the logits are not finite"


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_the_loaded_weights_are_the_rust_decoders_bit_for_bit(dtype):
    """The `from_pretrained` route against `llvq-dense-digest.json`, not only the `PackedModel` one.

    `tests/test_dequant.py` holds `PackedModel.dequantize` to the Rust digests on `tiny`. The module
    path is another reader of the same bytes: its buffers come through the `transformers` loader,
    which casts a floating tensor to the dtype of the buffer it fills. A row scale narrowed to f32
    on the way, or a tail widened from f16, would change the last bits and nothing above would see
    it. At f16 the layer holds the digest's f32 values narrowed once.
    """
    import hashlib
    import json

    from transformers import AutoModelForCausalLM

    want = json.loads((MINI / "llvq-dense-digest.json").read_text())
    model = AutoModelForCausalLM.from_pretrained(MINI, dtype=getattr(torch, dtype))
    params = dict(model.named_parameters())
    checked = 0
    for name, digest in {**want["records"], **want["raw"]}.items():
        w = params[name].detach()
        assert w.dtype == getattr(torch, dtype), (name, w.dtype)
        if dtype == "float32":
            got = hashlib.sha256(w.contiguous().numpy().astype("<f4").tobytes()).hexdigest()
            assert got == digest, name
        else:
            # Narrowing is a function of the f32 value, so equal f16 bits follow from the f32 path;
            # what this checks is that the f16 load narrows the same values once, not twice.
            ref = AutoModelForCausalLM.from_pretrained(MINI, dtype=torch.float32) if not checked else ref
            assert torch.equal(w, dict(ref.named_parameters())[name].detach().to(torch.float16)), name
        checked += 1
    assert checked == len(want["records"]) + len(want["raw"]) == 8
