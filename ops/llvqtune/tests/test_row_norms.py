"""The input axis: RMSNorm multipliers beside the row scales."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from llvqtune.adapters.torch_model import TorchModel  # noqa: E402
from llvqtune.trainables import build  # noqa: E402
from llvqtune.trainables.row_norms import RowNorms  # noqa: E402

QWEN3_4B_PARAMS = 4_022_458_880


def tiny_qwen3(seed=0):
    """Two blocks, every width a multiple of 24 plus a tail, nothing downloaded."""
    torch.manual_seed(seed)
    config = transformers.Qwen3Config(
        vocab_size=97,
        hidden_size=56,
        intermediate_size=104,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    model = transformers.Qwen3ForCausalLM(config).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
    return model


def routed(model):
    shapes = TorchModel.shapes_for(model)
    norms = TorchModel.norm_widths_for(model)
    mode = build("row_norms", shapes=shapes, norms=norms)
    return TorchModel(model, mode), mode


def test_the_mode_is_registered_and_free():
    mode = build("row_norms", shapes={"m": (8, 48)}, norms={"n": 48})
    assert isinstance(mode, RowNorms)
    assert mode.cost(QWEN3_4B_PARAMS).is_free


def test_an_empty_norm_set_is_refused():
    with pytest.raises(ValueError, match="use row_scales"):
        RowNorms({"m": (8, 48)}, norms={})


def test_every_norm_of_the_model_is_routed():
    model = tiny_qwen3()
    wrapped, mode = routed(model)
    assert sorted(wrapped.norms) == sorted(mode.norms)
    assert len(wrapped.norms) == 2 * 2 + 1


def test_an_untrained_run_is_the_artifact():
    """tau and sigma are 1 at step zero, so the logits are the input model's."""
    ids = torch.randint(0, 97, (2, 11))
    reference = tiny_qwen3()(input_ids=ids).logits
    wrapped, _ = routed(tiny_qwen3())
    assert torch.equal(wrapped.forward(ids), reference)


def test_the_gradient_reaches_tau_and_sigma_and_nothing_frozen():
    model = tiny_qwen3()
    matrices = len(TorchModel.shapes_for(model))
    wrapped, mode = routed(model)
    wrapped.forward(torch.randint(0, 97, (1, 9))).pow(2).mean().backward()
    assert len(mode.parameters()) == matrices + 5
    assert all(p.grad is not None for p in mode.parameters())
    # Routing renames a norm's weight to `<norm>.source.weight`; match both.
    checked = 0
    for name, p in model.named_parameters():
        if "layernorm" in name or name.startswith("model.norm."):
            assert p.grad is None, name
            checked += 1
    assert checked == 5


def test_the_fold_is_the_trained_function():
    """Writing w * tau into the norm weight gives the logits the run trained.

    This is the write-back's contract: a trained file is an input model whose
    norms were multiplied, and nothing else.
    """
    ids = torch.randint(0, 97, (2, 13))
    wrapped, mode = routed(tiny_qwen3())
    torch.manual_seed(1)
    with torch.no_grad():
        for p in mode.parameters():
            p.copy_(1.0 + 0.05 * torch.randn_like(p))
    trained = wrapped.forward(ids)

    folded = tiny_qwen3()
    tau = mode.export()["tau"]
    lattice = TorchModel.shapes_for(folded)
    sigma = mode.export()["sigma"]
    with torch.no_grad():
        for norm, values in tau.items():
            folded.get_submodule(norm).weight.mul_(torch.tensor(values))
        for matrix, values in sigma.items():
            _, cols = lattice[matrix]
            folded.get_submodule(matrix).weight[:, :cols].mul_(
                torch.tensor(values).unsqueeze(1)
            )
    assert torch.allclose(folded(input_ids=ids).logits, trained, atol=1e-5)


def test_a_norm_width_mismatch_is_refused():
    mode = RowNorms({"m": (8, 48)}, norms={"n": 48})
    with pytest.raises(ValueError, match="multipliers"):
        mode.scale_norm("n", torch.randn(3, 40))


def test_the_export_is_refused_by_rowscale():
    """`bin/rowscale` folds kind row_scales only; this kind must not match it."""
    payload = RowNorms({"m": (2, 24)}, norms={"n": 3}).export()
    assert payload["kind"] == "row_norms"
    assert payload["tau"]["n"] == [1.0, 1.0, 1.0]
    assert payload["sigma"]["m"] == [1.0, 1.0]
