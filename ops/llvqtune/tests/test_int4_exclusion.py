"""The int4 records `bin/export` lists are never routed.

The sealed paper-2 4B carries `o_proj` and `down_proj` of layers 12 to 23 as
int4 g128 records. A row multiplier on one of them has nothing to fold into,
and `bin/rowscale` refuses an export that names one, after the run is paid for.
"""

import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from llvqtune.adapters.torch_model import INT4_LIST, TorchModel  # noqa: E402
from llvqtune.trainables import build  # noqa: E402


def tiny_qwen3(seed=0):
    """Two blocks, nothing downloaded; `test_row_norms.py`'s toy."""
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
    return transformers.Qwen3ForCausalLM(config).eval()


INT4 = frozenset({"model.layers.0.self_attn.o_proj", "model.layers.1.mlp.down_proj"})


def test_no_list_means_none_and_an_empty_list_means_empty(tmp_path):
    assert TorchModel.int4_modules(tmp_path) is None
    (tmp_path / INT4_LIST).write_text(json.dumps({"int4": []}))
    assert TorchModel.int4_modules(tmp_path) == frozenset()


def test_the_list_is_read_as_module_names(tmp_path):
    names = [f"{n}.weight" for n in sorted(INT4)]
    (tmp_path / INT4_LIST).write_text(json.dumps({"int4": names}))
    assert TorchModel.int4_modules(tmp_path) == INT4


def test_a_listed_matrix_is_neither_shaped_nor_routed():
    model = tiny_qwen3()
    everything = TorchModel.shapes_for(model)
    shapes = TorchModel.shapes_for(model, exclude=INT4)
    assert set(everything) - set(shapes) == INT4
    mode = build("row_norms", shapes=shapes, norms=TorchModel.norm_widths_for(model))
    wrapped = TorchModel(model, mode, exclude=INT4)
    assert INT4.isdisjoint(wrapped.matrices)
    assert sorted(wrapped.matrices) == sorted(shapes)
    # The excluded linears stay plain linears: nothing reroutes them.
    for name in INT4:
        assert isinstance(model.get_submodule(name), torch.nn.Linear)
    # The norms are not matrices: every one is still routed.
    assert len(wrapped.norms) == 2 * 2 + 1


def test_the_export_names_no_int4_record_and_the_rest_train():
    model = tiny_qwen3()
    shapes = TorchModel.shapes_for(model, exclude=INT4)
    mode = build("row_norms", shapes=shapes, norms=TorchModel.norm_widths_for(model))
    TorchModel(model, mode, exclude=INT4)
    model(torch.randint(0, 97, (1, 8))).logits.sum().backward()
    assert all(p.grad is not None for p in mode.parameters()), (
        "every routed row scale and every norm multiplier must receive a gradient"
    )
    exported = mode.export()
    sigma = exported["sigma"]
    assert INT4.isdisjoint(sigma), "an int4 record leaked into the export"
    assert set(sigma) == set(shapes)
