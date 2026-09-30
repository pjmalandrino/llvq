"""The torch modules, against the reader they must agree with.

A module holds the file's tensors under the file's names and materializes a dense
weight after the load. What it must not do is disagree with `PackedModel`, which
is what gate A measured: two decode paths in one package is one too many unless
they are checked against each other.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.numpy import load_file

from llvqhf.modules import Int4Linear, RotationStore, TetraLinear
from llvqhf.reader import PackedModel
from llvqhf.tetra import TetraTables

FIXTURE = Path(__file__).parent / "fixtures" / "tiny"


@pytest.fixture(scope="module")
def bundle():
    tensors = load_file(FIXTURE / "model.safetensors")
    qc = json.loads((FIXTURE / "config.json").read_text())["quantization_config"]
    with PackedModel(FIXTURE) as m:
        yield tensors, qc, m, TetraTables.load()


def _fill(module, tensors, prefix):
    for name, buf in module.named_buffers():
        key = f"{prefix}.{name}"
        buf.copy_(torch.from_numpy(tensors[key].astype(buf.numpy().dtype)))


def test_a_tetra_module_agrees_with_the_reader(bundle):
    tensors, qc, packed, tables = bundle
    name, desc = next((n, d) for n, d in qc["records"].items() if d["kind"] == "tetra")
    module = TetraLinear(desc)
    _fill(module, tensors, desc["prefix"])
    rotations = {k: (tensors[f"llvq.rotations.{k}.signs"], tensors[f"llvq.rotations.{k}.small"])
                 for k in qc["rotations"]}
    with pytest.raises(RuntimeError, match="never materialized"):
        module(torch.zeros(1, desc["d_in"]))
    module.materialize(tables, rotations, torch.float32)
    want = packed.dequantize(name)
    assert np.array_equal(module.weight.detach().numpy(), want), "the two decode paths disagree"
    # The buffers are gone: what is compressed is the file, not the loaded model.
    assert not list(module.named_buffers())
    x = torch.zeros(2, desc["d_in"])
    assert module(x).shape == (2, desc["d_out"])


def test_an_int4_module_agrees_with_the_reader(bundle):
    tensors, qc, packed, tables = bundle
    name, desc = next((n, d) for n, d in qc["records"].items() if d["kind"] == "int4g128")
    module = Int4Linear(desc)
    _fill(module, tensors, desc["prefix"])
    module.materialize(tables, {}, torch.float32)
    assert np.array_equal(module.weight.detach().numpy(), packed.dequantize(name))


def test_a_missing_rotation_is_refused(bundle):
    tensors, qc, _, tables = bundle
    name, desc = next((n, d) for n, d in qc["records"].items()
                      if d["kind"] == "tetra" and d["rotation"])
    module = TetraLinear(desc)
    _fill(module, tensors, desc["prefix"])
    with pytest.raises(KeyError, match="is not in the file"):
        module.materialize(tables, {}, torch.float32)


def test_the_rotation_store_keys_are_the_files_keys(bundle):
    _, qc, _, _ = bundle
    store = RotationStore(qc["rotations"])
    keys = {n for n, _ in store.named_buffers()}
    for key in qc["rotations"]:
        assert f"rotations.{key}.signs" in keys
        assert f"rotations.{key}.small" in keys


def test_the_declared_dtypes_are_what_the_file_holds(bundle):
    """The loader keeps the dtype of an existing buffer, so a wrong declaration
    would silently narrow f64 scales at an f16 load."""
    tensors, qc, _, _ = bundle
    for name, desc in qc["records"].items():
        cls = TetraLinear if desc["kind"] == "tetra" else Int4Linear
        module = cls(desc)
        for buf_name, buf in module.named_buffers():
            got = tensors[f"{desc['prefix']}.{buf_name}"]
            assert buf.numpy().dtype == got.dtype, f"{name}.{buf_name}"
            assert buf.shape == got.shape, f"{name}.{buf_name}"
