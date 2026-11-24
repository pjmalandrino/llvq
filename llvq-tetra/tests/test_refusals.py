"""A packed directory with a tensor missing or cut short is refused, not loaded.

`transformers` fills a missing key with `torch.empty_like`, uninitialised memory,
and lists it in `missing_keys` with a warning. With a quantizer present it also
skips its shape check (`core_model_loading.py`, `hf_quantizer is None`), so a
tensor of the wrong length is assigned as it comes. Before 2026-11-21 both loaded:
on Metal a short `row_scales` or `scales` was read past its end by the kernel,
with `missing_keys` and `mismatched_keys` both empty, and the logits finite.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import llvq_tetra  # noqa: E402, F401
from safetensors.torch import load_file, save_file  # noqa: E402

MINI = Path(__file__).parent / "fixtures" / "mini"

CASES = [
    ("model.layers.0.self_attn.q_proj.row_scales", "drop"),
    ("model.layers.0.self_attn.q_proj.codes", "drop"),
    ("model.layers.0.self_attn.q_proj.tail", "drop"),
    ("model.layers.0.mlp.down_proj.scales", "drop"),
    ("llvq.rotations.136_0000000000005eed.signs", "drop"),
    ("model.layers.0.self_attn.q_proj.row_scales", "short"),
    ("model.layers.0.self_attn.q_proj.tail", "short"),
    ("model.layers.0.mlp.down_proj.scales", "short"),
]


def broken(tmp_path: Path, key: str, mode: str) -> Path:
    if not MINI.exists():
        pytest.fail(f"{MINI} is missing; write it with `cargo test -p llvq-llm --test hfpack`")
    out = tmp_path / "broken"
    shutil.copytree(MINI, out)
    tensors = load_file(out / "model.safetensors")
    assert key in tensors, f"{key} is not in the fixture: the case tests nothing"
    if mode == "drop":
        del tensors[key]
    else:
        tensors[key] = tensors[key][:-8].clone()
    save_file(tensors, out / "model.safetensors", metadata={"format": "pt"})
    return out


def load(path: Path, device_map):
    from transformers import AutoModelForCausalLM

    kw = {"dtype": torch.float32}
    if device_map is not None:
        kw["device_map"] = device_map
    return AutoModelForCausalLM.from_pretrained(path, **kw)


@pytest.mark.parametrize("key,mode", CASES)
def test_a_broken_checkpoint_is_refused_on_the_cpu(tmp_path, key, mode):
    with pytest.raises(ValueError, match=key.rpartition(".")[0].replace(".", r"\.")):
        load(broken(tmp_path, key, mode), None)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal only")
@pytest.mark.parametrize("key,mode", CASES)
def test_a_broken_checkpoint_is_refused_on_metal(tmp_path, key, mode):
    with pytest.raises(ValueError, match=key.rpartition(".")[0].replace(".", r"\.")):
        load(broken(tmp_path, key, mode), "mps")


def test_the_intact_fixture_still_loads(tmp_path):
    """The refusal must not fire on a whole file, or every case above passes for nothing."""
    out = tmp_path / "whole"
    shutil.copytree(MINI, out)
    load(out, None)
