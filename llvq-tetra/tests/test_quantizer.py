"""Registration, and the refusals that protect a load.

The end-to-end load is gate B and needs the 4B. What is checked here is that the
registration takes, that the config refuses what it cannot read, and that a
fingerprint from another map stops the quantizer before a single index is decoded.
"""

import json
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "tiny"


@pytest.fixture(scope="module")
def qc():
    return json.loads((FIXTURE / "config.json").read_text())["quantization_config"]


def test_registration_puts_llvq_in_the_auto_mappings():
    from transformers.quantizers.auto import (
        AUTO_QUANTIZATION_CONFIG_MAPPING,
        AUTO_QUANTIZER_MAPPING,
    )

    from llvq_tetra import quantizer as q

    assert AUTO_QUANTIZER_MAPPING["llvq"] is q.LlvqQuantizer
    assert AUTO_QUANTIZATION_CONFIG_MAPPING["llvq"] is q.LlvqConfig


def test_a_config_from_the_file_carries_the_record_table(qc):
    from llvq_tetra.quantizer import LlvqConfig

    cfg = LlvqConfig(**qc)
    assert cfg.quant_method == "llvq"
    assert len(cfg.records) == len(qc["records"])
    assert cfg.tetra_fingerprint == qc["tetra_fingerprint"]


def test_another_code_order_is_refused(qc):
    from llvq_tetra.quantizer import LlvqConfig

    with pytest.raises(ValueError, match="code_order"):
        LlvqConfig(**{**qc, "code_order": "little_endian_tetra48"})


def test_a_config_with_no_records_is_refused(qc):
    from llvq_tetra.quantizer import LlvqConfig

    with pytest.raises(ValueError, match="no record table"):
        LlvqConfig(**{**qc, "records": {}})


def test_a_foreign_fingerprint_stops_the_quantizer(qc):
    from llvq_tetra.quantizer import LlvqConfig, LlvqQuantizer

    with pytest.raises(ValueError, match="plausible wrong points"):
        LlvqQuantizer(LlvqConfig(**{**qc, "tetra_fingerprint": "dead0000dead0000"}))


def test_the_quantizer_cannot_quantize_or_serialize(qc):
    from llvq_tetra.quantizer import LlvqConfig, LlvqQuantizer

    q = LlvqQuantizer(LlvqConfig(**qc))
    assert q.requires_calibration is True
    assert q.is_trainable is False
    assert q.is_serializable() is False


@pytest.mark.parametrize("field", ["tetra_fingerprint", "block_dim"])
def test_a_config_without_its_map_fields_is_refused_by_name(qc, field):
    from llvq_tetra.quantizer import LlvqConfig

    without = {k: v for k, v in qc.items() if k != field}
    with pytest.raises(ValueError, match=f"no {field}: this directory is not a Tetra model"):
        LlvqConfig(**without)


def test_another_layout_version_is_refused_by_name(qc):
    from llvq_tetra.quantizer import LlvqConfig

    assert qc["llvq_layout_version"] == 1, "the fixture is the layout hfpack writes"
    with pytest.raises(ValueError, match="llvq_layout_version is 2"):
        LlvqConfig(**{**qc, "llvq_layout_version": 2})
    without = {k: v for k, v in qc.items() if k != "llvq_layout_version"}
    assert LlvqConfig(**without).llvq_layout_version is None


def test_the_reader_refuses_another_layout_version(qc, tmp_path):
    from llvq_tetra.reader import PackedModel

    config = json.loads((FIXTURE / "config.json").read_text())
    config["quantization_config"] = {**qc, "llvq_layout_version": 2}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="llvq_layout_version is 2"):
        PackedModel(tmp_path)


@pytest.mark.parametrize("device_map", ["cuda", "cuda:0", 0, {"": 0}, {"a": "cuda", "b": "cuda:0"}])
def test_the_current_card_is_accepted(device_map, monkeypatch):
    from llvq_tetra.integration import target_device

    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert target_device(device_map) == "cuda"


@pytest.mark.parametrize("device_map", ["cuda:1", 1, "1", {"": 1}, {"a": 0, "b": 1}])
def test_another_card_is_refused_rather_than_ignored(device_map, monkeypatch):
    from llvq_tetra.integration import target_device

    monkeypatch.delenv("LOCAL_RANK", raising=False)
    with pytest.raises(ValueError, match="current CUDA device"):
        target_device(device_map)


def _on_card(monkeypatch, current):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current)


def test_device_map_cuda_after_set_device_loads_on_the_current_card(monkeypatch):
    # What `from_pretrained(..., device_map="cuda")` hands a quantizer after
    # `torch.cuda.set_device(1)`: transformers has already rewritten "cuda" as cuda:$LOCAL_RANK.
    import torch

    from llvq_tetra.integration import target_device

    _on_card(monkeypatch, 1)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert target_device({"": torch.device("cuda:0")}) == "cuda"
    monkeypatch.setenv("LOCAL_RANK", "2")
    _on_card(monkeypatch, 0)
    assert target_device({"": torch.device("cuda:2")}) == "cuda"


@pytest.mark.parametrize("named", [2, "cuda:2", "torch:2", 0])
def test_a_card_only_the_caller_can_have_named_is_refused(named, monkeypatch):
    import torch

    from llvq_tetra.integration import target_device

    _on_card(monkeypatch, 1)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    value = torch.device("cuda:2") if named == "torch:2" else named
    with pytest.raises(ValueError, match=r"current CUDA device, cuda:1.*set_device\((2|0)\)"):
        target_device({"": value})


def _embedding_desc(fixture: str) -> dict:
    config = json.loads((FIXTURE.parent / fixture / "config.json").read_text())
    return config["quantization_config"]["raw"]["model.embed_tokens.weight"]


@pytest.mark.parametrize("fixture", ["tiny", "mini"])
def test_the_metal_vocabulary_takes_what_hfpack_writes(fixture):
    from llvq_tetra.vocab import Int4Vocab

    desc = _embedding_desc(fixture)
    vocab = Int4Vocab(desc)
    assert vocab.scales.shape == (desc["dims"][0], desc["groups_per_row"])


@pytest.mark.parametrize("change", [{"group": 128}, {"group": 32}, {"bits": 8}, {"dims": [4, 132]}])
def test_the_metal_vocabulary_refuses_a_layout_its_kernels_do_not_read(change):
    from llvq_tetra.vocab import Int4Vocab

    with pytest.raises(ValueError, match="int4 in groups of 64"):
        Int4Vocab({**_embedding_desc("tiny"), **change})


def test_a_cuda_tile_the_modules_do_not_allocate_is_refused(monkeypatch):
    import torch

    from llvq_tetra import metal

    monkeypatch.setenv("LLVQ_HF_TILE", "128")
    metal.cuda_tile.cache_clear()
    with pytest.raises(ValueError, match="LLVQ_HF_TILE=128"):
        metal._extension_cuda()
    if not torch.cuda.is_available():
        # The served tile and a smaller one pass the guard and reach the device check.
        for tile in ("64", "32"):
            monkeypatch.setenv("LLVQ_HF_TILE", tile)
            metal.cuda_tile.cache_clear()
            with pytest.raises(RuntimeError, match="no device"):
                metal._extension_cuda()
    metal.cuda_tile.cache_clear()


def test_the_cuda_tile_is_the_one_the_kernel_is_compiled_with(monkeypatch):
    """`LLVQ_HF_TILE` sets `TILE_BLOCKS` in the CUDA build, and the kernel tiles by it. The shared
    bytes come from the tile the caller passes, `tile_for("cuda")`, which reads the same cached
    value as the build. Before 2026-11-21 the caller always passed 64 whatever the define."""
    from llvq_tetra import integration, metal

    for value, want in (("32", 32), ("64", 64), (None, 64)):
        if value is None:
            monkeypatch.delenv("LLVQ_HF_TILE", raising=False)
        else:
            monkeypatch.setenv("LLVQ_HF_TILE", value)
        metal.cuda_tile.cache_clear()
        assert integration.tile_for("cuda") == want
        assert integration.tile_for("mps") == integration.TILE_BLOCKS
    for bad in ("100", "48", "16", "128", "x"):
        monkeypatch.setenv("LLVQ_HF_TILE", bad)
        metal.cuda_tile.cache_clear()
        with pytest.raises(ValueError, match="LLVQ_HF_TILE"):
            metal.cuda_tile()
    metal.cuda_tile.cache_clear()
