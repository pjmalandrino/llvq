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
