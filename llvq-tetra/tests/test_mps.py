"""The compressed path on Metal, on the 148 KB fixture: every layer and the vocabulary stay
compressed, and the logits match the dense path on the CPU."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal only")

MINI = Path(__file__).parent / "fixtures" / "mini"


def test_the_fixture_stays_compressed_on_metal():
    import llvq_tetra  # noqa: F401
    from transformers import AutoModelForCausalLM

    from llvq_tetra.fused import FusedTetraLinear, Int4FusedLinear
    from llvq_tetra.vocab import Int4TiedHead, Int4Vocab

    model, info = AutoModelForCausalLM.from_pretrained(
        MINI, dtype=torch.float32, device_map="mps", output_loading_info=True
    )
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        assert not info[key], f"{key}: {info[key]}"
    kinds = [type(m) for m in model.modules()]
    assert kinds.count(FusedTetraLinear) == 6
    assert kinds.count(Int4FusedLinear) == 1
    assert kinds.count(Int4Vocab) == 1 and kinds.count(Int4TiedHead) == 1
    assert not any(isinstance(m, torch.nn.Embedding) for m in model.modules()), "the vocabulary went dense"

    dense = AutoModelForCausalLM.from_pretrained(MINI, dtype=torch.float32)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    with torch.no_grad():
        got = model(ids.to("mps")).logits.cpu().double()
        ref = dense(ids).logits.double()
    err = ((got - ref).abs().amax(-1) / ref.abs().amax(-1)).max().item()
    # The kernel reads the tail in half and sums in its own order; the dense path is f64 then f32.
    assert err < 1e-2, f"compressed logits off the dense ones by {err:.2e}"
