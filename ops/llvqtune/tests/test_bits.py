"""Hard rule 6: every memory figure is b/param over the whole model."""

import pytest

from llvqtune.domain.bits import BitCost


def test_a_free_set_reports_zero():
    cost = BitCost(added_params=0, added_bits=0, model_params=4_022_458_880)
    assert cost.is_free
    assert cost.b_per_param_whole_model == 0.0
    assert "free" in str(cost)


def test_rate_divides_by_the_whole_model():
    cost = BitCost(added_params=100, added_bits=1600, model_params=1600)
    assert cost.b_per_param_whole_model == 1.0
    assert not cost.is_free


def test_negative_cost_is_refused():
    with pytest.raises(ValueError):
        BitCost(added_params=-1, added_bits=0, model_params=10)
    with pytest.raises(ValueError):
        BitCost(added_params=0, added_bits=-1, model_params=10)


def test_empty_model_is_refused():
    with pytest.raises(ValueError):
        BitCost(added_params=0, added_bits=0, model_params=0)
