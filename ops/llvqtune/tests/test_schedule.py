from llvqtune.domain.schedule import Constant, WarmupCosine

import pytest


def test_warmup_climbs_to_the_peak():
    s = WarmupCosine(peak=1.0, total_steps=100, warmup_steps=10)
    assert s.at(0) == pytest.approx(0.1)
    assert s.at(9) == pytest.approx(1.0)


def test_cosine_decays_to_the_floor():
    s = WarmupCosine(peak=1.0, total_steps=100, warmup_steps=0, final_ratio=0.1)
    assert s.at(0) == pytest.approx(1.0)
    assert s.at(99) == pytest.approx(0.1, abs=1e-3)


def test_rate_never_leaves_its_band():
    s = WarmupCosine(peak=2.0, total_steps=50, warmup_steps=5, final_ratio=0.25)
    for step in range(50):
        assert 0 < s.at(step) <= 2.0 + 1e-12


def test_constant_is_constant():
    s = Constant(peak=3e-4)
    assert s.at(0) == s.at(999) == 3e-4


def test_impossible_schedules_are_refused():
    with pytest.raises(ValueError):
        WarmupCosine(peak=1.0, total_steps=0)
    with pytest.raises(ValueError):
        WarmupCosine(peak=1.0, total_steps=10, warmup_steps=11)
    with pytest.raises(ValueError):
        WarmupCosine(peak=0.0, total_steps=10)


def test_negative_step_is_refused():
    with pytest.raises(ValueError):
        WarmupCosine(peak=1.0, total_steps=10).at(-1)
