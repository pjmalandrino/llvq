"""The three training modes, and the invariant they all share."""

import pytest

torch = pytest.importorskip("torch")

from llvqtune.trainables import build  # noqa: E402
from llvqtune.trainables.low_rank import LowRank  # noqa: E402
from llvqtune.trainables.row_scales import RowScales  # noqa: E402

QWEN3_4B_PARAMS = 4_022_458_880


def frozen(rows=8, cols=53):
    t = torch.randn(rows, cols)
    t.requires_grad_(False)
    return t


def test_row_scales_start_at_the_artifact():
    """Sigma is 1 at step zero, so an untrained run writes the same file."""
    t = frozen()
    mode = RowScales({"m": (8, 48)})
    assert torch.equal(mode.weight("m", t), t)


def test_the_tail_is_never_scaled():
    t = frozen(rows=8, cols=53)
    mode = RowScales({"m": (8, 48)})
    with torch.no_grad():
        mode.parameters()[0].fill_(2.0)
    out = mode.weight("m", t)
    assert torch.equal(out[:, 48:], t[:, 48:])
    assert torch.allclose(out[:, :48], t[:, :48] * 2.0)


def test_the_gradient_never_reaches_the_decoder():
    """The Leech decoder is a table lookup. Nothing differentiates through it."""
    t = frozen()
    mode = RowScales({"m": (8, 48)})
    mode.weight("m", t).sum().backward()
    assert t.grad is None
    assert mode.parameters()[0].grad is not None


def test_directions_that_carry_a_gradient_are_refused():
    t = torch.randn(8, 48, requires_grad=True)
    mode = RowScales({"m": (8, 48)})
    with pytest.raises(ValueError, match="never differentiated"):
        mode.weight("m", t)


def test_row_scales_cost_nothing():
    mode = RowScales({"m": (8, 48)})
    assert mode.cost(QWEN3_4B_PARAMS).is_free


def test_a_lattice_width_off_the_block_is_refused():
    with pytest.raises(ValueError, match="multiple of 24"):
        RowScales({"m": (8, 50)})


def test_export_names_how_it_is_folded_back():
    mode = RowScales({"m": (2, 24)})
    payload = mode.export()
    assert payload["kind"] == "row_scales"
    assert payload["apply"] == "row_scales[i] *= sigma[i]"
    assert payload["sigma"]["m"] == [1.0, 1.0]


def test_low_rank_starts_at_the_artifact():
    """B is zero at step zero, so the correction begins as a no-op."""
    t = frozen(rows=8, cols=48)
    mode = LowRank({"m": (8, 48)}, rank=4)
    assert torch.equal(mode.weight("m", t), t)


def test_low_rank_reproduces_the_roadmap_rate():
    """Row 20 prices r = 32 at +0.263 b/param and r = 16 at +0.131."""
    shapes = {}
    for layer in range(36):
        shapes[f"l{layer}.q"] = (4096, 2560)
        shapes[f"l{layer}.k"] = (1024, 2560)
        shapes[f"l{layer}.v"] = (1024, 2560)
        shapes[f"l{layer}.o"] = (2560, 4096)
        shapes[f"l{layer}.gate"] = (9728, 2560)
        shapes[f"l{layer}.up"] = (9728, 2560)
        shapes[f"l{layer}.down"] = (2560, 9728)
    r32 = LowRank(shapes, rank=32).cost(QWEN3_4B_PARAMS)
    r16 = LowRank(shapes, rank=16).cost(QWEN3_4B_PARAMS)
    assert r32.b_per_param_whole_model == pytest.approx(0.263, abs=5e-4)
    assert r16.b_per_param_whole_model == pytest.approx(0.131, abs=5e-4)
    assert not r32.is_free


def test_free_params_trains_the_tail_too():
    tails = {"m": torch.randn(8, 5)}
    mode = build("free_params", shapes={"m": (8, 48)}, tails=tails)
    assert len(mode.parameters()) == 2
    assert mode.cost(QWEN3_4B_PARAMS).is_free
    t = frozen(rows=8, cols=53)
    mode.weight("m", t).sum().backward()
    assert t.grad is None
    assert all(p.grad is not None for p in mode.parameters())


def test_an_unknown_mode_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown mode"):
        build("qlora", shapes={"m": (8, 48)})


def test_linear_agrees_with_building_the_weight():
    """The commuting form must be the same function, not merely a faster one."""
    torch.manual_seed(0)
    t = frozen(rows=8, cols=53)
    mode = RowScales({"m": (8, 48)}, dtype=torch.float64)
    with torch.no_grad():
        mode.parameters()[0].copy_(torch.linspace(0.8, 1.2, 8).double())
    x = torch.randn(3, 53, dtype=torch.float64)
    t64 = t.double()
    built = torch.nn.functional.linear(x, mode.weight("m", t64))
    fast = mode.linear("m", x, t64)
    assert torch.allclose(built, fast, atol=1e-12), (built - fast).abs().max()


def test_linear_carries_the_bias_and_the_gradient():
    t = frozen(rows=4, cols=29)
    mode = RowScales({"m": (4, 24)})
    bias = torch.zeros(4)
    mode.linear("m", torch.randn(2, 29), t, bias).sum().backward()
    assert t.grad is None
    assert mode.parameters()[0].grad is not None


def test_linear_refuses_directions_that_carry_a_gradient():
    t = torch.randn(4, 24, requires_grad=True)
    mode = RowScales({"m": (4, 24)})
    with pytest.raises(ValueError, match="never differentiated"):
        mode.linear("m", torch.randn(2, 24), t)
