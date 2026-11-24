"""A fused projection stays on the device its codes were armed on.

Its resident tensors are plain attributes in the kernel's layout, which `nn.Module.to` does not
move. These tests arm the modules with CPU tensors, so they need neither a card nor MPS: a
conversion that keeps the device passes, one that changes it is refused by name, and so is an
input on another device, and a dtype change that would reach the mix of a CUDA-arm rotation.
"""

import pytest
import torch

from llvq_tetra.fused import FusedTetraLinear, Int4FusedLinear

TETRA = {"d_out": 2, "d_in": 24, "prefix": "model.layers.0.self_attn.q_proj"}
INT4 = {"d_out": 2, "d_in": 256, "groups_per_row": 2, "prefix": "model.layers.0.mlp.down_proj"}


def tetra() -> FusedTetraLinear:
    m = FusedTetraLinear(TETRA, torch.zeros(2))
    m.arm(
        {"words": torch.zeros(4, dtype=torch.int32), "gscale": torch.zeros(2),
         "rscale": torch.zeros(2), "tail": torch.zeros(1, dtype=torch.float16)},
        None,
        64,
    )
    return m


def int4() -> Int4FusedLinear:
    m = Int4FusedLinear(INT4, torch.zeros(2))
    m._wq = torch.zeros(256, dtype=torch.uint8)
    m._scales = torch.zeros(8, dtype=torch.uint8)
    m._biases = torch.zeros(8, dtype=torch.uint8)
    m._tile = 2048
    return m


@pytest.mark.parametrize("build", [tetra, int4])
def test_a_conversion_on_the_same_device_is_accepted(build):
    m = build()
    m.to("cpu")
    m.to(torch.float64)
    assert m.bias.dtype == torch.float64 and m.bias.device.type == "cpu"


@pytest.mark.parametrize("build", [tetra, int4])
def test_a_move_to_another_device_is_refused_by_name(build):
    m = build()
    with pytest.raises(RuntimeError, match="holds its codes on cpu and cannot be moved to meta"):
        m.to("meta")
    assert m.bias.device.type == "cpu", "the refusal comes before anything moves"


def rotated(n: int) -> FusedTetraLinear:
    """The CUDA arm's form: a `Rotation` module, here on the CPU, with a mix of `n / (n & -n)`."""
    import numpy as np

    from llvq_tetra.fused import Rotation

    k = n // (n & -n)
    m = tetra()
    m.rotation = Rotation(np.ones(n), np.eye(k), "cpu")
    return m


@pytest.mark.parametrize("convert", [torch.float16, torch.bfloat16, torch.float64])
def test_a_dtype_change_that_reaches_the_mix_is_refused_by_name(convert):
    m = rotated(24)
    with pytest.raises(RuntimeError, match="rotates its input with torch.float32 tables"):
        m.to(convert)
    assert m.rotation.small.dtype == torch.float32, "the refusal comes before anything converts"
    assert m.bias.dtype == torch.float32


def test_a_conversion_that_keeps_float32_reaches_a_rotated_projection():
    m = rotated(24)
    m.to("cpu")
    m.to(torch.float32)
    assert m.rotation.small.dtype == torch.float32


def test_a_rotation_without_a_mix_takes_any_dtype():
    m = rotated(16)
    m.to(torch.float16)
    assert m.bias.dtype == torch.float16


@pytest.mark.parametrize("build", [tetra, int4])
def test_an_input_on_another_device_is_refused_by_name(build):
    m = build()
    x = torch.zeros(1, m.desc["d_in"], device="meta")
    with pytest.raises(RuntimeError, match="holds its codes on cpu and was given an input on meta"):
        m(x)
