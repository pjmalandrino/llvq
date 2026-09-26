"""The gauge names what it reads, and reads the three things it names.

No card is needed: the allocator's three functions are replaced by distinct
constants, so a swapped key or a dropped field fails here, on a Mac.
"""

import pytest

torch = pytest.importorskip("torch")

from llvqtune.adapters.cuda_gauge import CudaGauge  # noqa: E402


class _Props:
    total_memory = 150_000_000_000


def test_it_reads_allocated_reserved_and_total(monkeypatch):
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda d=None: 74_000_000_000)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda d=None: 78_000_000_000)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda d=None: _Props())
    assert CudaGauge("cuda")() == {
        "max_memory_allocated": 74_000_000_000,
        "max_memory_reserved": 78_000_000_000,
        "total_memory": 150_000_000_000,
    }


def test_it_refuses_a_device_that_is_not_cuda():
    with pytest.raises(ValueError, match="CUDA"):
        CudaGauge("cpu")
