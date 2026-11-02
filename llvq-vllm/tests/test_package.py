"""How vLLM finds the method, and what happens where vLLM is absent."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
HAS_VLLM = importlib.util.find_spec("vllm") is not None


def test_the_entry_point_is_the_one_vllm_loads_in_every_process():
    """`vllm.general_plugins` is the group `vllm/plugins/__init__.py` loads (vLLM 0.26.0)."""
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert meta["project"]["entry-points"]["vllm.general_plugins"] == {
        "llvq": "llvq_vllm:register"}


def test_no_dependency_can_move_the_images_torch():
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert meta["project"]["dependencies"] == []


def test_register_names_vllm_when_it_is_missing_and_registers_when_it_is_not():
    import llvq_vllm

    if not HAS_VLLM:
        with pytest.raises(ImportError, match="needs vLLM"):
            llvq_vllm.register()
        return
    from vllm.model_executor.layers.quantization import (
        QUANTIZATION_METHODS,
        get_quantization_config,
    )

    llvq_vllm.register()
    llvq_vllm.register()  # vLLM may load a plugin more than once in a process
    assert "llvq" in QUANTIZATION_METHODS
    assert get_quantization_config("llvq").__name__ == "LlvqVllmConfig"


def test_the_package_imports_without_vllm_and_without_registering():
    """A fresh interpreter: importing the package touches neither vLLM nor a card."""
    code = ("import sys, llvq_vllm, llvq_vllm.layer, llvq_vllm.ops; "
            "assert 'vllm' not in sys.modules; print('ok')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True)
    assert out.stdout.strip() == "ok"
