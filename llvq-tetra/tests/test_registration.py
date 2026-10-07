"""`import llvq_tetra` must register the method, with nothing else imported.

Every other test in this suite imports `llvq_tetra.quantizer` directly, and so does
every script of the package, which is exactly why the missing registration
survived four stages of measurement. So this test runs in a FRESH interpreter:
in this one, some earlier test has already imported the module and the question
cannot be asked.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

transformers = pytest.importorskip("transformers")


def run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                          capture_output=True, text=True, timeout=300)


def test_a_bare_import_registers_the_method():
    r = run("""
        import llvq_tetra
        from transformers.quantizers.auto import (
            AUTO_QUANTIZER_MAPPING, AUTO_QUANTIZATION_CONFIG_MAPPING,
        )
        assert "llvq" in AUTO_QUANTIZER_MAPPING, sorted(AUTO_QUANTIZER_MAPPING)
        assert "llvq" in AUTO_QUANTIZATION_CONFIG_MAPPING
        print("registered")
    """)
    assert r.returncode == 0, f"stdout {r.stdout!r} stderr {r.stderr[-2000:]!r}"
    assert "registered" in r.stdout


def test_the_reader_alone_needs_no_transformers():
    """`PackedModel` is numpy and safetensors, which the extras must not gate."""
    r = run("""
        import sys
        sys.modules["transformers"] = None  # an import of it now raises
        import importlib
        for m in [k for k in sys.modules if k.startswith("llvq_tetra")]:
            del sys.modules[m]
        import llvq_tetra
        assert llvq_tetra.PackedModel is not None
        assert llvq_tetra._quantizer is None, "the guard did not catch the ImportError"
        print("reader only")
    """)
    assert r.returncode == 0, f"stdout {r.stdout!r} stderr {r.stderr[-2000:]!r}"
    assert "reader only" in r.stdout


def test_an_in_tree_method_is_left_in_place():
    """The day `transformers` carries "llvq" itself, this import must not raise.

    `register_quantizer` raises on a name already held, and a version on PyPI
    cannot be changed afterwards, so the guard has to be in the first upload.
    """
    r = run("""
        from transformers.quantizers import HfQuantizer
        from transformers.quantizers.auto import (
            AUTO_QUANTIZER_MAPPING, AUTO_QUANTIZATION_CONFIG_MAPPING,
            register_quantization_config, register_quantizer,
        )
        from transformers.utils.quantization_config import QuantizationConfigMixin

        @register_quantization_config("llvq")
        class InTreeConfig(QuantizationConfigMixin):
            pass

        @register_quantizer("llvq")
        class InTreeQuantizer(HfQuantizer):
            pass

        import llvq_tetra
        assert AUTO_QUANTIZATION_CONFIG_MAPPING["llvq"] is InTreeConfig
        assert AUTO_QUANTIZER_MAPPING["llvq"] is InTreeQuantizer
        print("left in place")
    """)
    assert r.returncode == 0, f"stdout {r.stdout!r} stderr {r.stderr[-2000:]!r}"
    assert "left in place" in r.stdout


def test_a_transformers_too_old_raises_instead_of_registering_nothing():
    """A missing submodule is an incompatible `transformers`, not an absent one.

    Swallowed, it would leave "llvq" unregistered and `from_pretrained` would
    load a random model without raising. `transformers.core_model_loading`
    does not exist before 5.x, which is the realistic way to get here.
    """
    r = run("""
        import sys
        sys.modules["transformers.core_model_loading"] = None  # an import of it now raises
        import llvq_tetra
        print("imported")
    """)
    assert r.returncode != 0, f"the import succeeded: stdout {r.stdout!r}"
    assert "imported" not in r.stdout
    assert "core_model_loading" in r.stderr, r.stderr[-2000:]
