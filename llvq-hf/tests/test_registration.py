"""`import llvqhf` must register the method, with nothing else imported.

Every other test in this suite imports `llvqhf.quantizer` directly, and so does
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
        import llvqhf
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
        for m in [k for k in sys.modules if k.startswith("llvqhf")]:
            del sys.modules[m]
        import llvqhf
        assert llvqhf.PackedModel is not None
        assert llvqhf._quantizer is None, "the guard did not catch the ImportError"
        print("reader only")
    """)
    assert r.returncode == 0, f"stdout {r.stdout!r} stderr {r.stderr[-2000:]!r}"
    assert "reader only" in r.stdout
