"""The CUDA binding, checked on a machine with no card.

`csrc/tetra_cuda.cu` is compiled by nvcc in a job and never here. What a Mac can
hold is everything around the compile: that the sources it includes are the ones
the tables file attests, byte for byte, and that each op it defines has an
implementation and the argument list its Python callers pass. A schema that
drifted from its caller fails at the first call on a rented card otherwise.
"""

from __future__ import annotations

import json
import re

import pytest

from llvq_tetra import metal

BINDING = metal.CSRC_CUDA.read_text()


def schemas() -> dict[str, list[str]]:
    """`name -> [argument names]` from the `m.def` strings of the binding."""
    # The schema strings are split across C++ string literals; join them first.
    joined = re.sub(r'"\s*\n\s*"', "", BINDING)
    out = {}
    for name, args in re.findall(r'm\.def\("(\w+)\(([^)]*)\)\s*->\s*Tensor"\)', joined):
        out[name] = [a.strip().split()[-1] for a in args.split(",")]
    return out


def test_every_shipped_source_is_attested_and_unchanged():
    metal.verify_cuda_sources()
    shaders = json.loads(metal.CONSTANTS.read_text())["shaders"]
    attested = {f for f, e in shaders.items() if e["backend"] == "cuda"}
    assert attested == {p.name for p in metal.CUDA_SOURCES}


def test_a_drifted_copy_is_refused(tmp_path, monkeypatch):
    """The check is not decorative: a moved byte in a shipped source fails by name."""
    src = metal.CUDA_SOURCES[6]
    assert src.name == "tv_q4_h.cu"
    copy = tmp_path / src.name
    copy.write_bytes(src.read_bytes() + b"\n")
    sources = list(metal.CUDA_SOURCES)
    sources[6] = copy
    monkeypatch.setattr(metal, "CUDA_SOURCES", tuple(sources))
    with pytest.raises(ValueError, match="tv_q4_h.cu is not the source"):
        metal.verify_cuda_sources()


def test_every_defined_op_has_a_cuda_implementation():
    defined = set(schemas())
    implemented = set(re.findall(r'm\.impl\("(\w+)"', BINDING))
    assert defined == {"tv_tetra48", "tv_q4", "tv_emb_q4", "rot_apply_rows"}
    assert implemented == defined


@pytest.mark.parametrize(
    "op, args",
    [
        # The callers' argument lists, in the order they pass them: `fused.py`
        # for the Tetra matvec, `llvq_vllm/ops.py` for the four of them.
        ("tv_tetra48", ["words", "rows", "prefixes", "branches", "suffixes", "gscale",
                        "invnorm", "rscale", "tail", "x", "d_out", "nblocks", "tail_w",
                        "row_stride_u32", "tile", "threads"]),
        ("tv_q4", ["wq", "scales", "biases", "x", "d_out", "d_in", "gpr", "threads"]),
        ("tv_emb_q4", ["wq", "scales", "biases", "x", "vocab", "d", "gpr", "threads"]),
        ("rot_apply_rows", ["x", "signbits", "small", "n", "m", "k", "inv", "threads"]),
    ],
)
def test_the_schema_is_the_argument_list_the_callers_pass(op, args):
    assert schemas()[op] == args


def test_each_launch_names_the_served_kernel():
    """One wrapper, one served entry point, and no kernel written here."""
    for kernel in ("tv_tetra48_h", "tv_q4_h", "tv_emb_q4_h", "rot_apply_rows"):
        assert re.search(rf"\b{kernel}<<<", BINDING), f"no launch of {kernel}"
    assert "__global__" not in BINDING, "the binding defines a kernel of its own"
