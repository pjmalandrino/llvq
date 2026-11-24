"""The CUDA sources the wheel ships, held to the digests the tables file records.

The Metal shaders are checked by digest before every compile (`metal.shader_source`). Until
2026-11-21 the CUDA copies were only checked for existence, so a drifted copy compiled and ran.
These tests need no card: they check the copies, not the build.
"""

import shutil

import pytest

import llvq_tetra.metal as metal


def test_the_shipped_cuda_sources_match_their_digests():
    assert len(metal.require_cuda_sources()) == len(metal.CUDA_SOURCES) == 6


def test_a_drifted_cuda_copy_is_refused(tmp_path, monkeypatch):
    copy = tmp_path / "data"
    shutil.copytree(metal.DATA, copy)
    moved = tuple(copy / p.relative_to(metal.DATA) for p in metal.CUDA_SOURCES)
    target = moved[2]
    target.write_bytes(target.read_bytes() + b"\n// drift\n")
    monkeypatch.setattr(metal, "CONSTANTS", copy / "tetra-tables.json")
    monkeypatch.setattr(metal, "CUDA_SOURCES", moved)
    with pytest.raises(ValueError, match=target.name):
        metal.require_cuda_sources()
