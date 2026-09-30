"""The Metal op, against the numpy decode and against its own refusals.

The gate over the served 4B is `python -m llvqhf.checkdecode` and it needs the
1.4 GB object. These run on the fixture and on drawn labels, and they skip nothing:
a machine with no MPS device fails here by name rather than passing quietly.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.numpy import load_file

from llvqhf.metal import shader_source, stride_u32, tetra_decode, to_tetra48
from llvqhf.tetra import TetraTables, split_stream

FIXTURE = Path(__file__).parent / "fixtures" / "tiny"

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="the op runs on Metal, and this host has no MPS device",
)


@pytest.fixture(scope="module")
def record():
    qc = json.loads((FIXTURE / "config.json").read_text())["quantization_config"]
    name, d = next((n, d) for n, d in qc["records"].items() if d["kind"] == "tetra")
    codes = load_file(FIXTURE / "model.safetensors")[f"{d['prefix']}.codes"]
    return name, d, codes


def test_the_op_agrees_with_numpy_on_the_fixture(record):
    _, d, codes = record
    tables = TetraTables.load()
    labels, _ = split_stream(codes, d["d_out"] * d["nblocks"], d["index_bits"], d["gain_bits"])
    want = tables.decode(labels).astype(np.int8)
    got = tetra_decode(codes, d["d_out"], d["nblocks"]).cpu().numpy()
    assert np.array_equal(got, want), "the shader and numpy disagree"


def test_the_op_agrees_on_drawn_labels():
    """A whole row of labels drawn at random, which the fixture's 12 blocks do not cover."""
    rng = np.random.default_rng(0x2E)
    n = 4096
    tables = TetraTables.load()
    labels = rng.integers(0, 1 << 47, size=n, dtype=np.uint64)
    gains = rng.integers(0, 2, size=n, dtype=np.uint32)
    # Back to a disk stream: 47 index bits then one gain bit, MSB first.
    value = (labels << np.uint64(1)) | gains.astype(np.uint64)
    codes = np.empty((n, 6), dtype=np.uint8)
    for i in range(6):
        codes[:, i] = ((value >> np.uint64(8 * (5 - i))) & np.uint64(0xFF)).astype(np.uint8)
    got = tetra_decode(codes.reshape(-1), 1, n).cpu().numpy()
    assert np.array_equal(got, tables.decode(labels).astype(np.int8))


def test_the_shipped_shader_is_the_one_the_tables_record():
    src = shader_source()
    assert "kernel void tetra48_probe" in src
    assert "#pragma clang fp contract(off)" in src


def test_a_drifted_shader_copy_is_refused(tmp_path, monkeypatch):
    """Each copy is checked before every compile, not trusted because it shipped."""
    import llvqhf.metal as metal

    for name in ("llvq_tetra48.metal", "tv_q4_h.metal"):
        drifted = tmp_path / name
        drifted.write_bytes((metal.DATA / name).read_bytes() + b"\n// drift\n")
        monkeypatch.setattr(metal, "DATA", tmp_path)
        with pytest.raises(ValueError, match="not the shader the tables were dumped with"):
            metal.shader_source(name)
        monkeypatch.undo()


def test_an_unknown_shader_is_refused():
    from llvqhf import metal

    with pytest.raises(KeyError, match="not one of the shipped shaders"):
        metal.shader_source("tv_planes_h.metal")


def test_the_stride_covers_the_last_blocks_read_window():
    """`round_up(6·nblocks, 8) / 4`: eight and not four, for the two-word window.

    The invariant is the shader's own read, not a margin: `f1r_load` reads
    `row[w]` and `row[w + 1]` with `w = (3j) >> 1`, so for the last block both
    words must sit inside the row. At `nblocks = 106` that is what the pad buys,
    636 bytes of words against a stride of 640; at 512 the words already end on an
    eight-byte boundary and the pad is zero.
    """
    assert stride_u32(106) == 160
    for nblocks in (1, 2, 3, 105, 106, 512):
        w = (3 * (nblocks - 1)) >> 1
        assert 4 * (w + 2) <= stride_u32(nblocks) * 4, nblocks


def test_a_label_over_47_bits_is_refused():
    with pytest.raises(ValueError, match="collide with the gain bit"):
        to_tetra48(np.array([1 << 47], dtype=np.uint64), np.zeros(1, np.uint32), 1, 1)


def test_a_stream_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="code bytes for"):
        tetra_decode(np.zeros(11, dtype=np.uint8), 1, 2)
