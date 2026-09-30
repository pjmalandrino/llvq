"""The whole chain on the committed fixture, against the Rust decoder.

The fixture is written by the Rust test that asserts on the same object:

    LLVQ_HF_FIXTURE=../llvq-hf/tests/fixtures/tiny \
        cargo test -p llvq-llm --test hfpack

It holds one Tetra record with a tail and a rotation, one Int4G128 record, a
quantized embedding and an f16 norm, which is every arm of the format in 24 KB.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from llvqhf.dequant import unrotate_rows, wht_rows
from llvqhf.reader import PackedModel

FIXTURE = Path(__file__).parent / "fixtures" / "tiny"


@pytest.fixture(scope="module")
def packed():
    if not FIXTURE.exists():
        raise AssertionError(
            f"the fixture {FIXTURE} is missing. Write it with "
            "`LLVQ_HF_FIXTURE=../llvq-hf/tests/fixtures/tiny "
            "cargo test -p llvq-llm --test hfpack`"
        )
    with PackedModel(FIXTURE) as m:
        yield m


def sha_f32(a: np.ndarray) -> str:
    assert a.dtype == np.dtype("<f4"), a.dtype
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


@pytest.fixture(scope="module")
def dense():
    return json.loads((FIXTURE / "llvq-dense-digest.json").read_text())


def test_every_record_matches_the_rust_decoder(packed, dense):
    """Gate A, in the fast loop: bit for bit, not close."""
    assert set(packed.records) == set(dense["records"])
    for name in packed.records:
        assert sha_f32(packed.dequantize(name)) == dense["records"][name], name


def test_the_quantized_embedding_matches(packed, dense):
    assert list(dense["raw"]) == packed.quantized_raw()
    for name in dense["raw"]:
        assert sha_f32(packed.dequantize_raw(name)) == dense["raw"][name], name


def test_the_tetra_record_carries_a_tail_and_a_rotation(packed):
    """The fixture is only a fixture if it exercises the optional parts."""
    tetra = [n for n, d in packed.records.items() if d["kind"] == "tetra"]
    assert tetra, "no Tetra record in the fixture"
    d = packed.records[tetra[0]]
    assert d["tail_cols"] > 0, "a fixture with no tail tests no tail"
    assert d["rotation"], "a fixture with no rotation tests no rotation"
    assert packed.rotation(d["rotation"]) is not None


def test_a_dangling_rotation_is_refused(packed):
    with pytest.raises(KeyError, match="does not carry"):
        packed.rotation("2560_deadbeefdeadbeef")


def test_the_transform_agrees_with_a_dense_hadamard():
    """An independent yardstick for the butterfly: the matrix it stands for.

    `H_1 = [[1]]`, `H_2m = [[H, H], [H, −H]]`, scaled by `1/√m`. The butterfly is
    bit-exact against Rust; this says it is also the transform it claims to be.
    """
    for m in (1, 2, 4, 8, 64):
        h = np.array([[1.0]])
        while h.shape[0] < m:
            h = np.block([[h, h], [h, -h]])
        h = h / np.sqrt(float(m))
        v = np.random.default_rng(m).standard_normal((3, m))
        want = v @ h.T
        got = v.copy()
        wht_rows(got)
        assert np.allclose(got, want, atol=1e-13), m


def test_the_rotation_is_orthogonal_and_its_inverse_is_the_transpose():
    """`unrotate_rows` is `W ← W Q` for an orthogonal `Q`, so it preserves norms."""
    n, seed = 40, 7  # n = 8 · 5, so the odd factor is exercised
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=n)
    k = n // (n & -n)
    q, _ = np.linalg.qr(rng.standard_normal((k, k)))
    w = rng.standard_normal((6, n))
    out = unrotate_rows(w.copy(), signs, np.ascontiguousarray(q))
    assert np.allclose(np.linalg.norm(out, axis=1), np.linalg.norm(w, axis=1), atol=1e-12)


def test_a_wrong_sign_count_is_refused():
    with pytest.raises(ValueError, match="signs for"):
        unrotate_rows(np.zeros((2, 8)), np.ones(4), np.ones((1, 1)))
