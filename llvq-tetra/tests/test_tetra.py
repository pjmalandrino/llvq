"""The decode, against invariants of the lattice rather than against itself.

Gate A on the served 4B is the strong check and it needs a 1.4 GB object. These
run on the committed fixture and on the tables alone, in the fast loop.
"""

import numpy as np
import pytest

from llvq_tetra.tetra import TetraTables, split_stream

RNG = np.random.default_rng(0xF1)


@pytest.fixture(scope="module")
def tables():
    return TetraTables.load()


def test_the_origin_word_is_the_origin(tables):
    """Word 0 is p = 0, state 0, the all-zero rank vector: the origin block."""
    assert np.all(tables.decode(np.zeros(1, dtype=np.uint64)) == 0)


def test_every_decoded_point_is_in_the_lattice(tables):
    """Two invariants of Λ₂₄ that cost nothing and catch a table mix-up.

    A point of the integer embedding has all coordinates congruent mod 2, and a
    squared norm that is a multiple of 16. A decode that read the wrong row table,
    or mixed two sections, breaks one or the other almost surely.
    """
    labels = RNG.integers(0, 1 << tables.label_bits, size=20_000, dtype=np.uint64)
    y = tables.decode(labels).astype(np.int64)
    parity = y & 1
    assert np.all(parity.min(axis=1) == parity.max(axis=1)), "coordinates of mixed parity"
    n2 = (y**2).sum(axis=1)
    assert np.all(n2 % 16 == 0), "a squared norm that is not a multiple of 16"
    assert np.abs(y).max() <= 10, "a coordinate past the codebook's maximum of 10"
    assert n2.min() > 0, "a random label should not be the origin"


def test_the_gain_bit_is_not_part_of_the_map(tables):
    """A label is 47 bits. Bit 47 is the gain and belongs to the caller."""
    with pytest.raises(ValueError, match="above 47 bits"):
        tables.decode(np.array([1 << 47], dtype=np.uint64))


def test_a_fingerprint_from_another_map_is_refused(tables):
    with pytest.raises(ValueError, match="plausible wrong points"):
        tables.require_fingerprint("0000000000000000")
    tables.require_fingerprint(tables.fingerprint.upper())  # case is not the check


def test_the_stream_is_read_most_significant_bit_first():
    """48 bits a block, `label << 1 | gain`, big-endian over six bytes.

    The little-endian reading is the served `tetra48` layout, a different
    convention. Getting it backwards returns plausible wrong points, so the order
    is pinned on a word whose two readings differ.
    """
    codes = np.array([0x01, 0x00, 0x00, 0x00, 0x00, 0x03], dtype=np.uint8)
    labels, gains = split_stream(codes, 1, 47, 1)
    assert int(labels[0]) == (0x010000000003 >> 1)
    assert int(gains[0]) == 1


def test_an_unaligned_width_is_refused():
    with pytest.raises(ValueError, match="not byte aligned"):
        split_stream(np.zeros(6, dtype=np.uint8), 1, 47, 2)
