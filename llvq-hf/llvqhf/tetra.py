"""The Tetra map: a 47-bit label to a point of Λ₂₄, vectorized.

Ported from `llvq-search/src/tetra/mod.rs`, whose module header is the
specification. The decode is four table lookups and a nibble read per section:

    c1 = prefixes[s8][b1]   (c2, s16) = branches[s8][b2]   c3 = suffixes[s16][b3]
    row1 = rows[2048·r + i1]
    row2 = i2 < N0 ? rows[i2] : rows[2048 + i2 − N0]        δ = [i2 ≥ N0]
    row3 = rows[2048·(p ⊕ r ⊕ δ) + i3]
    y[8k + j] = val(p + 2·bit_j(c_k), nibble_j(row_k))      then y → natural order

## Why the tables are shipped and not derived

The map belongs to the codebook, not to a model, which is why a `.llvq` header
carries a fingerprint and no table. Re-deriving Golay, the trio and the trellis
here would be a second implementation of the same object, to be held bit-exact
forever. So `bin/tetratables` dumps them once from the Rust accessors, and
[`TetraTables.require_fingerprint`] refuses a model whose header names another
map.

## The word on disk is not the word in the kernel

`llvq-artifact` packs a block MSB-first: 47 index bits then one gain bit. So the
six bytes of a block read big-endian give `label << 1 | gain`, and the Tetra word
is `label | gain << 47`. The little-endian reading is the served `tetra48` layout
and a different convention; mixing the two returns plausible wrong points, which
is why [`split_stream`] is the only place either appears.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

DATA = Path(__file__).parent / "data"
TABLES = DATA / "tetra-tables.safetensors"
CONSTANTS = DATA / "tetra-tables.json"

U64 = np.uint64


def split_stream(codes: np.ndarray, nblocks: int, index_bits: int, gain_bits: int):
    """`(labels, gains)` out of a record's packed code stream.

    Only the byte-aligned case is implemented, which is every Tetra record: 47
    index bits and one gain bit make 48, six bytes a block, so nothing straddles
    a byte boundary. A Ball record is 47 to 50 bits wide and `hfpack` refuses to
    write one, so an unaligned stream here means the two sides disagree about a
    width and it says so instead of guessing.
    """
    width = index_bits + gain_bits
    if width % 8:
        raise ValueError(
            f"{width} bits a block is not byte aligned; this reader handles the aligned "
            "case only, and the packer writes no other kind"
        )
    stride = width // 8
    if codes.size != nblocks * stride:
        raise ValueError(f"{codes.size} code bytes for {nblocks} blocks of {stride}")
    words = codes.reshape(nblocks, stride).astype(U64)
    value = np.zeros(nblocks, dtype=U64)
    for i in range(stride):  # most significant byte first
        value = (value << U64(8)) | words[:, i]
    return value >> U64(gain_bits), (value & U64((1 << gain_bits) - 1)).astype(np.uint32)


@dataclass(frozen=True)
class TetraTables:
    """The universal tables, loaded once per process."""

    order: np.ndarray
    prefixes: np.ndarray
    suffixes: np.ndarray
    branch_c2: np.ndarray
    branch_s16: np.ndarray
    rows: np.ndarray
    values: np.ndarray
    fingerprint: str
    class_rows: int
    n0_mixed: int
    label_bits: int
    dim: int
    section: int
    fields: dict[str, tuple[int, int]]

    @classmethod
    def load(cls, tables: Path = TABLES, constants: Path = CONSTANTS) -> "TetraTables":
        if not tables.exists():
            raise FileNotFoundError(
                f"{tables} is missing. It is dumped by "
                "`cargo run --release -p llvq-llm --bin tetratables -- "
                "llvq-hf/llvqhf/data/tetra-tables.safetensors`"
            )
        t = load_file(tables)
        m = json.loads(constants.read_text())
        return cls(
            order=t["order"].astype(np.intp),
            prefixes=t["prefixes"].astype(U64),
            suffixes=t["suffixes"].astype(U64),
            branch_c2=t["branch_c2"].astype(U64),
            branch_s16=t["branch_s16"].astype(np.intp),
            rows=t["rows"].astype(U64),
            values=t["values"].astype(np.int32),
            fingerprint=m["tetra_fingerprint"],
            class_rows=int(m["class_rows"]),
            n0_mixed=int(m["n0_mixed"]),
            label_bits=int(m["label_bits"]),
            dim=int(m["dim"]),
            section=int(m["section"]),
            fields={f["name"]: (int(f["lo"]), int(f["width"])) for f in m["fields"]},
        )

    def require_fingerprint(self, stored: str) -> None:
        """Refuse a model whose header names another map.

        The check a reader outside Rust cannot skip: a Tetra word read against
        the wrong tables is in range, decodes to a lattice point, and is wrong.
        """
        if stored.lower() != self.fingerprint.lower():
            raise ValueError(
                f"this model was written by Tetra map {stored}, these tables are "
                f"{self.fingerprint}. The indices would decode to plausible wrong points"
            )

    def field(self, labels: np.ndarray, name: str) -> np.ndarray:
        lo, width = self.fields[name]
        return (labels >> U64(lo)) & U64((1 << width) - 1)

    def decode(self, labels: np.ndarray) -> np.ndarray:
        """`(n, 24)` int32 points in natural order, from `(n,)` uint64 labels."""
        labels = np.asarray(labels, dtype=U64)
        if labels.ndim != 1:
            raise ValueError(f"labels must be one-dimensional, got {labels.shape}")
        if labels.size and int(labels.max()) >> self.label_bits:
            raise ValueError("a label above 47 bits: the gain bit is not part of the map")
        f = {n: self.field(labels, n) for n in ("p", "r", "s8", "b1", "i1", "b2", "i2", "b3", "i3")}
        s8 = f["s8"].astype(np.intp)
        c1 = self.prefixes[s8, f["b1"].astype(np.intp)]
        c2 = self.branch_c2[s8, f["b2"].astype(np.intp)]
        s16 = self.branch_s16[s8, f["b2"].astype(np.intp)]
        c3 = self.suffixes[s16, f["b3"].astype(np.intp)]

        cr, n0 = U64(self.class_rows), U64(self.n0_mixed)
        row1 = self.rows[(cr * f["r"] + f["i1"]).astype(np.intp)]
        mixed = f["i2"] >= n0
        # Both indices are in range whatever the branch, so the gather is safe
        # before the choice: i2 < 2048 and class_rows + i2 − n0 < 2856.
        row2 = np.where(
            mixed,
            self.rows[(cr + f["i2"] - n0).astype(np.intp)],
            self.rows[f["i2"].astype(np.intp)],
        )
        delta = mixed.astype(U64)
        r3 = (f["p"] ^ f["r"] ^ delta) & U64(1)
        row3 = self.rows[(cr * r3 + f["i3"]).astype(np.intp)]

        y = np.empty((labels.size, self.dim), dtype=np.int32)
        for k, (c, row) in enumerate(((c1, row1), (c2, row2), (c3, row3))):
            for j in range(self.section):
                o = f["p"] + U64(2) * ((c >> U64(j)) & U64(1))
                rho = (row >> U64(4 * j)) & U64(15)
                if rho.size and int(rho.max()) >= 8:
                    raise ValueError("a rank past the progression: the row table is not this map's")
                y[:, self.section * k + j] = self.values[o.astype(np.intp), rho.astype(np.intp)]

        natural = np.empty_like(y)
        natural[:, self.order] = y
        return natural
