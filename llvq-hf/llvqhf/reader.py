"""A packed LLVQ directory: the record table, the tensors, the rotations.

What `hfpack` writes is a `model.safetensors` and a `config.json` whose
`quantization_config` block describes every record. This module is the only place
that reads that description, so a change of layout has one reader to update.

Tensors are opened lazily. A 4B holds 1264 of them and 1.41 GB, and a checker
that wants one matrix at a time must not pay for the rest.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from .dequant import dequantize_int4, dequantize_lattice
from .tetra import TetraTables, split_stream

MODEL_FILE = "model.safetensors"
CONFIG_FILE = "config.json"
QUANT_METHOD = "llvq"
CODE_ORDER = "msb_first_dense"


class PackedModel:
    """One packed directory, open for reading."""

    def __init__(self, path: str | Path, tables: TetraTables | None = None):
        self.path = Path(path)
        self.config = json.loads((self.path / CONFIG_FILE).read_text())
        try:
            self.qc = self.config["quantization_config"]
        except KeyError as e:
            raise ValueError(f"{self.path / CONFIG_FILE} has no quantization_config") from e
        if self.qc.get("quant_method") != QUANT_METHOD:
            raise ValueError(f"quant_method is {self.qc.get('quant_method')!r}, not {QUANT_METHOD!r}")
        if self.qc.get("code_order") != CODE_ORDER:
            raise ValueError(
                f"code_order is {self.qc.get('code_order')!r}, which this reader does not unpack"
            )
        self.tables = tables or TetraTables.load()
        # Before a single index is read: a Tetra word against the wrong map is in
        # range and decodes to the wrong point.
        self.tables.require_fingerprint(self.qc["tetra_fingerprint"])
        self.block_dim = int(self.qc["block_dim"])
        if self.block_dim != self.tables.dim:
            raise ValueError(f"block_dim {self.block_dim} against a map of {self.tables.dim}")
        self.records: dict = self.qc["records"]
        self.rotations: dict = self.qc["rotations"]
        self.raw: dict = self.qc["raw"]
        self._file = safe_open(self.path / MODEL_FILE, framework="np")
        self._names = set(self._file.keys())
        self._rot_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def __enter__(self) -> "PackedModel":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        self._file = None

    def tensor(self, name: str) -> np.ndarray:
        if name not in self._names:
            raise KeyError(f"{name} is not in {self.path / MODEL_FILE}")
        return self._file.get_tensor(name)

    def rotation(self, key: str | None) -> tuple[np.ndarray, np.ndarray] | None:
        """The `(signs, small)` tables of a rotation, read once per key."""
        if not key:
            return None
        if key not in self._rot_cache:
            if key not in self.rotations:
                raise KeyError(f"record points at rotation {key}, which the file does not carry")
            self._rot_cache[key] = (
                self.tensor(f"llvq.rotations.{key}.signs"),
                self.tensor(f"llvq.rotations.{key}.small"),
            )
        return self._rot_cache[key]

    def points(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """`(points, gains)` of a Tetra record: `(d_out·nblocks, 24)` and `(n,)`."""
        d = self.records[name]
        if d["kind"] != "tetra":
            raise ValueError(f"{name} is a {d['kind']} record and has no lattice points")
        labels, gains = split_stream(
            self.tensor(f"{d['prefix']}.codes"),
            d["d_out"] * d["nblocks"],
            d["index_bits"],
            d["gain_bits"],
        )
        return self.tables.decode(labels), gains

    def dequantize(self, name: str) -> np.ndarray:
        """`(d_out, d_in)` f32 in the natural basis, whatever the record's kind."""
        d = self.records[name]
        prefix = d["prefix"]
        if d["kind"] == "tetra":
            points, gains = self.points(name)
            tail = self.tensor(f"{prefix}.tail") if d["tail_cols"] else None
            return dequantize_lattice(
                points,
                gains,
                self.tensor(f"{prefix}.centroids"),
                self.tensor(f"{prefix}.row_scales"),
                tail,
                d["d_out"],
                d["d_in"],
                self.rotation(d.get("rotation")),
                self.block_dim,
            )
        if d["kind"] == "int4g128":
            return dequantize_int4(
                self.tensor(f"{prefix}.qweight"),
                self.tensor(f"{prefix}.scales"),
                self.tensor(f"{prefix}.biases"),
                d["d_out"],
                d["d_in"],
                d["group"],
            )
        raise ValueError(f"{name}: unknown record kind {d['kind']}")

    def dequantize_raw(self, name: str) -> np.ndarray:
        """A carried tensor as f32, whatever the encoding the file used."""
        d = self.raw[name]
        dims = tuple(d["dims"])
        if d["encoding"] == "f16":
            return self.tensor(name).astype(np.float32)
        if d["encoding"] == "quant":
            rows, row_len = d["rows"], dims[-1]
            flat = dequantize_int4(
                self.tensor(f"{d['prefix']}.qweight"),
                self.tensor(f"{d['prefix']}.scales"),
                self.tensor(f"{d['prefix']}.biases"),
                rows,
                row_len,
                d["group"],
            )
            return flat.reshape(dims)
        raise ValueError(f"{name}: unknown raw encoding {d['encoding']}")

    def quantized_raw(self) -> list[str]:
        """The carried tensors the file stores group-affine, the embedding among them."""
        return [n for n, d in self.raw.items() if d["encoding"] == "quant"]
