"""The modules that hold a packed record, and materialize it at load.

One module per record kind, with **buffer names equal to the tensor names in the
file**, so the ordinary `transformers` loader fills them and no per-parameter hook
is needed. That is the shape `quantizer_aqlm.py` uses, and in `transformers` 5.17
it is the only one left: `create_quantized_param` no longer exists.

Two facts about the loader this leans on, both read in
`transformers/core_model_loading.py`:

* a checkpoint tensor whose key exists in the model is cast to the dtype of the
  **existing** buffer, not to the model dtype. So `row_scales` declared f64 stays
  f64 through a load at f16, which matters: the format stores those scales in f64
  precisely so the product `centroid · row_scale` is the one the encoder saw.
* a non-floating tensor is never cast, so `codes` stays uint8.

After the load, [`materialize`] rebuilds the dense weight and frees the
compressed buffers. What is compressed is the file; the loaded model is dense,
which is the decision recorded in the stage 1 prereg §3.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .dequant import dequantize_int4, dequantize_lattice


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


class LlvqLinear(nn.Module):
    """What the two quantized linear kinds share."""

    def __init__(self, d_out: int, d_in: int, bias: torch.Tensor | None):
        super().__init__()
        self.in_features, self.out_features = d_in, d_out
        self.weight: nn.Parameter | None = None
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is None:
            raise RuntimeError(
                f"{type(self).__name__} was never materialized: the quantizer's "
                "post-load hook did not run, and the weights are still codes"
            )
        return nn.functional.linear(x, self.weight, self.bias)

    def _set_weight(self, w: np.ndarray, dtype: torch.dtype) -> None:
        t = torch.from_numpy(np.ascontiguousarray(w))
        self.weight = nn.Parameter(t.to(dtype), requires_grad=False)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


class TetraLinear(LlvqLinear):
    """A lattice record: codes, row scales, gain centroids, an optional tail."""

    def __init__(self, desc: dict, bias: torch.Tensor | None = None):
        super().__init__(desc["d_out"], desc["d_in"], bias)
        self.desc = desc
        self.register_buffer("codes", torch.empty(desc["code_bytes"], dtype=torch.uint8))
        self.register_buffer("row_scales", torch.empty(desc["d_out"], dtype=torch.float64))
        self.register_buffer("centroids", torch.empty(desc["n_centroids"], dtype=torch.float64))
        if desc["tail_cols"]:
            self.register_buffer(
                "tail", torch.empty(desc["d_out"], desc["tail_cols"], dtype=torch.float32)
            )

    def materialize(self, tables, rotations: dict, dtype: torch.dtype) -> None:
        from .tetra import split_stream

        d = self.desc
        labels, gains = split_stream(
            _np(self.codes), d["d_out"] * d["nblocks"], d["index_bits"], d["gain_bits"]
        )
        rot = rotations.get(d["rotation"]) if d.get("rotation") else None
        if d.get("rotation") and rot is None:
            raise KeyError(f"rotation {d['rotation']} is not in the file")
        w = dequantize_lattice(
            tables.decode(labels),
            gains,
            _np(self.centroids),
            _np(self.row_scales),
            _np(self.tail) if d["tail_cols"] else None,
            d["d_out"],
            d["d_in"],
            rot,
            tables.dim,
        )
        self._set_weight(w, dtype)
        for name in ("codes", "row_scales", "centroids", "tail"):
            if hasattr(self, name):
                delattr(self, name)


class Int4Linear(LlvqLinear):
    """An `Int4G128` record: `w = scale·q + bias` in the natural basis."""

    def __init__(self, desc: dict, bias: torch.Tensor | None = None):
        super().__init__(desc["d_out"], desc["d_in"], bias)
        self.desc = desc
        self.register_buffer(
            "qweight", torch.empty(desc["d_out"], desc["d_in"] // 2, dtype=torch.uint8)
        )
        self.register_buffer(
            "scales", torch.empty(desc["d_out"], desc["groups_per_row"], dtype=torch.float16)
        )
        self.register_buffer(
            "biases", torch.empty(desc["d_out"], desc["groups_per_row"], dtype=torch.float16)
        )

    def materialize(self, tables, rotations: dict, dtype: torch.dtype) -> None:
        d = self.desc
        w = dequantize_int4(
            _np(self.qweight), _np(self.scales), _np(self.biases),
            d["d_out"], d["d_in"], d["group"],
        )
        self._set_weight(w, dtype)
        for name in ("qweight", "scales", "biases"):
            delattr(self, name)


class RotationTable(nn.Module):
    """One `(signs, small)` pair, under the name the file gives it."""

    def __init__(self, n: int, k: int):
        super().__init__()
        self.register_buffer("signs", torch.empty(n, dtype=torch.float64))
        self.register_buffer("small", torch.empty(k, k, dtype=torch.float64))


class RotationStore(nn.Module):
    """`llvq.rotations.<key>.signs` and `.small`, shared by every record.

    A module and not a side load, so the keys of the file are the keys of the
    model and the ordinary loader fills them. `nn.ModuleDict` keys may be any
    string without a dot, and a rotation key is `<d_in>_<seed in hex>`.
    """

    def __init__(self, rotations: dict):
        super().__init__()
        self.rotations = nn.ModuleDict(
            {key: RotationTable(int(d["n"]), int(d["odd"])) for key, d in rotations.items()}
        )

    def tables(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {k: (_np(m.signs), _np(m.small)) for k, m in self.rotations.items()}
