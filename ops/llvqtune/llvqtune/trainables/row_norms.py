"""Row 14 plus the input axis: the RMSNorm weights, trained beside the row scales.

## Why this mode exists

The paper's note reads "learning the per-column scales". Row 14 trains one
scale per output **row**, and its prereg (`preregistration-tetranu-rowscales`
§6) left open whether that is the paper's axis. This mode adds an axis on the
**input** side without adding a value to the file.

An RMSNorm computes `w * x / rms(x)`. Its weight `w` is a per-column scale on
every matrix that reads its output, applied in the natural basis, before the
input rotation:

    input_layernorm            -> q_proj, k_proj, v_proj
    post_attention_layernorm   -> gate_proj, up_proj
    model.norm                 -> the language head, tied to the embedding

`bin/seal` carries every norm verbatim in f16 (`seal.rs:11-14`) and
`model.rs:1369` applies it before the projections rotate their input. So a
trained multiplier `tau` folds exactly into `w <- w * tau`, and the rate does
not move. The fold is rounded to f16 once, which is the width the file holds.

What the input axis does NOT reach, and why nothing is added for it:

- `down_proj`'s input is `act(gate) * up`. A column scale there is a row scale
  of `up_proj`, which row 14 already trains.
- `o_proj`'s input is the attention output, a mix of `v_proj` rows. A column
  scale there is a `v_proj` row scale under the GQA constraint, and `v_proj`
  is int4 with no `row_scales` (see `torch_model.py`). Not reached here.
- `q_norm` and `k_norm` scale q and k per head dimension after projection,
  which the q and k row scales already cover more finely.

## Write-back

Not done. `bin/rowscale` folds `kind = "row_scales"` only and refuses every
other kind by name, so an export of this mode cannot be half-applied. Folding
`tau` into the carried f16 norms is a raw-tensor rewrite still to be written.
"""

from __future__ import annotations

import torch

from ..domain.bits import BitCost
from .row_scales import RowScales


class RowNorms:
    """Row scales on the lattice matrices, multipliers on the RMSNorm weights."""

    def __init__(
        self,
        shapes: dict[str, tuple[int, int]],
        norms: dict[str, int],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """`norms` maps a norm module to its width."""
        if not norms:
            raise ValueError("no norm to train; use row_scales")
        for norm, width in norms.items():
            if width <= 0:
                raise ValueError(f"{norm}: degenerate width {width}")
        self._rows = RowScales(shapes, device=device, dtype=dtype)
        self._widths = dict(norms)
        self._tau = {
            norm: torch.ones(width, device=device, dtype=dtype, requires_grad=True)
            for norm, width in norms.items()
        }

    @property
    def name(self) -> str:
        return "row_norms"

    @property
    def norms(self) -> dict[str, int]:
        return dict(self._widths)

    def parameters(self) -> list[torch.Tensor]:
        return self._rows.parameters() + list(self._tau.values())

    def weight(self, matrix: str, frozen: torch.Tensor) -> torch.Tensor:
        return self._rows.weight(matrix, frozen)

    def linear(self, matrix, x, frozen, bias=None):
        return self._rows.linear(matrix, x, frozen, bias)

    def scale_norm(self, norm: str, y: torch.Tensor) -> torch.Tensor:
        """`y` is the frozen norm's output `w * x / rms(x)`; returns it times `tau`.

        Multiplying the output rather than the weight is the same function,
        `(w * tau) * n = tau * (w * n)`, and leaves the norm's own arithmetic,
        including its f32 upcast, untouched.
        """
        tau = self._tau[norm]
        if y.shape[-1] != tau.shape[0]:
            raise ValueError(f"{norm}: width {y.shape[-1]}, {tau.shape[0]} multipliers")
        return y * tau.to(y.dtype)

    def cost(self, model_params: int) -> BitCost:
        """Zero. Row scales and norm weights are both already in the file."""
        return BitCost(added_params=0, added_bits=0, model_params=model_params)

    def export(self) -> dict[str, object]:
        rows = self._rows.export()
        return {
            "kind": "row_norms",
            "apply": {
                "sigma": "row_scales[i] *= sigma[i]",
                "tau": "norm.weight[j] *= tau[j], rounded to f16 once",
            },
            "sigma": rows["sigma"],
            "tau": {
                norm: [float(v) for v in tensor.detach().cpu()]
                for norm, tensor in self._tau.items()
            },
        }
