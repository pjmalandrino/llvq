"""Row 17, Q6a: learn every free parameter the file already holds.

A superset of row 14. Beside the row scales it trains the tail, the columns
`d_in % 24` that no lattice code covers and that the artifact stores verbatim.
On the 4B that is 16,957,440 values against 1,105,920 scales, so the tail is
94 percent of the set.

## The reason this is not the default

52M tokens over 18M parameters is 3 tokens a parameter. Row 14 sits at 47.
The tail holds real weights, not format parameters, so training it is closer
to fine-tuning the model than to fitting the format, and it overfits the
calibration set at that ratio. Run row 14 first and read what it gives.
"""

from __future__ import annotations

import torch

from ..domain.bits import BitCost


class FreeParams:
    """Row scales and tail columns, both already present in the file."""

    def __init__(
        self,
        shapes: dict[str, tuple[int, int]],
        tails: dict[str, torch.Tensor],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """`tails` gives each matrix's current tail, shape `(rows, d_in % 24)`."""
        if not shapes:
            raise ValueError("no matrix to train")
        missing = set(shapes) - set(tails)
        if missing:
            raise ValueError(f"no tail supplied for {sorted(missing)}")
        self._shapes = dict(shapes)
        self._sigma = {
            matrix: torch.ones(rows, device=device, dtype=dtype, requires_grad=True)
            for matrix, (rows, _) in shapes.items()
        }
        self._tail = {
            matrix: tails[matrix].to(device=device, dtype=dtype).clone().requires_grad_(True)
            for matrix in shapes
        }

    @property
    def name(self) -> str:
        return "free_params"

    def parameters(self) -> list[torch.Tensor]:
        return list(self._sigma.values()) + list(self._tail.values())

    def weight(self, matrix: str, frozen: torch.Tensor) -> torch.Tensor:
        if frozen.requires_grad:
            raise ValueError(
                f"{matrix}: decoded directions carry a gradient; "
                "the Leech decoder is a table and is never differentiated"
            )
        sigma = self._sigma[matrix]
        rows, lattice = self._shapes[matrix]
        if frozen.shape[0] != rows:
            raise ValueError(f"{matrix}: {frozen.shape[0]} rows, {rows} scales")
        scaled = frozen[:, :lattice] * sigma.unsqueeze(1).to(frozen.dtype)
        tail = self._tail[matrix]
        if tail.numel() == 0:
            return scaled
        return torch.cat([scaled, tail.to(frozen.dtype)], dim=1)

    def cost(self, model_params: int) -> BitCost:
        """Zero. Widths do not move; only the values written into them do."""
        return BitCost(added_params=0, added_bits=0, model_params=model_params)

    def export(self) -> dict[str, object]:
        return {
            "kind": "free_params",
            "apply": "row_scales[i] *= sigma[i]; tail := tail",
            "sigma": {
                m: [float(v) for v in t.detach().cpu()] for m, t in self._sigma.items()
            },
            "tail_shapes": {
                m: list(t.shape) for m, t in self._tail.items()
            },
        }
