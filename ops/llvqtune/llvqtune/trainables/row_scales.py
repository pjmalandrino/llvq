"""Row 14 of ROADMAP-QUALITY: learn the row scales the file already holds.

This is the paper's own fine-tuning. Its note reads, transcribed in
docs/llvq-paper-notes.md: "The fine-tuning here is no more than learning the
per-column scales (< 0.001 bit/weight, ~52M tokens). It is not end-to-end
training." The paper measures +2.1 pp on Qwen3-4B, our exact model.

## What is learned, and what is not

A Tetra block reconstructs as `centroids[g] * row_scale * u`. This trains a
multiplier on `row_scale`, one per row, initialized at 1.0. No code moves, no
index byte changes, and the decoder stays byte-identical. The written file has
the same width as the one it started from, so the cost is zero.

The multiplier never reaches the tail. A row scale does not multiply the tail
in the artifact, and `artstat` measured the tail non-empty on all 252 records
of the 4B. Scaling it here and folding the result into `row_scales` would
write back something other than what was trained.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..domain.bits import BitCost


class RowScales:
    """Per-row multipliers on the artifact's `row_scales`, initialized at 1."""

    def __init__(
        self,
        shapes: dict[str, tuple[int, int]],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """`shapes` maps a matrix to `(rows, lattice_columns)`."""
        if not shapes:
            raise ValueError("no matrix to train")
        for matrix, (rows, lattice) in shapes.items():
            if rows <= 0 or lattice <= 0:
                raise ValueError(f"{matrix}: degenerate shape {(rows, lattice)}")
            if lattice % 24 != 0:
                raise ValueError(
                    f"{matrix}: {lattice} lattice columns is not a multiple of 24"
                )
        self._shapes = dict(shapes)
        self._sigma = {
            matrix: torch.ones(rows, device=device, dtype=dtype, requires_grad=True)
            for matrix, (rows, _) in shapes.items()
        }

    @property
    def name(self) -> str:
        return "row_scales"

    def parameters(self) -> list[torch.Tensor]:
        return list(self._sigma.values())

    def weight(self, matrix: str, frozen: torch.Tensor) -> torch.Tensor:
        """Scale the lattice columns of `frozen`, pass the tail through."""
        if frozen.requires_grad:
            raise ValueError(
                f"{matrix}: decoded directions carry a gradient; "
                "the Leech decoder is a table and is never differentiated"
            )
        sigma = self._sigma[matrix]
        rows, lattice = self._shapes[matrix]
        if frozen.shape[0] != rows:
            raise ValueError(
                f"{matrix}: {frozen.shape[0]} rows, {rows} scales"
            )
        if frozen.shape[1] < lattice:
            raise ValueError(
                f"{matrix}: {frozen.shape[1]} columns, {lattice} are coded"
            )
        scaled = frozen[:, :lattice] * sigma.unsqueeze(1).to(frozen.dtype)
        if frozen.shape[1] == lattice:
            return scaled
        return torch.cat([scaled, frozen[:, lattice:]], dim=1)

    def linear(self, matrix, x, frozen, bias=None):
        """`x @ (sigma * P)^T` without ever building `sigma * P`.

        The scale is on the output rows, so it commutes through the matmul:
        `x @ diag(sigma) P` equals `sigma * (x @ P)`. Building the weight
        instead puts a full copy of the model, 3.52 G values and 7.05 GB in
        f16, into the autograd graph at every step. On a machine where the
        student, the teacher and the loss already hold more than twenty
        gigabytes, that copy is what tips the run into thrashing.

        The tail breaks the identity, because a row scale does not multiply
        the tail. It is corrected rather than split out: the tail is
        `d_in % 24` columns, 8 or 16 of several thousand, so the correcting
        matmul is negligible while the main one stays contiguous.

            F.linear(x, P)    = A + B          A lattice, B tail
            wanted            = sigma A + B
                              = sigma (A + B) + (1 - sigma) B
        """
        if frozen.requires_grad:
            raise ValueError(
                f"{matrix}: decoded directions carry a gradient; "
                "the Leech decoder is a table and is never differentiated"
            )
        _, lattice = self._shapes[matrix]
        sigma = self._sigma[matrix].to(x.dtype)
        y = F.linear(x, frozen) * sigma
        if frozen.shape[1] > lattice:
            tail = F.linear(x[..., lattice:], frozen[:, lattice:])
            y = y + tail * (1.0 - sigma)
        if bias is not None:
            y = y + bias
        return y

    def cost(self, model_params: int) -> BitCost:
        """Zero. The scales are already in the file; only their values change."""
        return BitCost(added_params=0, added_bits=0, model_params=model_params)

    def export(self) -> dict[str, object]:
        """Multipliers to fold into `row_scales[i]`, matrix by matrix."""
        return {
            "kind": "row_scales",
            "apply": "row_scales[i] *= sigma[i]",
            "sigma": {
                matrix: [float(v) for v in tensor.detach().cpu()]
                for matrix, tensor in self._sigma.items()
            },
        }
