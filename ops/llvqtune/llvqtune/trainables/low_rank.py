"""Row 20, Q6b: a low-rank correction added to the decoded weights, EoRA or RILQ.

This is the LoRA-shaped lead, and unlike rows 14 and 17 it is not free. The
roadmap prices it at +0.263 b/param whole model at r = 32 in f16, +0.131 at
r = 16. `cost` recomputes that from the shapes rather than trusting the note.

## Where it stands against the budget

The product triplet caps the model at 3.00 b/param. On the DCLM base at 2.8126
there is room for r = 16 and not for r = 32. On the 2026-09-19 object at
2.9686 there is room for neither. Which base this runs on decides whether the
lead is playable at all, so `cost` is read before the run, not after.
"""

from __future__ import annotations

import torch

from ..domain.bits import BitCost

F16_BITS = 16


class LowRank:
    """`W = frozen + A @ B`, with `B` at zero so the run starts at the artifact."""

    def __init__(
        self,
        shapes: dict[str, tuple[int, int]],
        rank: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        stored_bits: int = F16_BITS,
    ) -> None:
        """`shapes` maps a matrix to `(rows, d_in)`, the full input width."""
        if rank <= 0:
            raise ValueError("rank must be positive")
        if not shapes:
            raise ValueError("no matrix to train")
        self._shapes = dict(shapes)
        self._rank = rank
        self._stored_bits = stored_bits
        generator = torch.Generator(device="cpu").manual_seed(0)
        self._a = {}
        self._b = {}
        for matrix, (rows, cols) in shapes.items():
            a = torch.randn(rows, rank, generator=generator, dtype=dtype) / rank**0.5
            self._a[matrix] = a.to(device).requires_grad_(True)
            self._b[matrix] = torch.zeros(
                rank, cols, device=device, dtype=dtype, requires_grad=True
            )

    @property
    def name(self) -> str:
        return f"low_rank_r{self._rank}"

    def parameters(self) -> list[torch.Tensor]:
        return list(self._a.values()) + list(self._b.values())

    def weight(self, matrix: str, frozen: torch.Tensor) -> torch.Tensor:
        if frozen.requires_grad:
            raise ValueError(
                f"{matrix}: decoded directions carry a gradient; "
                "the Leech decoder is a table and is never differentiated"
            )
        delta = self._a[matrix] @ self._b[matrix]
        return frozen + delta.to(frozen.dtype)

    def cost(self, model_params: int) -> BitCost:
        """Every factor is a value the served file did not hold before."""
        added = sum(
            rows * self._rank + self._rank * cols
            for rows, cols in self._shapes.values()
        )
        return BitCost(
            added_params=added,
            added_bits=added * self._stored_bits,
            model_params=model_params,
        )

    def export(self) -> dict[str, object]:
        return {
            "kind": "low_rank",
            "rank": self._rank,
            "apply": "W := decode(codes) + A @ B, new records",
            "shapes": {m: list(s) for m, s in self._shapes.items()},
        }
