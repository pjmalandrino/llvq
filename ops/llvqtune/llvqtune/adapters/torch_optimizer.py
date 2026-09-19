"""Adam over a `Trainable`'s parameters, with the rate set per step."""

from __future__ import annotations

import torch


class Adam:
    """The schedule owns the rate, so `step` writes it into every group."""

    def __init__(self, parameters: list[torch.Tensor], weight_decay: float = 0.0) -> None:
        if not parameters:
            raise ValueError("nothing to optimize")
        self._opt = torch.optim.AdamW(
            parameters, lr=1e-3, weight_decay=weight_decay
        )

    def zero_grad(self) -> None:
        self._opt.zero_grad(set_to_none=True)

    def backward(self, loss) -> None:
        loss.backward()

    def step(self, lr: float) -> None:
        for group in self._opt.param_groups:
            group["lr"] = lr
        self._opt.step()
