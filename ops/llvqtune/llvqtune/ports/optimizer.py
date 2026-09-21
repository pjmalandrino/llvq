"""Descent. The only place the domain lets a framework compute a gradient."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..domain.types import Tensor


@runtime_checkable
class OptimizerPort(Protocol):
    """Applies one update to the bound `Trainable`."""

    def zero_grad(self) -> None: ...

    def backward(self, loss: Tensor) -> None: ...

    def step(self, lr: float) -> None:
        """One update at rate `lr`. The schedule owns the rate, not the optimizer."""
        ...
