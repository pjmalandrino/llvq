"""The point of variation: what a run learns.

Row 14 of ROADMAP-QUALITY learns the row scales the file already holds. Row 17
learns every free parameter in it. Row 20 adds low-rank terms. They differ in
three places and agree everywhere else, so those three places are this
protocol and the rest is shared.

## The invariant

`weight` receives the frozen decoded directions and returns the weight the
forward pass uses. The gradient reaches the trainable set through that return
value and stops there. It never reaches the directions, because the Leech
decoder is a table lookup and not a differentiable function. An adapter that
lets a gradient into the directions is a bug, and `tests/test_frozen.py`
is what catches it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .bits import BitCost
from .types import Tensor


@runtime_checkable
class Trainable(Protocol):
    """A set of parameters a run optimizes, and the rate it costs."""

    @property
    def name(self) -> str:
        """Short identifier, used in journals and to select the mode."""
        ...

    def parameters(self) -> list[Tensor]:
        """Every tensor the optimizer updates. May be empty for a control arm."""
        ...

    def weight(self, matrix: str, frozen: Tensor) -> Tensor:
        """Build the weight of `matrix` from its frozen decoded directions.

        Called once per matrix per forward pass. `frozen` never carries a
        gradient. The return value does, unless the set is empty.
        """
        ...

    def cost(self, model_params: int) -> BitCost:
        """What this set adds to the served file. Zero when it adds no value."""
        ...

    def export(self) -> dict[str, object]:
        """The result, in the shape the write-back tool reads.

        Returns plain data, not tensors, so a run can be journalled and
        replayed without the framework that produced it.
        """
        ...
