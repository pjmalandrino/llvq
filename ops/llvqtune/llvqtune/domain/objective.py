"""What the run minimizes.

`errmap` measured that the layer objective and the model disagree on the same
knob: the closed form optimum is 0.999 and the perplexity minimum is 1.02
(docs/mesures/gain-scale-0.6b-2026-09-15.txt). Every objective here reads the
model's own output, never a layer residual, because of that measurement.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import Tensor


@runtime_checkable
class Objective(Protocol):
    """A scalar the optimizer descends."""

    @property
    def name(self) -> str: ...

    @property
    def needs_teacher(self) -> bool:
        """True when `__call__` reads reference logits.

        A run that binds no teacher and an objective that needs one is refused
        at wiring time, not at step 400.
        """
        ...

    def __call__(
        self, student: Tensor, teacher: Tensor | None, targets: Tensor | None
    ) -> Tensor:
        """Loss for one batch. Exactly one of `teacher` or `targets` is read."""
        ...

    def value(self, loss: Tensor) -> float:
        """The loss as a Python float, for the journal."""
        ...
