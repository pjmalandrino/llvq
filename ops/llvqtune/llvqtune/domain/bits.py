"""The rate a training run costs, in the accounting hard rule 6 demands.

Every trainable set declares its own cost. A set that adds no parameter
declares zero. This is a type, not a convention, so a new training mode
cannot ship without stating what it spends.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BitCost:
    """Extra bits a trained artifact carries, over the artifact it started from.

    `added_params` counts values that were not in the file before.
    `b_per_param_whole_model` is the figure hard rule 6 asks for: extra bits
    divided by the model's total parameter count, embedding included.
    """

    added_params: int
    added_bits: int
    model_params: int

    def __post_init__(self) -> None:
        if self.added_params < 0 or self.added_bits < 0:
            raise ValueError("a cost is never negative")
        if self.model_params <= 0:
            raise ValueError("model_params must be positive")

    @property
    def b_per_param_whole_model(self) -> float:
        return self.added_bits / self.model_params

    @property
    def is_free(self) -> bool:
        """True when the trained file has the same width as the one before it."""
        return self.added_bits == 0

    def __str__(self) -> str:
        if self.is_free:
            return "free: 0 added parameters, 0.000000 b/param"
        return (
            f"{self.added_params} added parameters, "
            f"+{self.b_per_param_whole_model:.6f} b/param whole model"
        )


ZERO_COST = BitCost(added_params=0, added_bits=0, model_params=1)
