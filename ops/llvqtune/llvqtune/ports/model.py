"""The student: the quantized model whose free parameters are being trained."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..domain.types import Tensor


@runtime_checkable
class ModelPort(Protocol):
    """A forward pass that routes its weights through a `Trainable`."""

    @property
    def matrices(self) -> list[str]:
        """Names of the quantized matrices, in the artifact's own naming."""
        ...

    @property
    def param_count(self) -> int:
        """Total parameters of the model, embedding included. Hard rule 6."""
        ...

    def rows_of(self, matrix: str) -> int: ...

    def lattice_columns_of(self, matrix: str) -> int:
        """Columns the lattice codes cover, so `d_in - d_in % 24`.

        The tail is the remainder. A row scale never multiplies the tail, and
        `artstat` measured that the tail is non-empty on all 252 records of
        the 4B. Scaling it would answer a different question than the one the
        artifact asks.
        """
        ...

    def forward(self, ids: Tensor) -> Tensor:
        """Logits for a batch of token ids."""
        ...
