"""Where the trained values go."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..domain.bits import BitCost


@runtime_checkable
class SinkPort(Protocol):
    """Writes a `Trainable.export()` where the write-back tool will read it.

    Nothing here touches the `.llvq` file. The format lives in Rust and stays
    there. This writes plain data beside it, and a Rust tool folds it in.
    """

    def write(self, payload: dict[str, object], cost: BitCost) -> str:
        """Persist the result. Returns the path written."""
        ...
