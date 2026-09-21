"""The data: batches of token ids."""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

from ..domain.types import Tensor


@runtime_checkable
class CorpusPort(Protocol):
    """A finite, repeatable stream of token batches.

    Repeatable matters: two arms compared on different draws are not compared.
    An adapter that cannot replay its own stream from a seed does not belong
    here.
    """

    @property
    def tokens_per_batch(self) -> int: ...

    def batches(self, count: int, seed: int) -> Iterator[Tensor]: ...
