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

    @property
    def name(self) -> str:
        """What the journal calls this stream.

        Two arms of the same ladder can differ by the corpus and by nothing
        else — `dclm-edu` against `mmlu-aux`. A journal that does not name it
        cannot tell them apart afterwards, and neither can a reader.
        """
        ...

    def batches(self, count: int, seed: int) -> Iterator[Tensor]: ...
