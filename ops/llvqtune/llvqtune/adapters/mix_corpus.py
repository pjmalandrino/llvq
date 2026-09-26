"""Two corpora, interleaved on a fixed ratio.

The pure task-format arm is the one that tests the mechanism, and it carries
one risk the generic-text arm does not: scales fitted only on multiple-choice
prompts may move perplexity the wrong way. The mix is the fallback that keeps
both objectives in the same run, and it is the shape `ROADMAP-QUALITY` row L04
asks for (a 50/50 of MMLU-format prompts and generic text).

The interleave is deterministic and carries no randomness: batch `i` comes
from the first corpus when `floor((i + 1) * ratio) > floor(i * ratio)`. At
ratio 0.5 that alternates; at 0.25, one batch in four. Two arms run at the same
seed therefore see the same batches in the same order, which is what
`CorpusPort` promises and what makes an A/B an A/B.
"""

from __future__ import annotations

from math import floor
from typing import Iterator


class MixCorpus:
    """Draws from `first` with probability-free frequency `ratio`."""

    def __init__(self, first, second, ratio: float = 0.5) -> None:
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("ratio must sit in [0, 1]")
        if first.tokens_per_batch != second.tokens_per_batch:
            raise ValueError(
                "the two corpora must yield the same batch shape: "
                f"{first.tokens_per_batch} against {second.tokens_per_batch}"
            )
        self._first = first
        self._second = second
        self._ratio = ratio

    @property
    def tokens_per_batch(self) -> int:
        return self._first.tokens_per_batch

    @property
    def name(self) -> str:
        return f"mix{self._ratio:g}"

    def batches(self, count: int, seed: int) -> Iterator:
        # Each side is asked for the whole count: the split is decided here,
        # and a side that runs dry ends the stream rather than silently
        # handing every remaining batch to the other one.
        a = self._first.batches(count, seed)
        b = self._second.batches(count, seed)
        for index in range(count):
            take_first = floor((index + 1) * self._ratio) > floor(index * self._ratio)
            nxt = next(a, None) if take_first else next(b, None)
            if nxt is None:
                return
            yield nxt
