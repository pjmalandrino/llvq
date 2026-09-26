"""Yields one fixed batch over and over. The gradient sanity check.

A streaming corpus gives a different text at every step, so the loss at step 1
and the loss at step N are measured on different data and their difference says
nothing about the parameters. Overfitting a single batch removes that confound:
if the loss on one fixed batch does not fall, the gradient is not reaching the
trainable set, and no amount of real training will help.
"""

from __future__ import annotations

from typing import Iterator


class RepeatCorpus:
    """Wraps a corpus and replays its first batch."""

    def __init__(self, inner, warm_seed: int = 0) -> None:
        self._inner = inner
        self._warm_seed = warm_seed
        self._batch = None

    @property
    def tokens_per_batch(self) -> int:
        return self._inner.tokens_per_batch

    @property
    def name(self) -> str:
        return f"repeat1({getattr(self._inner, 'name', 'inner')})"

    def batches(self, count: int, seed: int) -> Iterator:
        if self._batch is None:
            for first in self._inner.batches(1, self._warm_seed):
                self._batch = first
                break
        if self._batch is None:
            return
        for _ in range(count):
            yield self._batch
