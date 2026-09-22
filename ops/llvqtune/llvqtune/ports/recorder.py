"""Where a run says what it did."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RecorderPort(Protocol):
    """One line per event. Journals are the repository's unit of evidence."""

    def opened(self, header: dict[str, object]) -> None:
        """Called once, before the first step, with the run's full wiring."""
        ...

    def step(self, index: int, loss: float, lr: float) -> None: ...

    def checkpoint(
        self, index: int, path: str, gauge: dict[str, object] | None
    ) -> None:
        """A partial write landed at `path`. `gauge` is the device memory then,
        or None when no gauge is bound."""
        ...

    def closed(self, summary: dict[str, object]) -> None: ...
