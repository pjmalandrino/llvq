"""The reference: the dense model the student is fitted to."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..domain.types import Tensor


@runtime_checkable
class TeacherPort(Protocol):
    """Reference logits. Frozen, never trained, may be absent."""

    def logits(self, ids: Tensor) -> Tensor: ...
