"""Learning rate over the run. Pure arithmetic, no framework."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class WarmupCosine:
    """Linear warmup, then cosine decay to `final_ratio` of the peak."""

    peak: float
    total_steps: int
    warmup_steps: int = 0
    final_ratio: float = 0.1

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not 0 <= self.warmup_steps <= self.total_steps:
            raise ValueError("warmup_steps must lie inside the run")
        if self.peak <= 0:
            raise ValueError("peak must be positive")

    def at(self, step: int) -> float:
        """Rate at `step`, counted from zero."""
        if step < 0:
            raise ValueError("step is counted from zero")
        if step < self.warmup_steps:
            return self.peak * (step + 1) / self.warmup_steps
        span = self.total_steps - self.warmup_steps
        if span <= 0:
            return self.peak
        t = min(1.0, (step - self.warmup_steps) / span)
        cosine = 0.5 * (1.0 + math.cos(math.pi * t))
        return self.peak * (self.final_ratio + (1.0 - self.final_ratio) * cosine)


@dataclass(frozen=True)
class Constant:
    """One rate for the whole run. The control arm of any schedule claim."""

    peak: float

    def at(self, step: int) -> float:
        if step < 0:
            raise ValueError("step is counted from zero")
        return self.peak
