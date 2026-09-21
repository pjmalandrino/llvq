"""The registry. Adding a training mode means adding one entry here.

Every mode implements `llvqtune.domain.trainable.Trainable`, so the loop,
the objectives, the corpus and the journal are shared and untouched.
"""

from __future__ import annotations

MODES = ("row_scales", "free_params", "low_rank")


def build(mode: str, **kwargs):
    """Instantiate a mode by name. Refuses an unknown name instead of guessing."""
    if mode == "row_scales":
        from .row_scales import RowScales

        return RowScales(**kwargs)
    if mode == "free_params":
        from .free_params import FreeParams

        return FreeParams(**kwargs)
    if mode == "low_rank":
        from .low_rank import LowRank

        return LowRank(**kwargs)
    raise ValueError(f"unknown mode {mode!r}; known modes are {', '.join(MODES)}")
