"""Writes the trained values beside the artifact, never into it.

The `.llvq` format lives in Rust and stays there. This writes what a Rust
write-back tool reads, so no format knowledge is duplicated in Python.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..domain.bits import BitCost


class JsonSink:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def write(self, payload: dict, cost: BitCost) -> str:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "cost": {
                "added_params": cost.added_params,
                "added_bits": cost.added_bits,
                "b_per_param_whole_model": cost.b_per_param_whole_model,
                "is_free": cost.is_free,
            },
            "result": payload,
        }
        self._path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
        return str(self._path)
