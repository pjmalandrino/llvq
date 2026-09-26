"""One JSON object a line. The journal a run leaves behind."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


class JsonlRecorder:
    def __init__(self, path: str | Path | None = None, echo: bool = True) -> None:
        self._path = Path(path) if path is not None else None
        self._echo = echo
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, record: dict) -> None:
        record["at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        line = json.dumps(record, sort_keys=True)
        if self._path is not None:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        if self._echo:
            print(line, file=sys.stderr, flush=True)

    def opened(self, header: dict) -> None:
        self._write({"event": "opened", **header})

    def step(self, index: int, loss: float, lr: float) -> None:
        self._write({"event": "step", "index": index, "loss": loss, "lr": lr})

    def checkpoint(self, index: int, path: str, gauge: dict | None) -> None:
        record = {"event": "checkpoint", "index": index, "path": path}
        if gauge is not None:
            record["gauge"] = gauge
        self._write(record)

    def closed(self, summary: dict) -> None:
        self._write({"event": "closed", **summary})
