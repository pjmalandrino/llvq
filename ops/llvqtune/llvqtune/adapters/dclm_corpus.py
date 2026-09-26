"""DCLM-edu, the corpus the LLVQ paper calibrates on.

One shard is already on disk: 776,000 rows in a single row group, 4.68 G
characters, 2,905,491,151 bytes (*measured*, 2026-09-07, llvq-llm/src/corpus.rs).
At the 4.5356 bytes a token the Qwen3-4B tokenizer reads, that shard is about
1.03 G tokens. The paper's fine-tuning consumes ~52 M, so the shard carries
about twenty times the need and nothing has to be downloaded.

## Repeatability

`batches` is a pure function of `(count, seed)`. The seed picks a row offset
and rows are then consumed in order. Seed 0 is the prefix, which is what the
Rust calibration path reads, so an arm run at seed 0 sees the same text the
encoder saw.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import torch

SHARD = (
    Path.home()
    / ".cache/huggingface/hub/datasets--HuggingFaceTB--dclm-edu/snapshots"
    / "dbad8ad71224482740cd9c9d353591adbf62fe04/data/000_00000.parquet"
)
REPO = "HuggingFaceTB/dclm-edu"
SHARD_IN_REPO = "data/000_00000.parquet"
ROWS_IN_SHARD = 776_000


def fetch_shard() -> Path:
    """The shard, downloading it if this machine has never seen it.

    The Rust harness pulls it through `LLVQ_CALIB=dclm-edu`. A training job
    runs in a torch image that holds no Rust binary, so it has to be able to
    pull the same file by itself. Same repo, same path, so both sides read
    one object.
    """
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(repo_id=REPO, filename=SHARD_IN_REPO, repo_type="dataset")
    )


class DclmCorpus:
    """Packs shard text into fixed-length token batches."""

    def __init__(
        self,
        tokenizer,
        batch_size: int = 1,
        seq_len: int = 2048,
        path: Path | None = None,
        read_rows: int = 512,
        device: str = "cpu",
    ) -> None:
        if batch_size <= 0 or seq_len <= 0:
            raise ValueError("batch_size and seq_len must be positive")
        self._tokenizer = tokenizer
        self._batch = batch_size
        self._seq = seq_len
        self._read_rows = read_rows
        # The batch must land where the model is. An adapter is allowed to
        # know the device; the domain and the port are not, and a batch
        # delivered on the wrong one fails at the embedding lookup.
        self._device = device
        if path is not None:
            self._path = Path(path)
        elif SHARD.exists():
            self._path = SHARD
        else:
            self._path = fetch_shard()
        if not self._path.exists():
            raise FileNotFoundError(f"{self._path} is missing after the fetch")

    @property
    def tokens_per_batch(self) -> int:
        return self._batch * self._seq

    @property
    def name(self) -> str:
        return "dclm-edu"

    def _texts(self, seed: int) -> Iterator[str]:
        import pyarrow.parquet as pq

        offset = (seed * 9973) % ROWS_IN_SHARD
        handle = pq.ParquetFile(self._path)
        skipped = 0
        for chunk in handle.iter_batches(
            batch_size=self._read_rows, columns=["text"]
        ):
            values = chunk.column("text").to_pylist()
            if skipped + len(values) <= offset:
                skipped += len(values)
                continue
            start = max(0, offset - skipped)
            skipped += len(values)
            for text in values[start:]:
                if text:
                    yield text

    def batches(self, count: int, seed: int) -> Iterator[torch.Tensor]:
        """Yield exactly `count` batches, or stop early if the shard runs out."""
        need = self._batch * self._seq
        buffer: list[int] = []
        produced = 0
        for text in self._texts(seed):
            buffer.extend(self._tokenizer(text, add_special_tokens=False)["input_ids"])
            while len(buffer) >= need:
                block = buffer[:need]
                del buffer[:need]
                yield torch.tensor(
                    block, dtype=torch.long, device=self._device
                ).view(self._batch, self._seq)
                produced += 1
                if produced >= count:
                    return
