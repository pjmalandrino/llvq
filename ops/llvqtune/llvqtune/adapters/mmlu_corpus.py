"""MMLU-format prompts, from the split MMLU ships for training.

The row scales trained on DCLM-edu bought +3.15 pp
(`docs/mesures/dclm-rowscales-2026-09-20.txt`) while perplexity fell from
17.05 to 9.26. The generic-text objective has given what it can: the object
sits nine points under f16 on MMLU and 1.0074 times f16 in perplexity. This
corpus asks the other question — whether distilling the teacher on text shaped
like the *task* moves the task, at the same zero bits and the same file width.

## Which split, and why not the others

`cais/mmlu` ships four: `auxiliary_train` (99,842 rows), `dev` (285),
`validation` (1,531) and `test` (14,042). Only `auxiliary_train` may be read
here, and [`fetch_split`] refuses the other three by name.

`dev` is where the harness draws its five worked examples (`bin/mmlu.rs`), and
`test` is the bar. Training on either is contamination, not a lead.

The overlap between `auxiliary_train` and the scored splits was measured
before a line of this file was written (*measured*, 2026-09-23): 0 rows of
`test`, `dev` or `validation` share question **and** choices with any
`auxiliary_train` row. 14 rows of `test` share the question text alone, all of
them a bare stem — "Which of the following is true?" — under four unrelated
options. So the split is disjoint in substance, and the filter that would drop
those 14 would drop nothing.

## The shape

One block is what `bin/mmlu.rs` `block()` writes, character for character:

```text
<question>
A. <choice>
B. <choice>
C. <choice>
D. <choice>
Answer: B

```

Blocks are grouped `GROUP` at a time behind the harness's own header, which
is the 5-shot prefix the scored question is read under, and the stream is then
packed into fixed-length sequences the way `DclmCorpus` packs text. The header
carries no subject: `auxiliary_train` ships none (every row's `subject` is the
empty string, *measured*), and the evaluation prefix names one. That is the
one respect in which this corpus is not the evaluation's own distribution, and
it is written here rather than discovered in a result.

## Repeatability

`batches` is a pure function of `(count, seed)`, the contract `CorpusPort`
states. The seed picks a row offset and rows are consumed in order from there.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import torch

REPO = "cais/mmlu"
TRAIN_SPLIT = "auxiliary_train"
SCORED_SPLITS = ("dev", "validation", "test")
ROWS_IN_SPLIT = 99_842
TOKENS_A_QUESTION = 247.9
"""Header included, under the Qwen3-4B tokenizer (*measured*, 2026-09-23:
297,438 ids for 1,200 rows in groups of six).

The whole split is therefore about 24.75 M tokens, and the DCLM arm of
2026-09-20 consumed 19,470,336 — 0.79 of a pass. So the arm that matches it
token for token fits, at seed 0, with 27 % to spare and no repetition. At
seed 3 the offset alone eats 30 % of the split and it does not fit, which is
what `batches_available` exists to say before the run rather than after it."""
LETTERS = ("A", "B", "C", "D")
HEADER = "The following are multiple choice questions (with answers).\n\n"
GROUP = 6


def fetch_split(split: str = TRAIN_SPLIT) -> Path:
    """The parquet for `split`, downloading it if this machine lacks it.

    Refuses the scored splits by name. A run that trains on `dev` would train
    on the five shots every scored prompt carries, and nothing downstream —
    not the fingerprint, not the paired bar — would show it.
    """
    if split in SCORED_SPLITS:
        raise ValueError(
            f"{split!r} is scored by the harness; training on it is contamination"
        )
    if split != TRAIN_SPLIT:
        raise ValueError(f"unknown mmlu split {split!r}")
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=REPO,
            filename=f"all/{split}-00000-of-00001.parquet",
            repo_type="dataset",
        )
    )


def block(question: str, choices, answer: int) -> str:
    """One worked example, as `bin/mmlu.rs` writes it.

    Kept in step with that function by `tests/test_mmlu_corpus.py`, which holds
    the expected string literally.
    """
    text = f"{question.strip()}\n"
    for letter, choice in zip(LETTERS, choices):
        text += f"{letter}. {choice.strip()}\n"
    return text + f"Answer: {LETTERS[answer]}\n\n"


class MmluAuxCorpus:
    """Packs MMLU-format prompts into fixed-length token batches."""

    def __init__(
        self,
        tokenizer,
        batch_size: int = 1,
        seq_len: int = 2048,
        path: Path | None = None,
        read_rows: int = 256,
        device: str = "cpu",
        group: int = GROUP,
        header: str = HEADER,
    ) -> None:
        if batch_size <= 0 or seq_len <= 0:
            raise ValueError("batch_size and seq_len must be positive")
        if group <= 0:
            raise ValueError("group must be positive")
        self._tokenizer = tokenizer
        self._batch = batch_size
        self._seq = seq_len
        self._read_rows = read_rows
        self._device = device
        self._group = group
        self._header = header
        self._path = Path(path) if path is not None else fetch_split()
        if not self._path.exists():
            raise FileNotFoundError(f"{self._path} is missing after the fetch")

    @property
    def tokens_per_batch(self) -> int:
        return self._batch * self._seq

    @property
    def name(self) -> str:
        return "mmlu-aux"

    def batches_available(self, seed: int) -> int:
        """How many batches the split can still yield from `seed`'s offset.

        *Estimated*, from a token count measured once on this split, not read
        from the file: counting for real means tokenizing 47 MB, which is
        minutes, and this is checked before every run. It is deliberately the
        quantity the wiring refuses on — a run that outruns its corpus stops
        early, writes a shorter sigma and looks like a completed arm.
        """
        rows_left = ROWS_IN_SPLIT - (seed * 9973) % ROWS_IN_SPLIT
        return int(rows_left * TOKENS_A_QUESTION) // self.tokens_per_batch

    def _rows(self, seed: int) -> Iterator[dict]:
        import pyarrow.parquet as pq

        offset = (seed * 9973) % ROWS_IN_SPLIT
        handle = pq.ParquetFile(self._path)
        skipped = 0
        for chunk in handle.iter_batches(
            batch_size=self._read_rows, columns=["question", "choices", "answer"]
        ):
            values = chunk.to_pylist()
            if skipped + len(values) <= offset:
                skipped += len(values)
                continue
            start = max(0, offset - skipped)
            skipped += len(values)
            yield from values[start:]

    def _texts(self, seed: int) -> Iterator[str]:
        """Groups of `group` blocks, each behind the harness's header."""
        pending: list[str] = []
        for row in self._rows(seed):
            choices = row["choices"]
            answer = row["answer"]
            # A malformed row is skipped, not raised on: this runs on a card,
            # two hours in, and one bad row out of 99,842 must not end the run.
            if len(choices) != 4 or not isinstance(answer, int) or not 0 <= answer < 4:
                continue
            pending.append(block(row["question"], choices, answer))
            if len(pending) == self._group:
                yield self._header + "".join(pending)
                pending = []
        if pending:
            yield self._header + "".join(pending)

    def batches(self, count: int, seed: int) -> Iterator[torch.Tensor]:
        """Yield exactly `count` batches, or stop early if the split runs out."""
        need = self._batch * self._seq
        buffer: list[int] = []
        produced = 0
        for text in self._texts(seed):
            buffer.extend(self._tokenizer(text, add_special_tokens=False)["input_ids"])
            while len(buffer) >= need:
                bloc = buffer[:need]
                del buffer[:need]
                yield torch.tensor(bloc, dtype=torch.long, device=self._device).view(
                    self._batch, self._seq
                )
                produced += 1
                if produced >= count:
                    return
