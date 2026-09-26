"""The task-format corpus: the block, the splits it refuses, the packing.

No download and no card. The parquet is built here with three rows, so the
shape of the stream is tested against text the test itself wrote.
"""

import pytest

torch = pytest.importorskip("torch")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from llvqtune.adapters.mix_corpus import MixCorpus  # noqa: E402
from llvqtune.adapters.mmlu_corpus import (  # noqa: E402
    MmluAuxCorpus,
    block,
    fetch_split,
)

# What `llvq-llm/src/bin/mmlu.rs` `block()` writes for a worked example, held
# literally. If that function moves, this test is the alarm: training text
# that is not the evaluation's text makes the whole arm unreadable.
EXPECTED = (
    "Which of the following is true?\n"
    "A. alpha\n"
    "B. beta\n"
    "C. gamma\n"
    "D. delta\n"
    "Answer: B\n"
    "\n"
)


class Tok:
    """Characters as token ids. Deterministic, and no tokenizer to download."""

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [ord(c) % 256 for c in text]}


def parquet(tmp_path, rows):
    path = tmp_path / "aux.parquet"
    pq.write_table(
        pa.table(
            {
                "question": [r[0] for r in rows],
                "choices": [r[1] for r in rows],
                "answer": [r[2] for r in rows],
                "subject": ["" for _ in rows],
            }
        ),
        path,
    )
    return path


def rows(n, choices=None):
    return [
        (f"question {i}", choices or ["alpha", "beta", "gamma", "delta"], i % 4)
        for i in range(n)
    ]


def test_the_block_is_the_harness_block():
    assert block(
        "  Which of the following is true?  ",
        ["alpha ", " beta", "gamma", "delta"],
        1,
    ) == EXPECTED


@pytest.mark.parametrize("split", ("dev", "validation", "test"))
def test_a_scored_split_is_refused_by_name(split):
    with pytest.raises(ValueError, match="contamination"):
        fetch_split(split)


def test_an_unknown_split_is_refused():
    with pytest.raises(ValueError, match="unknown mmlu split"):
        fetch_split("train")


def test_batches_have_the_asked_shape_and_count(tmp_path):
    corpus = MmluAuxCorpus(
        Tok(), batch_size=2, seq_len=16, path=parquet(tmp_path, rows(40))
    )
    assert corpus.tokens_per_batch == 32
    assert corpus.name == "mmlu-aux"
    got = list(corpus.batches(5, seed=0))
    assert len(got) == 5
    assert all(tuple(b.shape) == (2, 16) for b in got)
    assert all(b.dtype == torch.long for b in got)


def test_the_stream_is_a_function_of_count_and_seed(tmp_path):
    path = parquet(tmp_path, rows(60))
    def stream(seed):
        c = MmluAuxCorpus(Tok(), batch_size=1, seq_len=32, path=path)
        return [b.tolist() for b in c.batches(4, seed=seed)]

    assert stream(0) == stream(0)
    assert stream(0) != stream(7)


def test_the_header_opens_every_group(tmp_path):
    corpus = MmluAuxCorpus(
        Tok(), batch_size=1, seq_len=8, path=parquet(tmp_path, rows(12)), group=3
    )
    texts = list(corpus._texts(seed=0))
    assert len(texts) == 4
    for text in texts:
        assert text.startswith(
            "The following are multiple choice questions (with answers).\n\n"
        )
        assert text.count("Answer:") == 3


def test_a_row_that_is_not_four_choices_is_skipped(tmp_path):
    path = parquet(
        tmp_path,
        [
            ("good one", ["a", "b", "c", "d"], 0),
            ("short one", ["a", "b"], 0),
            ("out of range", ["a", "b", "c", "d"], 9),
            ("good two", ["a", "b", "c", "d"], 3),
        ],
    )
    corpus = MmluAuxCorpus(Tok(), seq_len=8, path=path, group=1)
    texts = list(corpus._texts(seed=0))
    assert len(texts) == 2
    assert "good one" in texts[0] and "good two" in texts[1]


def test_the_split_running_out_ends_the_stream(tmp_path):
    corpus = MmluAuxCorpus(
        Tok(), batch_size=1, seq_len=4096, path=parquet(tmp_path, rows(4))
    )
    assert list(corpus.batches(10, seed=0)) == []


def test_a_bad_shape_is_refused_before_the_run():
    with pytest.raises(ValueError, match="positive"):
        MmluAuxCorpus(Tok(), batch_size=0, path=__file__)


def test_the_split_says_how_much_it_holds(tmp_path):
    corpus = MmluAuxCorpus(
        Tok(), batch_size=2, seq_len=1024, path=parquet(tmp_path, rows(4))
    )
    # 99,842 rows at 247.9 tokens, over 2,048 tokens a batch.
    assert corpus.batches_available(0) == 12_085
    # seed 3 starts 29,919 rows in, so it holds a quarter less.
    assert corpus.batches_available(3) == 8_463
    assert corpus.batches_available(3) < corpus.batches_available(0)


class Fixed:
    """A corpus that yields a fixed label, to read the interleave."""

    def __init__(self, label, tokens=8, available=1000):
        self.tokens_per_batch = tokens
        self.name = label
        self._label = label
        self._available = available

    def batches(self, count, seed):
        for i in range(min(count, self._available)):
            yield f"{self._label}{i}"


def test_the_mix_alternates_at_one_half():
    mix = MixCorpus(Fixed("m"), Fixed("d"), 0.5)
    assert mix.name == "mix0.5"
    assert list(mix.batches(6, seed=0)) == ["d0", "m0", "d1", "m1", "d2", "m2"]


@pytest.mark.parametrize(
    "ratio,expected",
    (
        (0.0, ["d0", "d1", "d2", "d3"]),
        (1.0, ["m0", "m1", "m2", "m3"]),
        (0.25, ["d0", "d1", "d2", "m0"]),
    ),
)
def test_the_mix_holds_its_ratio(ratio, expected):
    assert list(MixCorpus(Fixed("m"), Fixed("d"), ratio).batches(4, 0)) == expected


def test_the_mix_is_a_function_of_count_and_seed():
    a = list(MixCorpus(Fixed("m"), Fixed("d"), 0.5).batches(5, 3))
    b = list(MixCorpus(Fixed("m"), Fixed("d"), 0.5).batches(5, 3))
    assert a == b


def test_a_side_running_dry_ends_the_mix():
    mix = MixCorpus(Fixed("m", available=1), Fixed("d"), 0.5)
    assert list(mix.batches(8, seed=0)) == ["d0", "m0", "d1"]


def test_the_mix_refuses_two_shapes():
    with pytest.raises(ValueError, match="same batch shape"):
        MixCorpus(Fixed("m", tokens=8), Fixed("d", tokens=16), 0.5)


def test_the_mix_refuses_a_ratio_outside_zero_one():
    with pytest.raises(ValueError, match="ratio"):
        MixCorpus(Fixed("m"), Fixed("d"), 1.5)
