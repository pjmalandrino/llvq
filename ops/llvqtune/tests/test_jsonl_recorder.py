"""The journal line a checkpoint leaves, with and without a gauge."""

import json

from llvqtune.adapters.jsonl_recorder import JsonlRecorder


def lines(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


def test_a_checkpoint_line_carries_the_gauge(tmp_path):
    journal = tmp_path / "j.jsonl"
    JsonlRecorder(journal, echo=False).checkpoint(
        200, "/out/sigma.json", {"max_memory_allocated": 74_000_000_000}
    )
    (record,) = lines(journal)
    assert record["event"] == "checkpoint"
    assert record["index"] == 200
    assert record["path"] == "/out/sigma.json"
    assert record["gauge"] == {"max_memory_allocated": 74_000_000_000}


def test_without_a_gauge_the_line_has_no_gauge_key(tmp_path):
    journal = tmp_path / "j.jsonl"
    JsonlRecorder(journal, echo=False).checkpoint(200, "/out/sigma.json", None)
    (record,) = lines(journal)
    assert "gauge" not in record
