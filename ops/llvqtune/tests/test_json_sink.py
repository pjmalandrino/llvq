"""The document `rowscale` folds, and the count of steps beside it."""

import json

from llvqtune.adapters.json_sink import JsonSink
from llvqtune.domain.bits import BitCost

COST = BitCost(added_params=0, added_bits=0, model_params=1000)


def test_the_steps_sit_beside_the_result(tmp_path):
    out = tmp_path / "sigma.json"
    JsonSink(out).write({"kind": "row_scales", "sigma": {}}, COST, steps=(3, 5))
    doc = json.loads(out.read_text())
    assert doc["steps"] == {"run": 3, "planned": 5}
    assert doc["result"] == {"kind": "row_scales", "sigma": {}}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["sigma.json"]


def test_without_steps_the_document_keeps_its_shape(tmp_path):
    out = tmp_path / "sigma.json"
    JsonSink(out).write({"kind": "row_scales", "sigma": {}}, COST)
    assert sorted(json.loads(out.read_text())) == ["cost", "result"]


def test_a_second_write_replaces_the_first(tmp_path):
    out = tmp_path / "sigma.json"
    sink = JsonSink(out)
    sink.write({"kind": "row_scales", "sigma": {}}, COST, steps=(200, 9507))
    sink.write({"kind": "row_scales", "sigma": {}}, COST, steps=(9507, 9507))
    assert json.loads(out.read_text())["steps"] == {"run": 9507, "planned": 9507}
