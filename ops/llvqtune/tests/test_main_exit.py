"""What `main` returns once the loop is done, and the modes it refuses before loading anything.

The harness of `test_main_gauge`: a one-layer Qwen3 on the CPU for both checkpoints, a fake
corpus, and a `run` that returns the outcome the test chooses. `main` writes `sigma.json` for real.
"""

import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import llvqtune.adapters.dclm_corpus as dclm_corpus  # noqa: E402
import llvqtune.domain.loop as loop  # noqa: E402
import llvqtune.trainables as trainables  # noqa: E402
from llvqtune.__main__ import main  # noqa: E402
from llvqtune.domain.bits import BitCost  # noqa: E402
from llvqtune.domain.loop import Outcome  # noqa: E402
from llvqtune.trainables.row_scales import RowScales  # noqa: E402

from test_main_gauge import _Corpus, _Loader, _Tokenizer  # noqa: E402
from test_main_pairing import _NoWeights, _configs, _tf  # noqa: E402
from test_pairing import QWEN3_8B  # noqa: E402


def _outcome(steps_run, losses):
    return Outcome(
        steps_run=steps_run, first_loss=losses[0], last_loss=losses[-1],
        losses=tuple(losses), seconds=1.0,
        cost=BitCost(added_params=0, added_bits=0, model_params=1000),
        export={"kind": "row_scales", "sigma": {}}, steps_planned=6,
    )


def _wire(monkeypatch, outcome):
    monkeypatch.setattr(_tf(), "AutoConfig",
                        _configs({"student": QWEN3_8B, "teacher": QWEN3_8B}))
    monkeypatch.setattr(_tf(), "AutoModelForCausalLM", _Loader)
    monkeypatch.setattr(_tf(), "AutoTokenizer", _Tokenizer)
    monkeypatch.setattr(dclm_corpus, "DclmCorpus", _Corpus)
    monkeypatch.setattr(trainables, "build", lambda mode, **k: RowScales(k["shapes"]))
    monkeypatch.setattr(loop, "run", lambda **kwargs: outcome)


def _main(tmp_path, *extra):
    return main(["--student", "student", "--teacher", "teacher", "--device", "cpu",
                 "--steps", "6", "--warmup", "2", "--out", str(tmp_path / "sigma.json"), *extra])


def test_a_run_cut_short_exits_6_and_its_file_says_so(monkeypatch, tmp_path, capsys):
    _wire(monkeypatch, _outcome(4, [1.0, 0.9, 0.8, 0.7]))
    assert _main(tmp_path) == 6
    doc = json.loads((tmp_path / "sigma.json").read_text())
    assert doc["steps"] == {"run": 4, "planned": 6}
    assert "ran dry at step 4 of 6" in capsys.readouterr().err


def test_a_full_streaming_run_exits_0_whatever_its_loss_did(monkeypatch, tmp_path):
    rising = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    _wire(monkeypatch, _outcome(6, rising))
    assert _main(tmp_path) == 0
    doc = json.loads((tmp_path / "sigma.json").read_text())
    assert doc["steps"] == {"run": 6, "planned": 6}


def test_on_one_repeated_batch_a_loss_that_does_not_fall_exits_1(monkeypatch, tmp_path):
    _wire(monkeypatch, _outcome(6, [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]))
    assert _main(tmp_path, "--repeat-one-batch") == 1


@pytest.mark.parametrize("mode", ["free_params", "low_rank"])
def test_a_mode_that_saves_nothing_is_refused_before_any_load(mode, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(_tf(), "AutoConfig",
                        _configs({"student": QWEN3_8B, "teacher": QWEN3_8B}))
    monkeypatch.setattr(_tf(), "AutoModelForCausalLM", _NoWeights)
    assert _main(tmp_path, "--mode", mode) == 2
    assert f"mode {mode} exports no trained value" in capsys.readouterr().err
    assert not (tmp_path / "sigma.json").exists()
