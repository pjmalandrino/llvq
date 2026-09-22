"""`main` checks the pairing before it loads a single weight.

The loaders are replaced: the config loader returns the 8B and 4B configs,
and the weight loader fails the test if it is ever reached. A `main` that
loads first and checks later, or never checks, fails here.
"""

import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from llvqtune.__main__ import main  # noqa: E402

from test_pairing import QWEN3_4B, QWEN3_8B  # noqa: E402


class _Config:
    def __init__(self, values):
        self._values = values

    def to_dict(self):
        return dict(self._values)


def _tf():
    """The live module. transformers 5.x swaps its `sys.modules` entry the
    first time a model class is built, so a reference taken at import time
    can be a stale object that `main` never reads."""
    return sys.modules["transformers"]


def _configs(by_name):
    class _AutoConfig:
        @staticmethod
        def from_pretrained(name, *a, **k):
            return _Config(by_name[name])

    return _AutoConfig


class _NoWeights:
    @staticmethod
    def from_pretrained(*a, **k):
        raise AssertionError("weights were loaded before the pairing was checked")


def test_a_mismatched_teacher_is_refused_before_any_weight(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(_tf(), "AutoConfig",
                        _configs({"student": QWEN3_8B, "teacher": QWEN3_4B}))
    monkeypatch.setattr(_tf(), "AutoModelForCausalLM", _NoWeights)
    code = main(["--student", "student", "--teacher", "teacher",
                 "--device", "cpu", "--out", str(tmp_path / "sigma.json")])
    assert code == 2
    assert "hidden_size" in capsys.readouterr().err


def test_a_matching_teacher_goes_on_to_load_the_student(monkeypatch, tmp_path):
    """The check does not refuse the twin: the next thing reached is the loader."""
    monkeypatch.setattr(_tf(), "AutoConfig",
                        _configs({"student": QWEN3_8B, "teacher": QWEN3_8B}))
    monkeypatch.setattr(_tf(), "AutoModelForCausalLM", _NoWeights)
    with pytest.raises(AssertionError, match="weights were loaded"):
        main(["--student", "student", "--teacher", "teacher",
              "--device", "cpu", "--out", str(tmp_path / "sigma.json")])
