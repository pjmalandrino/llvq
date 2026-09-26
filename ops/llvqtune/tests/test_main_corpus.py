"""`main` binds the corpus `--corpus` names, and no other.

The two arms of this ladder differ by one flag. A flag that silently kept
reading DCLM would produce a plausible sigma, a plausible journal and a
measurement of nothing, two hours and four dollars later.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import llvqtune.adapters.dclm_corpus as dclm_corpus  # noqa: E402
import llvqtune.adapters.mmlu_corpus as mmlu_corpus  # noqa: E402
import llvqtune.domain.loop as loop  # noqa: E402
import llvqtune.trainables as trainables  # noqa: E402
from llvqtune.__main__ import main  # noqa: E402
from llvqtune.adapters.mix_corpus import MixCorpus  # noqa: E402
from llvqtune.trainables.row_scales import RowScales  # noqa: E402

from test_main_gauge import _Loader, _Stop, _Tokenizer  # noqa: E402
from test_main_pairing import _configs, _tf  # noqa: E402
from test_pairing import QWEN3_8B  # noqa: E402


class _Generic:
    name = "dclm-edu"

    def __init__(self, *a, **k):
        self.tokens_per_batch = 8

    def batches(self, count, seed):
        return iter(())


class _Task(_Generic):
    name = "mmlu-aux"


def _wire(monkeypatch):
    seen = {}

    def fake_run(**kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(_tf(), "AutoConfig",
                        _configs({"student": QWEN3_8B, "teacher": QWEN3_8B}))
    monkeypatch.setattr(_tf(), "AutoModelForCausalLM", _Loader)
    monkeypatch.setattr(_tf(), "AutoTokenizer", _Tokenizer)
    monkeypatch.setattr(dclm_corpus, "DclmCorpus", _Generic)
    monkeypatch.setattr(mmlu_corpus, "MmluAuxCorpus", _Task)
    monkeypatch.setattr(trainables, "build", lambda mode, **k: RowScales(k["shapes"]))
    monkeypatch.setattr(loop, "run", fake_run)
    return seen


def _main(tmp_path, *extra):
    with pytest.raises(_Stop):
        main(["--student", "student", "--teacher", "teacher", "--device", "cpu",
              "--out", str(tmp_path / "sigma.json"), *extra])


def test_the_default_is_the_corpus_every_published_arm_ran_on(monkeypatch, tmp_path):
    seen = _wire(monkeypatch)
    _main(tmp_path)
    assert seen["corpus"].name == "dclm-edu"


def test_mmlu_aux_binds_the_task_format_corpus(monkeypatch, tmp_path):
    seen = _wire(monkeypatch)
    _main(tmp_path, "--corpus", "mmlu-aux")
    assert seen["corpus"].name == "mmlu-aux"


def test_mix_binds_both_and_carries_its_ratio(monkeypatch, tmp_path):
    seen = _wire(monkeypatch)
    _main(tmp_path, "--corpus", "mix", "--mix-ratio", "0.25")
    assert isinstance(seen["corpus"], MixCorpus)
    assert seen["corpus"].name == "mix0.25"


def test_a_plan_that_outruns_its_corpus_is_refused_with_exit_2(monkeypatch, tmp_path, capsys):
    class _Short(_Task):
        def batches_available(self, seed):
            return 3

    _wire(monkeypatch)
    monkeypatch.setattr(mmlu_corpus, "MmluAuxCorpus", _Short)
    code = main(["--student", "student", "--teacher", "teacher", "--device", "cpu",
                 "--corpus", "mmlu-aux", "--steps", "9507",
                 "--out", str(tmp_path / "sigma.json")])
    assert code == 2
    assert "refused: the corpus holds about 3 batches" in capsys.readouterr().err


def test_an_unknown_corpus_is_refused_by_name(monkeypatch, tmp_path, capsys):
    _wire(monkeypatch)
    with pytest.raises(SystemExit):
        _main(tmp_path, "--corpus", "wikitext2")
    assert "invalid choice" in capsys.readouterr().err
