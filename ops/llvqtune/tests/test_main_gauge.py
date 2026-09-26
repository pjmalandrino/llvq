"""`main` binds the CUDA gauge to the loop when the device is cuda.

Everything a card would supply is replaced: a one-layer Qwen3 on the CPU
stands in for both checkpoints, the corpus is a fake, and `run` records what
it was handed and stops. No card, no download, no weight.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import llvqtune.adapters.dclm_corpus as dclm_corpus  # noqa: E402
import llvqtune.domain.loop as loop  # noqa: E402
import llvqtune.trainables as trainables  # noqa: E402
from llvqtune.__main__ import main  # noqa: E402
from llvqtune.adapters.cuda_gauge import CudaGauge  # noqa: E402
from llvqtune.trainables.row_scales import RowScales  # noqa: E402

from test_main_pairing import _configs, _tf  # noqa: E402
from test_pairing import QWEN3_8B  # noqa: E402


class _Stop(Exception):
    pass


def _tiny():
    config = transformers.Qwen3Config(
        vocab_size=32, hidden_size=48, intermediate_size=48,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=24,
    )
    model = transformers.Qwen3ForCausalLM(config)
    model.to = lambda *a, **k: model          # stays on the CPU
    return model


class _Loader:
    @staticmethod
    def from_pretrained(*a, **k):
        return _tiny()


class _Tokenizer:
    @staticmethod
    def from_pretrained(*a, **k):
        return object()


class _Corpus:
    def __init__(self, *a, **k):
        self.tokens_per_batch = 8
        self.name = "fake_corpus"

    def batches(self, count, seed):
        return iter(())


def _wire(monkeypatch):
    seen = {}

    def fake_run(**kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(_tf(), "AutoConfig",
                        _configs({"student": QWEN3_8B, "teacher": QWEN3_8B}))
    monkeypatch.setattr(_tf(), "AutoModelForCausalLM", _Loader)
    monkeypatch.setattr(_tf(), "AutoTokenizer", _Tokenizer)
    monkeypatch.setattr(dclm_corpus, "DclmCorpus", _Corpus)
    monkeypatch.setattr(trainables, "build", lambda mode, **k: RowScales(k["shapes"]))
    monkeypatch.setattr(loop, "run", fake_run)
    return seen


def _main(device, tmp_path):
    with pytest.raises(_Stop):
        main(["--student", "student", "--teacher", "teacher", "--device", device,
              "--out", str(tmp_path / "sigma.json")])


def test_on_cuda_the_loop_receives_the_cuda_gauge(monkeypatch, tmp_path):
    seen = _wire(monkeypatch)
    _main("cuda", tmp_path)
    assert isinstance(seen["gauge"], CudaGauge)


def test_off_cuda_no_gauge_is_bound(monkeypatch, tmp_path):
    seen = _wire(monkeypatch)
    _main("cpu", tmp_path)
    assert seen["gauge"] is None
