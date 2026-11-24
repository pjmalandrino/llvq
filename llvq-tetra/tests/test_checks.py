"""The refusals of the gate scripts, which run without a model, a GPU or an install.

`checkq4guards` and `ops/hf_cleanroom_check.py` load a model or demand an installed
copy, so their gates cannot run here. What can is the part that decides whether a
gate ran at all: a mutant applied at the wrong number of sites, and an argument
list that would skip the 4B.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

CLEANROOM = Path(__file__).resolve().parents[2] / "ops" / "hf_cleanroom_check.py"


def test_a_mutant_at_the_wrong_site_count_is_refused():
    from llvq_tetra.checkq4guards import SPLIT, mutate

    src = f"head\n{SPLIT}(x) {{ a; a; }}\nkernel void tv_q4_metal(y) {{ a; }}\n"
    want = f"head\n{SPLIT}(x) {{ b; b; }}\nkernel void tv_q4_metal(y) {{ a; }}\n"
    assert mutate(src, "a;", "b;", 2) == want
    with pytest.raises(ValueError, match="2 sites for 'a;' in the tiled kernel, not 1"):
        mutate(src, "a;", "b;", 1)


def test_the_cleanroom_check_refuses_a_packed_directory_without_its_dump(capsys):
    spec = importlib.util.spec_from_file_location("hf_cleanroom_check", CLEANROOM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["hf_cleanroom_check.py", "fixture", "packed"]) == 2
    assert "the 4B gate needs both" in capsys.readouterr().err
