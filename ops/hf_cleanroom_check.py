#!/usr/bin/env python3
"""Stage 5's gate, minus the Hub: a clean environment installs the package and loads.

    uv venv /tmp/cleanroom --python 3.12
    VIRTUAL_ENV=/tmp/cleanroom uv pip install ./llvq-tetra torch transformers
    /tmp/cleanroom/bin/python ops/hf_cleanroom_check.py \
        llvq-tetra/tests/fixtures/mini <packed dir> <bin/run dump>

The interpreter must be the clean one and not the repository's: the first thing
this file does is refuse to run from anywhere but `site-packages`. That guard is
the whole point. Running it under `uv run` inside `llvq-tetra/` would import the
working tree, and the question asked here is precisely what a stranger gets.

Nothing here imports from the repository. The point is that `pip install llvq-tetra`
and nothing else is enough to read a Tetra file, which is what a reviewer will
try first and what the in-tree guide of stage 6 requires.

Two levels. The 148 KB `fixtures/mini`, which proves the install, the
registration and a forward pass with no GPU, no ninja and no compiler, and which
a reviewer can run without fetching anything. Then the real 4B on the dense CPU
path against the 256 ids of `bin/run`, which is the gate itself.

Give it `fixtures/tiny` and it fails by name, "q_proj is 4096 by 96, the record
is 4 by 88": that object describes nothing on purpose and is for the packer tests.
Pointing this script at it is how the gap was found on 2026-10-01.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM

import llvq_tetra  # noqa: F401  registers the quantizer and its config

QUANT_MODULES = ("TetraLinear", "Int4Linear", "FusedTetraLinear", "Int4FusedLinear")


def fixture(d: Path) -> int:
    cfg = AutoConfig.from_pretrained(d)
    print(f"  config read, model_type {cfg.model_type}, "
          f"quant_method {cfg.quantization_config['quant_method']}")
    model, info = AutoModelForCausalLM.from_pretrained(
        d, dtype=torch.float32, output_loading_info=True
    )
    # The lists, not the absence of an exception. With the method unregistered
    # `transformers` only warns, reinitializes every dense weight it reports
    # MISSING, and the forward pass below then passes on random numbers.
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        if info[key]:
            print(f"REFUSED: {key} is not empty: {sorted(info[key])}", file=sys.stderr)
            return 1
    n_lin = sum(1 for m in model.modules() if type(m).__name__ in QUANT_MODULES)
    if n_lin == 0:
        print("REFUSED: no record was replaced, the method did not run", file=sys.stderr)
        return 1
    print(f"  loaded, {n_lin} records replaced, no key missing or unexpected, "
          f"{sum(p.numel() for p in model.parameters())} parameters")
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        out = model(ids).logits
    if out.shape[:2] != (1, 4) or not torch.isfinite(out).all():
        print(f"REFUSED: logits {tuple(out.shape)}, finite "
              f"{bool(torch.isfinite(out).all())}", file=sys.stderr)
        return 1
    print(f"  forward ok, logits {tuple(out.shape)}, "
          f"range [{out.min():.3f} ; {out.max():.3f}]")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    print(f"python {sys.version.split()[0]}, torch {torch.__version__}, "
          f"transformers {transformers.__version__}")
    for name in ("ninja", "accelerate"):
        print(f"  {name} present: {importlib.util.find_spec(name) is not None}")
    print(f"llvq_tetra from {Path(llvq_tetra.__file__).parent}")
    if "site-packages" not in str(Path(llvq_tetra.__file__)):
        print("REFUSED: llvq_tetra is not the installed copy, the test is void",
              file=sys.stderr)
        return 2

    print("\n== the fixture ==")
    rc = fixture(Path(argv[1]))
    if rc or len(argv) < 4:
        return rc

    print("\n== the 4B, dense on the CPU, against bin/run ==")
    from llvq_tetra import comparetokens, gentokens

    dump = "/tmp/cleanroom-tokens.json"
    rc = gentokens.main(["gentokens", argv[2], "--new", "64", "--dtype", "f32",
                         "--device", "cpu", "--dump", dump])
    if rc:
        return rc
    return comparetokens.main(["comparetokens", argv[3], dump])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
