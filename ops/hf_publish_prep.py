#!/usr/bin/env python3
"""Add to a packed directory what publishing it needs and packing it does not.

    uv run ops/hf_publish_prep.py <packed dir> [--checkpoint <HF repo>] [--revision <sha>]

`hfpack` writes the sealed file's own `config.json` plus a quantization block,
and that is the right thing for a format: the packed directory is then the sealed
file, field for field, and `llvq_hf_check.py` can say so. A Hub repository needs
three things on top, none of which belong in the format.

**The tokenizer files.** `hfpack` carries the two blobs the sealed file carries,
`config.json` and `tokenizer.json`, and writes a 64 byte `tokenizer_config.json`
stub. The real one is 9,732 bytes and holds the chat template, so a repository
published as packed gives a tokenizer where `apply_chat_template` fails. Copied
verbatim from the base checkpoint at a pinned revision.

**`auto_map`, and the shim it points at.** `transformers` has no entry-point
discovery for quantizers, so the method is registered by `import llvq_tetra` and
nothing else, and a caller that imports `transformers` alone loads a RANDOMLY
INITIALIZED model without raising (*measured*,
`docs/mesures/hf-tripwire-2026-10-03.txt`). This does not fix that: the probe
showed `auto_map` cannot, because `transformers` resolves a class from
`model_type` and never consults the map. What it does buy is that
`trust_remote_code=True` starts working, where today it changes nothing: the shim
imports `llvq_tetra` on the way in, so a caller who passes the flag gets a
correct model instead of a random one, and a caller without the package gets an
`ImportError` naming it.

Closing the trap needs an unresolvable `model_type`, which costs the `qwen3`
string, or the in-tree PR, which costs a review. Neither is this script's call.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# The shim that travels with a published model. Two jobs and no third: register
# the method by importing the package, and name the package when it is absent.
# It must NOT set `config_class`: `auto_map` here maps the model and not the
# config, so `AutoConfig` resolves Qwen3's own class and a declared
# `config_class` would make transformers refuse on an inconsistency that has
# nothing to do with loading this file.
SHIM = '''\
"""Importing this registers the LLVQ Tetra quantization method.

`transformers` discovers quantization methods by import and not by entry point,
so a caller that imports `transformers` alone would read this model's
`quant_method: "llvq"`, warn, skip the quantization, reinitialize every weight it
then finds missing, and run. The logits would look ordinary and the model would
be random.

This file is reached through `auto_map` when you pass `trust_remote_code=True`.
It does not protect a caller who passes nothing: `transformers` resolves the
model class from `model_type` and never looks here. The reliable form is to
import the package yourself:

    import llvq_tetra
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("{repo}")
"""

from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

try:
    import llvq_tetra  # noqa: F401  the import is the whole point of this file
except ImportError as e:  # pragma: no cover - the path this file exists for
    raise ImportError(
        "This model is stored in the LLVQ Tetra format and needs its reader: "
        "`pip install llvq-tetra`. Without it, transformers loads a randomly "
        "initialized model and does not raise."
    ) from e


class LlvqQwen3ForCausalLM(Qwen3ForCausalLM):
    """Qwen3's model, unchanged. Importing this module is what matters."""
'''

TOKENIZER_FILES = (
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
)
SHIM_FILE = "modeling_llvq.py"
AUTO_MAP = {"AutoModelForCausalLM": f"{SHIM_FILE[:-3]}.LlvqQwen3ForCausalLM"}


def checkpoint_dir(repo: str, revision: str | None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, revision=revision, allow_patterns=TOKENIZER_FILES))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="hf_publish_prep")
    ap.add_argument("directory")
    ap.add_argument("--checkpoint", default="Qwen/Qwen3-4B")
    ap.add_argument("--revision", default=None,
                    help="pin it; an unpinned tokenizer is a tokenizer that moves")
    ap.add_argument("--repo", default="Pier-Jean/Qwen3-4B-LLVQ-Tetra",
                    help="only to write the example in the shim's docstring")
    a = ap.parse_args(argv[1:])

    d = Path(a.directory)
    cfg_path = d / "config.json"
    if not cfg_path.exists():
        print(f"REFUSED: {cfg_path} is missing, this is not a packed model", file=sys.stderr)
        return 2
    cfg = json.loads(cfg_path.read_text())
    if "quantization_config" not in cfg:
        print("REFUSED: no quantization_config, this is not a packed model", file=sys.stderr)
        return 2

    src = checkpoint_dir(a.checkpoint, a.revision)
    for name in TOKENIZER_FILES:
        s = src / name
        if not s.exists():
            print(f"REFUSED: {a.checkpoint} has no {name}", file=sys.stderr)
            return 1
        before = (d / name).stat().st_size if (d / name).exists() else 0
        shutil.copy(s, d / name)
        print(f"  {name}: {before} -> {(d / name).stat().st_size} bytes")

    (d / SHIM_FILE).write_text(SHIM.format(repo=a.repo))
    print(f"  {SHIM_FILE}: {(d / SHIM_FILE).stat().st_size} bytes")

    cfg["auto_map"] = AUTO_MAP
    cfg_path.write_text(json.dumps(cfg, indent=1))
    print(f"  config.json: auto_map -> {AUTO_MAP}")
    print("\nThis does not close the silent trap. It makes trust_remote_code=True work.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
