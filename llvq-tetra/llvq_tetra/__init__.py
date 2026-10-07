"""Reading a Tetra-quantized LLVQ model outside the Rust engine.

Stage 1 of `docs/plan-transformers.md`. The `.llvq` format lives in
`llvq-artifact`; `hfpack` rewrites a sealed file as a safetensors directory that
stays compressed, and this package reads that directory in `transformers`.

The pieces, in the order the weights travel:

* `tetra` holds the map from a 47-bit label to a point of the Leech lattice. The
  tables are dumped from `llvq_search::tetra` by `bin/tetratables` and shipped
  here; nothing is re-derived, because a second derivation is a second thing to
  keep bit-exact.
* `reader` opens a packed directory: the record table of
  `config.json`'s `quantization_config`, the tensors, the rotation tables.
* `dequant` rebuilds a matrix the way `llvq_artifact::decode_matrix` does, in
  f64, un-rotating before it narrows to f32.
* `quantizer` registers `LlvqQuantizer` with `transformers` and swaps the
  `nn.Linear` layers for modules that hold the compressed tensors.

Bit-exactness against the Rust decoder is a gate, not an aspiration:
`python -m llvq_tetra.checkdense` compares one SHA-256 per record against the
digests `bin/hfdense` writes from `decode_matrix` itself.
"""

import importlib.util

from .tetra import TetraTables
from .reader import PackedModel

__all__ = ["TetraTables", "PackedModel"]

# Importing this package registers the method with `transformers`, so
# `AutoModelForCausalLM.from_pretrained` on a packed directory works after
# `import llvq_tetra` and nothing else. That is how an out-of-tree quantizer is
# reached, and it was missing until 2026-10-01: every entry point of this
# package imported `.quantizer` by hand, so four stages of measurements passed
# while a plain `import llvq_tetra` registered nothing. Worse, the failure is soft.
# `transformers` only warns, "Unknown quantization type, got llvq ... we will
# skip the quantization", then raises about a corrupted checkpoint fifty lines
# later. `tests/test_registration.py` holds it from a fresh interpreter.
#
# Guarded for one case only: `torch` or `transformers` not installed at all,
# which a `--no-deps` install can produce, and where `PackedModel` still reads a
# file with numpy and safetensors. Any other import failure raises. A
# `transformers` too old for this package fails here on a missing submodule,
# `transformers.core_model_loading`, and swallowing that would leave the method
# unregistered, so `from_pretrained` would load a random model without a word:
# the trap of 2026-10-03, reopened by a version.
try:
    from . import quantizer as _quantizer  # noqa: F401
except ModuleNotFoundError as e:
    _root = (e.name or "").partition(".")[0]
    if _root not in ("torch", "transformers") or importlib.util.find_spec(_root) is not None:
        raise
    _quantizer = None
else:
    __all__.append("quantizer")
