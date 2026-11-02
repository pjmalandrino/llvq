"""Serve a Tetra-quantized LLVQ model in vLLM, compressed on the card.

A prototype, written against vLLM 0.26.0. `docs/plan-vllm.md` holds the design
and the open questions; nothing here has run on a card yet.

vLLM finds the method through the `vllm.general_plugins` entry point this
distribution declares, in every process it starts, before it reads a model
config. `register` is that entry point:

    pip install --no-deps ./llvq-tetra ./llvq-vllm   # inside the vLLM image
    vllm serve <packed directory> --dtype float16

`--no-deps` is not a convenience. `llvq-tetra` pins `torch>=2.14` for its
`transformers` path, and the pinned vLLM image ships torch 2.11.0, which a
resolver would replace and so break every compiled kernel of the image.

The pieces, in the order the weights travel: `plan` maps the file's records to
vLLM's fused layers, `buffers` builds the served layout on the device, `ops`
holds the kernels as torch ops vLLM can compile around, `layer` arms and runs
one layer, and `vllm_glue` is the adapter vLLM calls.
"""

from __future__ import annotations

from .plan import QUANT_METHOD

__all__ = ["QUANT_METHOD", "register"]


def register() -> None:
    """Register the `llvq` quantization method with vLLM. Safe to call twice.

    vLLM loads general plugins once per process and warns that a plugin may be
    loaded in several processes, so this keeps no state of its own.
    """
    try:
        from vllm.model_executor.layers.quantization import register_quantization_config
    except ModuleNotFoundError as e:
        raise ImportError(
            "llvq_vllm.register needs vLLM, which is not installed here. The method is "
            "registered inside a vLLM process by its entry point; the rest of this "
            "package runs without it"
        ) from e

    from .vllm_glue import LlvqVllmConfig

    register_quantization_config(QUANT_METHOD)(LlvqVllmConfig)
