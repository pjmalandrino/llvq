"""What a quantizer does with an LLVQ checkpoint, in one place.

The package's own quantizer (`quantizer.py`) and an in-tree `transformers` integration call these
four functions and nothing else, so the two cannot drift:

* `target_device`: the single device the model will run on, read from `device_map`;
* `replace_modules`: before the load, swap the layers so the file's keys are the model's keys;
* `weight_conversions`: the tensors that are dequantized during the load, if any;
* `finalize`: after the load, compress the layers on a GPU, or rebuild them dense on the CPU.

On `"mps"` and `"cuda"` the weights stay compressed in memory. On `"cpu"` they are rebuilt dense.
The checkpoint loads on the CPU in every case, because the codes are transcoded to the kernels'
layout there, and the model moves to its device at the end.
"""

from __future__ import annotations

import os

import torch
from torch import nn

from .modules import Int4Linear, RotationStore, TetraLinear

# Activation blocks one threadgroup stages for the lattice matvec, and columns for the 4-bit one:
# the values the kernels were validated with.
TILE_BLOCKS = 64
INT4_TILE_COLUMNS = 2048


def target_device(device_map) -> str:
    """`"cuda"` or `"mps"` to keep the weights compressed, `"cpu"` to rebuild them dense.

    Without a `device_map`, `LLVQ_HF_FUSED=1` still asks for the compressed path on whatever
    accelerator exists, as `llvq-tetra` 0.1.0 did, and `LLVQ_HF_DEVICE` names it.
    """
    if device_map is None:
        want = os.environ.get("LLVQ_HF_FUSED", "0")
        if want not in ("0", "1"):
            raise ValueError(f"LLVQ_HF_FUSED={want!r} is neither 0 nor 1")
        if want == "0":
            return "cpu"
        device_map = os.environ.get("LLVQ_HF_DEVICE", "auto")
    if device_map == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    values = set(device_map.values()) if isinstance(device_map, dict) else {device_map}
    kinds = set()
    for value in values:
        name = str(value)
        if isinstance(value, int) or name.isdigit() or name.startswith("cuda"):
            kinds.add("cuda")
        elif name.startswith("mps"):
            kinds.add("mps")
        elif name == "cpu":
            kinds.add("cpu")
        else:
            raise ValueError(f"LLVQ models cannot be placed on {value!r}")
    if len(kinds) != 1:
        raise ValueError(f"LLVQ models run on a single device, the device_map spans {sorted(kinds)}")
    return kinds.pop()


def replace_modules(model: nn.Module, quantization_config, device: str) -> nn.Module:
    """Swap each quantized `nn.Linear` for the layer that holds its codes, and on Metal the
    vocabulary for the module that keeps it 4-bit."""
    records = {desc["prefix"]: desc for desc in quantization_config.records.values()}
    replaced = 0
    for name, module in list(model.named_modules()):
        desc = records.get(name)
        if desc is None:
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"{name} is a {type(module).__name__}, LLVQ only quantizes nn.Linear layers")
        if (module.out_features, module.in_features) != (desc["d_out"], desc["d_in"]):
            raise ValueError(
                f"{name} is {module.out_features} by {module.in_features}, "
                f"the checkpoint stores {desc['d_out']} by {desc['d_in']}"
            )
        layer_class = TetraLinear if desc["kind"] == "tetra" else Int4Linear
        model.set_submodule(name, layer_class(desc, module.bias))
        replaced += 1
    if replaced != len(records):
        raise ValueError(f"{replaced} layers replaced for {len(records)} records in the checkpoint")

    # The rotations are tensors of the checkpoint, shared by name across layers.
    if quantization_config.rotations:
        model.llvq = RotationStore(quantization_config.rotations)

    if device == "mps":
        _keep_vocabulary_4bit(model, quantization_config)
    return model


def _keep_vocabulary_4bit(model: nn.Module, quantization_config) -> None:
    from .vocab import Int4TiedHead, Int4Vocab

    for desc in quantization_config.raw.values():
        if desc["encoding"] != "quant":
            continue
        prefix = desc["prefix"]
        module = model.get_submodule(prefix)
        if not isinstance(module, nn.Embedding):
            raise NotImplementedError(
                f"{prefix} is stored in 4 bits and is a {type(module).__name__}; only an embedding "
                "can stay 4-bit on Metal"
            )
        vocab = Int4Vocab(desc)
        model.set_submodule(prefix, vocab)
        replaced = [prefix]
        if getattr(model.config, "tie_word_embeddings", False) and isinstance(getattr(model, "lm_head", None), nn.Linear):
            model.set_submodule("lm_head", Int4TiedHead(vocab))
            replaced.append("lm_head")
            # The head reads the table itself, so there is nothing left for transformers to tie.
            model.config.tie_word_embeddings = False
        # transformers lists the tied parameters when it builds the model, and after the load it looks each one up.
        # The table is no longer a `weight`, so its ties go.
        tied = getattr(model, "all_tied_weights_keys", None)
        if isinstance(tied, dict):
            for target, source in list(tied.items()):
                if any(k.startswith(p + ".") for k in (target, source) for p in replaced):
                    tied.pop(target)


def weight_conversions(quantization_config, dtype: torch.dtype, device: str) -> list:
    """The tensors stored as 4-bit groups outside the linear layers, the embedding among them,
    dequantized during the load: three checkpoint keys become one dense tensor. On Metal they load
    as stored instead, and this returns nothing."""
    if device == "mps":
        return []
    from transformers.core_model_loading import WeightConverter

    from .conversions import Int4Dequantize

    conversions = []
    for tensor_name, desc in quantization_config.raw.items():
        if desc["encoding"] != "quant":
            continue
        prefix = desc["prefix"]
        conversions.append(
            WeightConverter(
                [f"{prefix}.qweight", f"{prefix}.scales", f"{prefix}.biases"],
                tensor_name,
                operations=[Int4Dequantize(desc, dtype)],
            )
        )
    return conversions


def finalize(model: nn.Module, tables, dtype: torch.dtype, device: str) -> nn.Module:
    """After the load: compress the layers on a GPU and move the model there, or rebuild them dense."""
    from .vocab import Int4Vocab

    rotations = model.llvq.tables() if hasattr(model, "llvq") else {}
    if device == "cpu":
        for module in model.modules():
            if isinstance(module, (TetraLinear, Int4Linear)):
                module.materialize(tables, rotations, dtype)
    else:
        _compress(model, tables, rotations, dtype, device)
    for module in model.modules():
        if isinstance(module, Int4Vocab):
            module.out_dtype = dtype
    if hasattr(model, "llvq"):
        del model.llvq
    if device != "cpu":
        model.to(device)
    return model


def rotation_for(signs, small, device: str, consumers: int = 1):
    """The rotation on `device`: a Metal kernel, kept for its `consumers`, or torch ops on CUDA."""
    if device == "mps":
        from .mps import RotationKernel

        return RotationKernel(signs, small, device, consumers)
    from .fused import Rotation

    return Rotation(signs, small, device)


def _compress(model: nn.Module, tables, rotations: dict, dtype: torch.dtype, device: str) -> None:
    from .fused import FusedTetraLinear, Int4FusedLinear

    built = {}
    dense = 0
    # How many projections read each rotation: three for q, k, v and two for gate, up in a Qwen3 layer.
    consumers = {}
    for module in model.modules():
        if isinstance(module, TetraLinear) and module.desc.get("rotation"):
            consumers[module.desc["rotation"]] = consumers.get(module.desc["rotation"], 0) + 1
    for name, module in list(model.named_modules()):
        if isinstance(module, Int4Linear):
            if Int4FusedLinear.supports(device):
                model.set_submodule(name, Int4FusedLinear.from_loaded(module, INT4_TILE_COLUMNS, device))
            else:
                module.materialize(tables, rotations, dtype)
                dense += 1
            continue
        if not isinstance(module, TetraLinear):
            continue
        key = module.desc.get("rotation")
        if key and key not in built:
            signs, small = rotations[key]
            built[key] = rotation_for(signs, small, device, consumers[key])
        model.set_submodule(name, FusedTetraLinear.from_loaded(module, built.get(key), TILE_BLOCKS, device))
    if dense:
        import warnings

        warnings.warn(
            f"{dense} 4-bit LLVQ layers have no {device} kernel yet and were rebuilt dense; "
            "the lattice layers stay compressed.",
            stacklevel=2,
        )
