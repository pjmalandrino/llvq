"""The parameters a layer registers before the load, and their weight loaders.

No vLLM here: a loader is a function `(param, tensor, shard_id=None)`, which is
the signature vLLM's fused layers call (`QKVParallelLinear.load_weights` and
`MergedColumnParallelLinear.load_weights` pass the shard id the model's mapper
stamped on the tensor) and the one `AutoWeightsLoader` calls with two arguments
for an unfused layer. So `tests/test_loading.py` drives these loaders with the
file's own tensor stream and vLLM's own renames, on a Mac.

Every parameter records which shards its loader received, and `require_loaded`
refuses a hole. vLLM checks for missing weights only when a model is not
quantized, so without this a renamed or forgotten tensor would leave
uninitialized memory in a served weight, and nothing would fail.
"""

from __future__ import annotations

import torch
from torch import nn

from .plan import ParamSpec, Slot, rotation_param, shard_index

DTYPES = {
    "uint8": torch.uint8,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def make_loader(slots: dict[int, Slot], what: str):
    """A weight loader for one flat parameter: shard id to slot, check, copy, record."""

    def loader(param: nn.Parameter, loaded: torch.Tensor, shard_id=None) -> None:
        index = shard_index(shard_id)
        if index is None:
            if len(slots) != 1:
                raise ValueError(f"{what}: a tensor without a shard id for a fused layer")
            (index,) = slots
        if index not in slots:
            raise KeyError(f"{what}: no slot for shard {shard_id!r}")
        slot = slots[index]
        if tuple(loaded.shape) != tuple(slot.shape):
            raise ValueError(f"{what}[{shard_id!r}]: checkpoint shape {tuple(loaded.shape)}, "
                             f"want {tuple(slot.shape)}")
        if param.llvq_loaded.get(index):
            raise ValueError(f"{what}[{shard_id!r}] loaded twice")
        param.data.reshape(-1)[slot.offset: slot.offset + slot.numel].copy_(loaded.reshape(-1))
        param.llvq_loaded[index] = True

    return loader


def new_param(t: torch.Tensor, loader, expected) -> nn.Parameter:
    """A frozen parameter that carries its loader and the shards it waits for."""
    p = nn.Parameter(t, requires_grad=False)
    p.llvq_loaded = {}
    p.llvq_expected = tuple(expected)
    p.weight_loader = loader
    return p


def require_loaded(param: nn.Parameter, what: str) -> None:
    missing = [i for i in param.llvq_expected if not param.llvq_loaded.get(i)]
    if missing:
        raise RuntimeError(
            f"{what}: shards {missing} were never loaded. vLLM does not check this for a "
            "quantized model, and serving the uninitialized memory would give plausible text"
        )


def register_layer_params(holder: nn.Module, specs: list[ParamSpec], prefix: str) -> None:
    """One flat parameter a spec, named as the checkpoint names the tensor."""
    for spec in specs:
        t = torch.empty(spec.numel, dtype=DTYPES[spec.dtype])
        holder.register_parameter(
            spec.name, new_param(t, make_loader(spec.slots, f"{prefix}.{spec.name}"), spec.slots))


def register_rotation_params(holder: nn.Module, qc: dict, keys: list[str], prefix: str) -> None:
    """The two tables of every rotation this layer owns, in the file's f64."""
    for key in keys:
        r = qc["rotations"][key]
        shapes = {"signs": (int(r["n"]),), "small": (int(r["odd"]), int(r["odd"]))}
        for table, shape in shapes.items():
            slots = {0: Slot(0, shape)}
            name = rotation_param(key, table)
            holder.register_parameter(name, new_param(
                torch.empty(shape, dtype=torch.float64), make_loader(slots, f"{prefix}.{name}"),
                slots))


def loaded_flats(holder: nn.Module, specs: list[ParamSpec], prefix: str) -> dict[str, torch.Tensor]:
    """The loaded flat tensors of a layer, refused if any shard is missing."""
    out = {}
    for spec in specs:
        p = getattr(holder, spec.name)
        require_loaded(p, f"{prefix}.{spec.name}")
        out[spec.name] = p.data
    return out
