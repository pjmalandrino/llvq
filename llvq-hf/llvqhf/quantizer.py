"""`LlvqQuantizer`, registered with `transformers`, no fork and no patch.

    import llvqhf.quantizer  # registers "llvq"
    model = AutoModelForCausalLM.from_pretrained("<packed directory>")

`register_quantization_config` and `register_quantizer` are the out-of-tree route
(`transformers/quantizers/auto.py`). Nothing here touches the model code: the
quantizer replaces the `nn.Linear` layers it owns before the weights load, the
ordinary loader fills their buffers, and a post-load hook turns the codes into
dense weights.

## Where the rotation went

The file stores the codes in a rotated basis, and the un-rotation is folded into
the dequantization, exactly as `llvq_artifact::decode_matrix` does it. So no
rotation runs in the forward pass and the hook has nothing to host. That is a
decision of the stage 1 prereg §3 and it is what voids the kill criterion the plan
had written for this stage: the hard question, a kernel that reads rotated weights
and rotates the activation, belongs to stages 2 and 4.

## What is dense, and what is not

The **file** is compressed, 1.42 GB for the served 4B. The **loaded model** is
dense: each module materializes its weight and frees its buffers. No memory claim
is made at this stage.
"""

from __future__ import annotations

import os

import torch
from transformers.quantizers import HfQuantizer
from transformers.quantizers.auto import register_quantization_config, register_quantizer
from transformers.utils.quantization_config import QuantizationConfigMixin

from transformers.core_model_loading import WeightConverter

from .conversions import Int4Dequantize
from .modules import Int4Linear, RotationStore, TetraLinear
from .tetra import TetraTables

QUANT_METHOD = "llvq"


@register_quantization_config(QUANT_METHOD)
class LlvqConfig(QuantizationConfigMixin):
    """The `quantization_config` block `hfpack` writes, as an object.

    Kept as the file's own fields rather than a curated subset: the record table is
    the contract between the packer and this reader, and a config that dropped a
    field would make the mismatch surface as a wrong weight instead of an error.
    """

    def __init__(self, **kwargs):
        self.quant_method = QUANT_METHOD
        self.llvq_layout_version = kwargs.get("llvq_layout_version")
        self.artifact_version = kwargs.get("artifact_version")
        self.artifact_sha256 = kwargs.get("artifact_sha256")
        self.artifact_kinds = kwargs.get("artifact_kinds")
        self.codebook_fingerprint = kwargs.get("codebook_fingerprint")
        self.tetra_fingerprint = kwargs.get("tetra_fingerprint")
        self.block_dim = kwargs.get("block_dim")
        self.code_order = kwargs.get("code_order")
        self.records = kwargs.get("records") or {}
        self.rotations = kwargs.get("rotations") or {}
        self.raw = kwargs.get("raw") or {}
        if self.code_order != "msb_first_dense":
            raise ValueError(
                f"code_order {self.code_order!r} is not one this reader unpacks"
            )
        if not self.records:
            raise ValueError("the quantization_config carries no record table")


@register_quantizer(QUANT_METHOD)
class LlvqQuantizer(HfQuantizer):
    """Loads a packed LLVQ directory. Cannot quantize: the encoder is offline."""

    requires_calibration = True
    requires_parameters_quantization = False

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.dtype: torch.dtype | None = None
        self.tables = TetraTables.load()
        # Before any index is read. A Tetra word against the wrong map is in
        # range, decodes to a lattice point, and is wrong.
        self.tables.require_fingerprint(quantization_config.tetra_fingerprint)
        if int(quantization_config.block_dim) != self.tables.dim:
            raise ValueError(
                f"block_dim {quantization_config.block_dim} against a map of {self.tables.dim}"
            )

    def validate_environment(self, *args, **kwargs):
        return

    def update_dtype(self, dtype):
        """f32 unless asked otherwise, and remembered for the two decode paths.

        The weights are rebuilt in f64 and narrowed once, which is what the
        artifact's own decoder does. Narrowing to f16 as well is the caller's
        choice and is not made here. `from_pretrained` calls this before it
        loads, which is where a quantizer learns the dtype that was asked for.
        """
        self.dtype = torch.float32 if dtype is None else dtype
        return self.dtype

    def _process_model_before_weight_loading(self, model, **kwargs):
        """Swap the modules so the file's keys are the model's keys."""
        qc = self.quantization_config
        by_prefix = {d["prefix"]: d for d in qc.records.values()}
        replaced = 0
        for name, module in list(model.named_modules()):
            desc = by_prefix.get(name)
            if desc is None:
                continue
            if not isinstance(module, torch.nn.Linear):
                raise TypeError(f"{name} is a {type(module).__name__}, not an nn.Linear")
            if module.in_features != desc["d_in"] or module.out_features != desc["d_out"]:
                raise ValueError(
                    f"{name} is {module.out_features} by {module.in_features}, "
                    f"the record is {desc['d_out']} by {desc['d_in']}"
                )
            cls = TetraLinear if desc["kind"] == "tetra" else Int4Linear
            _set_module(model, name, cls(desc, module.bias))
            replaced += 1
        if replaced != len(qc.records):
            raise ValueError(f"{replaced} modules replaced for {len(qc.records)} records")

        if qc.rotations:
            model.llvq = RotationStore(qc.rotations)

    def get_weight_conversions(self):
        """The carried tensors the file stores group-affine, the embedding among them.

        Three checkpoint keys to one parameter, during the load, through
        `transformers`' own pipeline. An f16 carried tensor needs nothing: its key
        and its dtype are already the model's.
        """
        out = []
        for tensor_name, desc in self.quantization_config.raw.items():
            if desc["encoding"] != "quant":
                continue
            prefix = desc["prefix"]
            out.append(
                WeightConverter(
                    [f"{prefix}.qweight", f"{prefix}.scales", f"{prefix}.biases"],
                    tensor_name,
                    operations=[Int4Dequantize(desc, self.dtype)],
                )
            )
        return out

    def _process_model_after_weight_loading(self, model, **kwargs):
        """Either materialize the weights, or arm the kernel on them.

        `LLVQ_HF_FUSED=1` keeps the Tetra records compressed on the device and puts
        the served matvec in the forward pass (M1 of `docs/plan-transformers.md`).
        Unset, every record is dequantized into a dense weight, which is what stage
        1 measured. An unknown value is refused rather than treated as off: a typo
        would silently publish the wrong arm.
        """
        want = os.environ.get("LLVQ_HF_FUSED", "0")
        if want not in ("0", "1"):
            raise ValueError(f"LLVQ_HF_FUSED={want!r} is neither 0 nor 1")
        fused = want == "1"
        rotations = model.llvq.tables() if hasattr(model, "llvq") else {}
        dtype = self.dtype or self.update_dtype(None)
        if fused:
            self._arm_fused(model, rotations)
        else:
            for module in model.modules():
                if isinstance(module, (TetraLinear, Int4Linear)):
                    module.materialize(self.tables, rotations, dtype)
        if hasattr(model, "llvq"):
            del model.llvq
        return model

    def _arm_fused(self, model, rotations: dict):
        """Swap each record for its resident form, on the device it is already on.

        The int4 records go through `tv_q4_metal_tiled`, added beside the served
        `tv_q4_metal` rather than replacing it, and bit-identical to it wherever both
        run. `LLVQ_HF_Q4_TILE` sets the slice, 2,048 columns by default, and a value
        that is not a multiple of 256 is refused by the binding rather than rounded.

        The device is read from the module's own buffers and is not assumed. It used
        to be the string "mps", which made every non-Apple caller fail inside `.to()`
        on a device that does not exist there.

        Where a kind has no kernel for that device it is materialized dense instead,
        and the count is kept so the memory report can say what it measured. Today
        that is the int4 records on a card: `csrc/tetra_cuda.cu` binds `tv_tetra48`
        and nothing else. A silent fallback would let a run be published as fused
        when a third of its projections are dense.
        """
        from .fused import FusedTetraLinear, Int4FusedLinear, Rotation

        tile = int(os.environ.get("LLVQ_HF_TILE", "64"))
        q4_tile = int(os.environ.get("LLVQ_HF_Q4_TILE", "2048"))
        built: dict[tuple[str, str], Rotation] = {}
        dtype = self.dtype or self.update_dtype(None)
        self.armed = {"fused": 0, "dense_for_want_of_a_kernel": 0}
        for name, module in list(model.named_modules()):
            if isinstance(module, Int4Linear):
                device = _module_device(module, name)
                if Int4FusedLinear.supports(device):
                    _set_module(model, name,
                                Int4FusedLinear.from_loaded(module, q4_tile, device))
                    self.armed["fused"] += 1
                else:
                    module.materialize(self.tables, rotations, dtype)
                    self.armed["dense_for_want_of_a_kernel"] += 1
                continue
            if not isinstance(module, TetraLinear):
                continue
            device = _module_device(module, name)
            key = module.desc.get("rotation")
            if key and (key, device) not in built:
                signs, small = rotations[key]
                built[(key, device)] = Rotation(signs, small, device)
            _set_module(
                model, name,
                FusedTetraLinear.from_loaded(module, built.get((key, device)), tile, device),
            )
            self.armed["fused"] += 1

    def resident_bytes(self, model) -> int:
        """Device bytes the armed projections hold, measured and not computed."""
        from .fused import FusedTetraLinear

        return sum(m.resident_bytes() for m in model.modules()
                   if isinstance(m, FusedTetraLinear))

    def _process_model_after_loading(self, model, **kwargs):
        return model

    @property
    def is_trainable(self) -> bool:
        """The codes are frozen and there is no gradient through the lattice."""
        return False

    def is_serializable(self, safe_serialization=None) -> bool:
        """`save_pretrained` would write the dense weights, not our format."""
        return False


def _module_device(module, name: str) -> str:
    """The device a loaded record is already on, as a type string.

    `_arm_fused` used to hard-code "mps". Reading it from the buffers keeps the
    kernel on whatever device `transformers` put the weights, and a record with no
    buffer at all is a bug worth naming rather than defaulting.
    """
    for t in list(module.buffers()) + list(module.parameters()):
        return t.device.type
    raise RuntimeError(f"{name} carries no buffer, so its device cannot be read")


def _set_module(model, name: str, new: torch.nn.Module) -> None:
    parent, _, leaf = name.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, leaf, new)
