"""`LlvqQuantizer`, registered with `transformers`, no fork and no patch.

    import llvq_tetra  # registers "llvq"
    model = AutoModelForCausalLM.from_pretrained("<packed directory>", device_map="mps")

`register_quantization_config` and `register_quantizer` are the out-of-tree route
(`transformers/quantizers/auto.py`). Nothing here touches the model code. Every
hook delegates to `integration.py`, the same four functions an in-tree
`transformers` integration calls, so the two routes cannot drift.

## What is compressed, and where

The `device_map` decides. On `"mps"` and `"cuda"` the layers stay compressed in
memory and the kernels multiply them as stored; on Metal the vocabulary stays
4-bit too. Without a `device_map` the model loads on the CPU and every layer is
rebuilt dense, in f64 then narrowed once, as `llvq_artifact::decode_matrix` does.
`LLVQ_HF_FUSED=1`, the switch of 0.1.0, still asks for the compressed path when no
`device_map` is given.
"""

from __future__ import annotations

import torch
from transformers.quantizers import HfQuantizer
from transformers.quantizers.auto import (
    AUTO_QUANTIZATION_CONFIG_MAPPING,
    AUTO_QUANTIZER_MAPPING,
    register_quantization_config,
    register_quantizer,
)
from transformers.utils.quantization_config import QuantizationConfigMixin

# Imported for what it asserts: the conversion pipeline `integration.weight_conversions`
# builds on, which transformers before 5.x does not have. Too old, the import fails
# here, and `__init__` raises rather than registering nothing.
from transformers.core_model_loading import WeightConverter  # noqa: F401

from . import integration
from .tetra import TetraTables

QUANT_METHOD = "llvq"


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


class LlvqQuantizer(HfQuantizer):
    """Loads a packed LLVQ directory. Cannot quantize: the encoder is offline."""

    requires_calibration = True
    requires_parameters_quantization = False

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.dtype: torch.dtype | None = None
        self.target_device: str | None = None
        self.tables = TetraTables.load()
        # Before any index is read. A Tetra word against the wrong map is in
        # range, decodes to a lattice point, and is wrong.
        self.tables.require_fingerprint(quantization_config.tetra_fingerprint)
        if int(quantization_config.block_dim) != self.tables.dim:
            raise ValueError(
                f"block_dim {quantization_config.block_dim} against a map of {self.tables.dim}"
            )

    def validate_environment(self, *args, device_map=None, **kwargs):
        """CUDA builds its kernels with `cpp_extension`, which needs ninja; Metal needs nothing."""
        if integration.target_device(device_map) == "cuda":
            from torch.utils.cpp_extension import is_ninja_available

            if not is_ninja_available():
                raise ImportError(
                    "Keeping an LLVQ model compressed on cuda compiles its kernels on first use, "
                    "which needs ninja: `pip install ninja`"
                )

    def update_device_map(self, device_map):
        """The checkpoint loads on the CPU, and `finalize` moves the model to its device after.

        The codes are transcoded to the kernels' layout on the CPU, and loading straight onto an
        Apple GPU would also fail on the float64 scales the checkpoint carries.
        """
        self.target_device = integration.target_device(device_map)
        if device_map is None or self.target_device == "cpu":
            return device_map
        return {"": "cpu"}

    def update_dtype(self, dtype):
        """f32 unless asked otherwise, and remembered for the two decode paths.

        The weights are rebuilt in f64 and narrowed once, which is what the
        artifact's own decoder does. Narrowing to f16 as well is the caller's
        choice and is not made here. `from_pretrained` calls this before it
        loads, which is where a quantizer learns the dtype that was asked for.
        """
        self.dtype = torch.float32 if dtype is None else dtype
        return self.dtype

    def _device(self) -> str:
        # `update_device_map` sets it; a caller that skips it gets the 0.1.0 behaviour.
        if getattr(self, "target_device", None) is None:
            self.target_device = integration.target_device(None)
        return self.target_device

    def _process_model_before_weight_loading(self, model, **kwargs):
        """Swap the modules so the file's keys are the model's keys."""
        integration.replace_modules(model, self.quantization_config, self._device())

    def get_weight_conversions(self):
        """The carried tensors the file stores group-affine, the embedding among them.

        Three checkpoint keys to one parameter, during the load, through
        `transformers`' own pipeline, unless they stay 4-bit on Metal.
        """
        return integration.weight_conversions(
            self.quantization_config, self.dtype or self.update_dtype(None), self._device()
        )

    def _process_model_after_weight_loading(self, model, **kwargs):
        """Compress the layers on a GPU and move the model there, or rebuild them dense."""
        return integration.finalize(model, self.tables, self.dtype or self.update_dtype(None), self._device())

    def resident_bytes(self, model) -> int:
        """Device bytes the compressed layers and the 4-bit vocabulary hold, measured."""
        return sum(m.resident_bytes() for m in model.modules() if hasattr(m, "resident_bytes"))

    @property
    def is_trainable(self) -> bool:
        """The codes are frozen and there is no gradient through the lattice."""
        return False

    def is_serializable(self, safe_serialization=None) -> bool:
        """`save_pretrained` would write the dense weights, not our format."""
        return False


# Registered only under a free name. `register_quantization_config` and
# `register_quantizer` raise on a name `transformers` already holds
# (`quantizers/auto.py`), and the day this method is in tree it holds "llvq". A
# version on PyPI cannot be changed afterwards, so an unconditional registration
# would make `import llvq_tetra` fail on every later `transformers` that carries
# the method. Leaving the in-tree classes in place is the right outcome: they
# are the ones that version of `transformers` was tested with.
if QUANT_METHOD not in AUTO_QUANTIZATION_CONFIG_MAPPING:
    register_quantization_config(QUANT_METHOD)(LlvqConfig)
if QUANT_METHOD not in AUTO_QUANTIZER_MAPPING:
    register_quantizer(QUANT_METHOD)(LlvqQuantizer)
