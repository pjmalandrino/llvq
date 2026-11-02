"""The adapter to vLLM 0.26.0: a quantization config and two quantize methods.

Written against the pinned sources (`vllm/vllm-openai:v0.26.0`, commit
568afb3a), read on 2026-11-02. The API this file leans on, and where:

* `register_quantization_config("llvq")` appends the name to the methods vLLM
  accepts (`layers/quantization/__init__.py`); the checkpoint's
  `quantization_config.quant_method` then selects it (`config/model.py`).
* `QuantizationConfig.get_quant_method(layer, prefix)` is asked once per linear
  layer, once per vocabulary layer, and twice per attention layer, which gets
  `None` and so keeps an f16 KV cache.
* `LinearMethodBase.create_weights` registers the layer's parameters before the
  load, `process_weights_after_loading` runs once all weights are in, and
  `apply(layer, x, bias)` is the forward.
* `get_cache_scale_mapper()` returns a `WeightsMapper` that `AutoWeightsLoader`
  applies to the whole weight stream (`models/utils.py`). It is how the
  rotation tables reach their owner layers, see `plan.py`.

Nothing here has run inside vLLM: the image is Linux and CUDA only. What the
Mac holds is everything this file delegates to, in `params.py` and `layer.py`.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn

from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.utils import WeightsMapper

from . import layer as L
from . import ops
from .buffers import RotationBuffers, rotation_buffers
from .params import register_layer_params, register_rotation_params, require_loaded
from .plan import (
    QUANT_METHOD,
    embedding_desc,
    layer_plan,
    owned_keys,
    param_specs,
    rotation_param,
    rotation_renames,
    validate_header,
)


def env_int(name: str, default: int, lo: int, hi: int) -> int:
    """An integer knob, refused by name when it is not one, never rounded."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    if not raw.isdigit() or not lo <= int(raw) <= hi:
        raise ValueError(f"{name}={raw!r} is not an integer in [{lo}, {hi}]")
    return int(raw)


class LlvqVllmConfig(QuantizationConfig):
    """The packed file's `quantization_config`, read once, shared by every layer."""

    def __init__(self, qc: dict[str, Any]):
        super().__init__()
        validate_header(qc)
        from llvq_tetra.tetra import TetraTables

        # Before any index is read: a Tetra word against the wrong map is in
        # range, decodes to a lattice point, and is wrong.
        TetraTables.load().require_fingerprint(qc["tetra_fingerprint"])
        self.qc = qc
        # The tile is a compile-time define of the Tetra kernel, so it is read
        # from the variable `llvq_tetra.metal` reads when it builds the extension.
        self.tile = env_int("LLVQ_HF_TILE", 64, 1, 1024)
        self.rows_max = env_int("LLVQ_VLLM_ROWS_MAX", 16, 1, 512)
        self._owner_layers: dict[str, nn.Module] = {}
        self._rotations: dict[tuple[str, str], RotationBuffers] = {}

    def get_name(self) -> str:
        return QUANT_METHOD

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # Every kernel reads and writes f16; bf16 would need a second build.
        return [torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Run on sm_80 (A100) and sm_89 (L4, L40S); `__dp4a` alone would allow
        # sm_61, which nothing here has run on.
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LlvqVllmConfig":
        return cls(config)

    def get_quant_method(self, layer: nn.Module, prefix: str) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            return LlvqLinearMethod(self, prefix)
        if isinstance(layer, VocabParallelEmbedding):
            desc = embedding_desc(self.qc, prefix)
            return None if desc is None else LlvqEmbeddingMethod(self, prefix, desc)
        return None

    def get_cache_scale_mapper(self) -> WeightsMapper:
        """vLLM's KV-scale renames, plus the rotation tables onto their owners."""
        renames = rotation_renames(self.qc, self.packed_modules_mapping)
        return QuantizationConfig.get_cache_scale_mapper() | WeightsMapper(
            orig_to_new_prefix=renames)

    def register_owner(self, key: str, layer: nn.Module) -> None:
        self._owner_layers[key] = layer

    def rotation(self, key: str, device: torch.device) -> RotationBuffers:
        """One rotation's kernel buffers on `device`, built once and shared."""
        cache = (key, str(device))
        if cache not in self._rotations:
            owner = self._owner_layers.get(key)
            if owner is None:
                raise KeyError(f"rotation {key} has no owner layer; the plan and the model "
                               "disagree")
            tables = []
            for table in ("signs", "small"):
                p = getattr(owner, rotation_param(key, table))
                require_loaded(p, f"rotation {key}.{table}")
                tables.append(p.detach().cpu().numpy())
            self._rotations[cache] = rotation_buffers(key, *tables, device)
        return self._rotations[cache]


class LlvqLinearMethod(LinearMethodBase):
    """A vLLM linear layer served from one or more records of the file."""

    def __init__(self, config: LlvqVllmConfig, prefix: str):
        self.config = config
        self.prefix = prefix

    def create_weights(self, layer: nn.Module, input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int, output_size: int,
                       params_dtype: torch.dtype, **extra_weight_attrs) -> None:
        if getattr(layer, "tp_size", 1) != 1:
            raise NotImplementedError("tensor parallelism is not supported: one card only")
        cfg = self.config
        plan = layer_plan(cfg.qc, self.prefix, cfg.packed_modules_mapping)
        widths = [s.d_out for s in plan.shards]
        if list(output_partition_sizes) != widths or input_size_per_partition != plan.d_in:
            raise ValueError(
                f"{self.prefix}: vLLM builds {output_partition_sizes} by "
                f"{input_size_per_partition}, the file holds {widths} by {plan.d_in}")
        specs = param_specs(plan)
        register_layer_params(layer, specs, self.prefix)
        keys = owned_keys(cfg.qc, cfg.packed_modules_mapping, self.prefix)
        register_rotation_params(layer, cfg.qc, keys, self.prefix)
        for key in keys:
            cfg.register_owner(key, layer)
        layer.llvq_plan = plan
        layer.llvq_specs = specs

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        plan, specs = layer.llvq_plan, layer.llvq_specs
        device = getattr(layer, specs[0].name).device
        ops.prepare(device)
        rotations = {k: self.config.rotation(k, device) for k in plan.rotations}
        layer.llvq_armed = L.arm_loaded(layer, plan, specs, rotations, self.config.tile,
                                        self.config.rows_max, self.prefix)

    def apply(self, layer: nn.Module, x: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor:
        return L.forward(layer, layer.llvq_armed, x, bias)


class LlvqEmbeddingMethod(QuantizeMethodBase):
    """The int4 g64 vocabulary: the embedding gather and, tied, the output layer.

    Qwen3-4B ties its head to its embedding by making `lm_head` the same module
    (`models/qwen3.py`), so `apply` on this method is the logits and `embedding`
    is the input rows, both from one table that is never widened.
    """

    def __init__(self, config: LlvqVllmConfig, prefix: str, desc: dict):
        self.config = config
        self.prefix = prefix
        self.desc = desc

    def create_weights(self, layer: nn.Module, input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int, output_size: int,
                       params_dtype: torch.dtype, **extra_weight_attrs) -> None:
        if getattr(layer, "tp_size", 1) != 1:
            raise NotImplementedError("tensor parallelism is not supported: one card only")
        d, gpr = self.desc["dim"], self.desc["gpr"]
        rows = output_partition_sizes[0]
        if input_size_per_partition != d or rows < self.desc["rows"]:
            raise ValueError(f"{self.prefix}: vLLM builds {rows} by {input_size_per_partition}, "
                             f"the file holds {self.desc['rows']} by {d}")
        vocab_loader = extra_weight_attrs["weight_loader"]

        def loader(param, loaded, *args):
            # VocabParallelEmbedding's own loader narrows the rows to this rank
            # and zeroes the vocabulary padding; this only records that it ran.
            vocab_loader(param, loaded, *args)
            param.llvq_loaded[0] = True

        shapes = {"qweight": ((rows, d // 2), torch.uint8),
                  "scales": ((rows, gpr), torch.float16),
                  "biases": ((rows, gpr), torch.float16)}
        for name, (shape, dtype) in shapes.items():
            p = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
            p.llvq_loaded, p.llvq_expected = {}, (0,)
            p.output_dim = 0
            p.weight_loader = loader
            layer.register_parameter(name, p)
        layer.llvq_vocab = (rows, d, gpr)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        for name in ("qweight", "scales", "biases"):
            require_loaded(getattr(layer, name), f"{self.prefix}.{name}")
        ops.prepare(layer.qweight.device)

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        rows, d, gpr = layer.llvq_vocab
        return torch.ops.llvq_vllm.emb_q4_gather(input_, layer.qweight, layer.scales,
                                                 layer.biases, d, gpr)

    def apply(self, layer: nn.Module, x: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor:
        rows, d, gpr = layer.llvq_vocab
        y = torch.ops.llvq_vllm.emb_q4_mm(x.reshape(-1, d), layer.qweight, layer.scales,
                                          layer.biases, rows, d, gpr, self.config.rows_max)
        if bias is not None:
            y = y + bias.to(y.dtype)
        return y.reshape(*x.shape[:-1], rows)

    def tie_weights(self, layer: nn.Module, embed_tokens: nn.Module) -> nn.Module:
        """Share the table with a separate head module, for a model that has one."""
        for name in ("qweight", "scales", "biases"):
            setattr(layer, name, getattr(embed_tokens, name))
        layer.llvq_vocab = embed_tokens.llvq_vocab
        return layer
