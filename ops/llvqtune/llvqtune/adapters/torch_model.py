"""The student: a dequantized checkpoint whose linears route through a Trainable.

## Where the weights come from

`llvq-llm --bin export` turns a sealed `.llvq` into an f16 Hugging Face
checkpoint. Its own note says why that substitution is legitimate: the
artifact decodes bit for bit to those weights. So the exported tensor of a
lattice matrix is exactly `centroids[g] * row_scale * u` on the coded columns,
and the verbatim tail after them.

That is what lets Python train scales without reading the `.llvq` format. A
per-row multiplier on the exported tensor's coded columns is a per-row
multiplier on `row_scales`, and the write-back is one multiply in Rust.

## Which matrices are routed, and which are not

`v_proj` is served as int4 g128 in `configs/qwen3-4b-tetra-q5.json`. An int4
record holds no `row_scales`, so there is nothing for a row multiplier to fold
into. It is excluded by default, the same exclusion `rhoapply` makes by
construction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

BLOCK = 24
LATTICE_TYPES = ("q_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
# The norms whose weight is a per-column scale on a matrix's input, applied
# before the rotation. `row_norms.py` says which matrix each one feeds.
NORM_TYPES = ("input_layernorm", "post_attention_layernorm")


class RoutedLinear(nn.Module):
    """A linear whose weight is rebuilt from frozen directions at every call."""

    def __init__(
        self, name: str, source: nn.Linear, trainable, commute: bool = False
    ) -> None:
        super().__init__()
        self.name = name
        self._commute = commute
        frozen = source.weight.detach().clone()
        frozen.requires_grad_(False)
        self.register_buffer("frozen", frozen, persistent=False)
        self.bias = source.bias
        self._trainable = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # A mode may offer `linear`, which reaches the output without
        # building the weight and keeps 7.05 GB of rebuilt weights out of the
        # autograd graph. It is OFF by default because it was *measured*
        # slower on MPS: 32.53 s a step against 26.21 s, the tail correction
        # costing 216 extra small matmuls whose launch overhead exceeds what
        # the rebuild costs (2026-09-19, Qwen3-4B, seq 1024 batch 2).
        #
        # It is kept, and proven equal to the rebuilt form to 1e-12 in f64,
        # because that trade turns on the device and has never been read on a
        # card.
        if self._commute:
            fast = getattr(self._trainable, "linear", None)
            if fast is not None:
                return fast(self.name, x, self.frozen, self.bias)
        weight = self._trainable.weight(self.name, self.frozen)
        return F.linear(x, weight, self.bias)


class RoutedNorm(nn.Module):
    """A frozen RMSNorm whose output a Trainable rescales per column."""

    def __init__(self, name: str, source: nn.Module, trainable) -> None:
        super().__init__()
        self.name = name
        for parameter in source.parameters():
            parameter.requires_grad_(False)
        self.source = source
        self._trainable = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._trainable.scale_norm(self.name, self.source(x))


class TorchModel:
    """Wraps a `transformers` causal LM and routes its quantized linears."""

    def __init__(
        self,
        model,
        trainable,
        types: tuple[str, ...] = LATTICE_TYPES,
        layers: range | None = None,
        commute: bool = False,
    ) -> None:
        self._model = model
        self._types = types
        self._commute = commute
        self._routed: dict[str, RoutedLinear] = {}
        self._shapes: dict[str, tuple[int, int]] = {}
        self._param_count = sum(p.numel() for p in model.parameters())
        self._install(trainable, layers)
        self._norms: dict[str, RoutedNorm] = {}
        if hasattr(trainable, "scale_norm"):
            self._install_norms(trainable, layers)
        if not self._routed:
            raise ValueError(f"no linear matched types {types}")

    def _install(self, trainable, layers: range | None) -> None:
        for index, block in enumerate(self._model.model.layers):
            if layers is not None and index not in layers:
                continue
            for parent_name, parent in (
                ("self_attn", block.self_attn),
                ("mlp", block.mlp),
            ):
                for kind in self._types:
                    child = getattr(parent, kind, None)
                    if not isinstance(child, nn.Linear):
                        continue
                    name = f"model.layers.{index}.{parent_name}.{kind}"
                    routed = RoutedLinear(
                        name, child, trainable, commute=self._commute
                    )
                    setattr(parent, kind, routed)
                    self._routed[name] = routed
                    rows, cols = child.weight.shape
                    self._shapes[name] = (rows, cols)

    def _install_norms(self, trainable, layers: range | None) -> None:
        wanted = set(trainable.norms)
        for name, width in self.norm_widths_for(self._model, layers).items():
            if name not in wanted:
                continue
            parent_name, _, child = name.rpartition(".")
            parent = self._model.get_submodule(parent_name)
            routed = RoutedNorm(name, getattr(parent, child), trainable)
            setattr(parent, child, routed)
            self._norms[name] = routed
        missing = wanted - set(self._norms)
        if missing:
            raise ValueError(f"norms not found in the model: {sorted(missing)}")

    @staticmethod
    def norm_widths_for(model, layers: range | None = None) -> dict[str, int]:
        """Every routable norm and its width, the argument `RowNorms` takes.

        `model.norm` is included: it scales the columns of the language head,
        which is the tied embedding on the 4B, and folds the same way.
        """
        found: dict[str, int] = {}
        for index, block in enumerate(model.model.layers):
            if layers is not None and index not in layers:
                continue
            for kind in NORM_TYPES:
                norm = getattr(block, kind)
                found[f"model.layers.{index}.{kind}"] = norm.weight.shape[0]
        found["model.norm"] = model.model.norm.weight.shape[0]
        return found

    @property
    def norms(self) -> list[str]:
        return list(self._norms)

    @staticmethod
    def shapes_for(model, types=LATTICE_TYPES, layers: range | None = None):
        """Row and lattice-column counts, the argument `RowScales` takes.

        Read before the model is routed, so a mode can be built and priced
        before a single weight is touched.
        """
        found: dict[str, tuple[int, int]] = {}
        for index, block in enumerate(model.model.layers):
            if layers is not None and index not in layers:
                continue
            for parent_name, parent in (
                ("self_attn", block.self_attn),
                ("mlp", block.mlp),
            ):
                for kind in types:
                    child = getattr(parent, kind, None)
                    if not isinstance(child, nn.Linear):
                        continue
                    rows, cols = child.weight.shape
                    found[f"model.layers.{index}.{parent_name}.{kind}"] = (
                        rows,
                        cols - cols % BLOCK,
                    )
        return found

    @property
    def matrices(self) -> list[str]:
        return list(self._routed)

    @property
    def param_count(self) -> int:
        return self._param_count

    def rows_of(self, matrix: str) -> int:
        return self._shapes[matrix][0]

    def lattice_columns_of(self, matrix: str) -> int:
        cols = self._shapes[matrix][1]
        return cols - cols % BLOCK

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self._model(input_ids=ids).logits


class TorchTeacher:
    """The dense reference. Frozen, evaluated without a graph."""

    def __init__(self, model) -> None:
        self._model = model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def logits(self, ids: torch.Tensor) -> torch.Tensor:
        return self._model(input_ids=ids).logits
