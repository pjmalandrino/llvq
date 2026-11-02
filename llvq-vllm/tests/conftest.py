"""Shared fixtures: the packed directories, vLLM's two naming tables, a loader.

vLLM is not installed on a Mac, so the two tables below are copied from the
pinned sources rather than imported. They are the only facts about vLLM the
tests take on trust, and each names the file and line it was read at.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from llvq_tetra.reader import PackedModel
from llvq_vllm.buffers import rotation_buffers
from llvq_vllm.layer import arm_loaded
from llvq_vllm.params import register_layer_params
from llvq_vllm.plan import INT4, layer_plan, param_specs

HERE = Path(__file__).parent
REPO = HERE.parent.parent
MINI = REPO / "llvq-tetra" / "tests" / "fixtures" / "mini"
Q4B_CONFIG = HERE / "fixtures" / "qwen3-4b-config.json"

# `Qwen3ForCausalLM.packed_modules_mapping`, vLLM 0.26.0, models/qwen3.py:275.
PACKED = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}

# `Qwen2Model.hf_to_vllm_mapper.orig_to_new_stacked`, vLLM 0.26.0,
# models/qwen2.py:323-331, which `Qwen3Model` inherits. The value is the
# renamed module and the shard id vLLM stamps on the tensor.
STACKED = {
    ".q_proj": (".qkv_proj", "q"),
    ".k_proj": (".qkv_proj", "k"),
    ".v_proj": (".qkv_proj", "v"),
    ".gate_proj": (".gate_up_proj", 0),
    ".up_proj": (".gate_up_proj", 1),
}

LAYERS = (
    "model.layers.0.self_attn.qkv_proj",
    "model.layers.0.self_attn.o_proj",
    "model.layers.0.mlp.gate_up_proj",
    "model.layers.0.mlp.down_proj",
)


def quant_config(directory_or_file: Path) -> dict:
    path = directory_or_file / "config.json" if directory_or_file.is_dir() else directory_or_file
    return json.loads(path.read_text())["quantization_config"]


@pytest.fixture(scope="session")
def mini():
    with PackedModel(MINI) as pm:
        yield pm


@pytest.fixture(scope="session")
def mini_qc():
    return quant_config(MINI)


@pytest.fixture(scope="session")
def q4b_qc():
    return quant_config(Q4B_CONFIG)


class Source:
    """Tensors by checkpoint name, from the file or added by a test, and the dense reference."""

    def __init__(self, pm: PackedModel, qc: dict, extra: dict | None = None,
                 dense: dict | None = None):
        self.pm, self.qc = pm, qc
        self.extra = extra or {}
        self.dense_extra = dense or {}

    def tensor(self, name: str) -> np.ndarray:
        return self.extra[name] if name in self.extra else self.pm.tensor(name)

    def dense(self, record: str) -> np.ndarray:
        """`(d_out, d_in)` f32 in the natural basis, through `llvq-tetra`'s f64 decoder."""
        if record in self.dense_extra:
            return self.dense_extra[record]
        return self.pm.dequantize(record)


def mixed_source(pm: PackedModel, qc: dict, seed: int = 0) -> Source:
    """The 4B's qkv layout on the one-layer fixture: q and k Tetra, v int4 g128.

    No fixture has it, and it is the served case. v takes random int4 payloads
    of the fixture's width, 64 by 136, so its last group holds 8 columns.
    """
    from llvq_tetra.dequant import dequantize_int4

    rng = np.random.default_rng(seed)
    d_out, d_in = 64, 136
    gpr = -(-d_in // 128)
    qweight = rng.integers(0, 256, size=(d_out, d_in // 2), dtype=np.uint8)
    scales = (rng.standard_normal((d_out, gpr)) * 0.02).astype(np.float16)
    biases = (rng.standard_normal((d_out, gpr)) * 0.01).astype(np.float16)
    qc = copy.deepcopy(qc)
    prefix = "model.layers.0.self_attn.v_proj"
    qc["records"][f"{prefix}.weight"] = {
        "bits": 4, "d_in": d_in, "d_out": d_out, "group": 128, "groups_per_row": gpr,
        "kind": INT4, "prefix": prefix,
    }
    extra = {f"{prefix}.qweight": qweight, f"{prefix}.scales": scales, f"{prefix}.biases": biases}
    dense = {f"{prefix}.weight": dequantize_int4(qweight, scales, biases, d_out, d_in, 128)}
    return Source(pm, qc, extra, dense)


def vllm_shard_id(prefix: str):
    """The shard id vLLM's mapper stamps on a tensor of this checkpoint module."""
    for part, (_, shard) in STACKED.items():
        if prefix.endswith(part):
            return shard
    return None


def build_layer(src: Source, vllm_prefix: str, tile: int = 64, rows_max: int = 4):
    """One vLLM layer, loaded through its own loaders and armed, on the CPU.

    The tensors go through `weight_loader(param, tensor, shard_id)` exactly as
    vLLM's fused layers call it, so the slots and the shard ids are exercised.
    """
    plan = layer_plan(src.qc, vllm_prefix, PACKED)
    specs = param_specs(plan)
    holder = nn.Module()
    register_layer_params(holder, specs, vllm_prefix)
    for s in plan.shards:
        for spec in specs:
            if s.index in spec.slots:
                p = getattr(holder, spec.name)
                t = torch.from_numpy(np.array(src.tensor(f"{s.prefix}.{spec.name}")))
                p.weight_loader(p, t, vllm_shard_id(s.prefix))
    rotations = {k: rotation_buffers(k, *src.pm.rotation(k), "cpu") for k in plan.rotations}
    armed = arm_loaded(holder, plan, specs, rotations, tile, rows_max, vllm_prefix)
    return holder, armed, plan


def dense_of(src: Source, plan) -> np.ndarray:
    return np.concatenate([src.dense(s.record) for s in plan.shards], axis=0)


def row_error(y: torch.Tensor, ref: np.ndarray) -> float:
    """The worst row's relative error, the per-row check of the CUDA stage."""
    got = y.to(torch.float64).numpy()
    num = np.linalg.norm(got - ref, axis=-1)
    den = np.linalg.norm(ref, axis=-1)
    return float((num / den).max())
