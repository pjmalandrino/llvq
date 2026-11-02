"""The file's tensor stream, renamed as vLLM 0.26.0 renames it, through our loaders.

vLLM loads a Qwen3 in two levels. `Qwen3ForCausalLM.load_weights` applies the
quantization config's mapper, which is where the rotation tables move onto
their owners. `Qwen2Model.load_weights` then applies the stacked renames, which
send `q_proj` to `qkv_proj` with shard id "q" and so on. A fused layer calls
`weight_loader(param, tensor, shard_id)`, an unfused one is reached by
`AutoWeightsLoader` with two arguments. This replays that, so a name, a slot or
a shard id that does not line up fails here and not in a billed job.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from conftest import LAYERS, PACKED, STACKED, Source, dense_of, mixed_source, row_error
from llvq_vllm.buffers import rotation_buffers
from llvq_vllm.layer import arm_loaded, forward
from llvq_vllm.params import register_layer_params, register_rotation_params
from llvq_vllm.plan import layer_plan, owned_keys, param_specs, rotation_param, rotation_renames


def vllm_name(name: str, renames: dict) -> tuple[str | None, object]:
    """`(name, shard_id)` after both levels of vLLM's renames, `None` when dropped."""
    for src, dst in renames.items():  # level 1, `orig_to_new_prefix`
        if name.startswith(src):
            if dst is None:
                return None, None
            name = name.replace(src, dst, 1)
    head, _, rest = name.partition(".")
    if head != "model":
        return name, None
    shard = None
    for substr, (new, new_shard) in STACKED.items():  # level 2, no break, as vLLM loops
        if substr in rest:
            rest = rest.replace(substr, new, 1)
            shard = new_shard
    return f"model.{rest}", shard


def stream(src: Source, drop: tuple[str, ...] = ()):
    """Every tensor of the record table and of the rotation table, in file order."""
    names = set()
    for d in src.qc["records"].values():
        suffixes = (("codes", "row_scales", "centroids") + (("tail",) if d.get("tail_cols") else ())
                    if d["kind"] == "tetra" else ("qweight", "scales", "biases"))
        names.update(f"{d['prefix']}.{s}" for s in suffixes)
    for key in src.qc["rotations"]:
        names.update(f"llvq.rotations.{key}.{t}" for t in ("signs", "small"))
    for name in sorted(names - set(drop)):
        yield name, torch.from_numpy(np.array(src.tensor(name)))


def load_model(src: Source, drop: tuple[str, ...] = ()) -> dict[str, nn.Module]:
    """The four linear layers of the fixture, created, then loaded through the stream."""
    holders, plans = {}, {}
    for prefix in LAYERS:
        plan = layer_plan(src.qc, prefix, PACKED)
        holder = nn.Module()
        register_layer_params(holder, param_specs(plan), prefix)
        register_rotation_params(holder, src.qc, owned_keys(src.qc, PACKED, prefix), prefix)
        holders[prefix], plans[prefix] = holder, plan
    renames = rotation_renames(src.qc, PACKED)
    for name, tensor in stream(src, drop):
        mapped, shard = vllm_name(name, renames)
        layer, _, attr = mapped.rpartition(".")
        param = getattr(holders[layer], attr)
        if shard is None:
            param.weight_loader(param, tensor)
        else:
            param.weight_loader(param, tensor, shard)
    return holders, plans


def arm_model(src: Source, holders: dict, plans: dict) -> dict:
    owners = {}
    for prefix in LAYERS:
        for key in owned_keys(src.qc, PACKED, prefix):
            owners[key] = holders[prefix]
    armed = {}
    for prefix in LAYERS:
        plan = plans[prefix]
        rotations = {}
        for key in plan.rotations:
            owner = owners[key]
            tables = [getattr(owner, rotation_param(key, t)).detach().numpy()
                      for t in ("signs", "small")]
            rotations[key] = rotation_buffers(key, *tables, "cpu")
        armed[prefix] = arm_loaded(holders[prefix], plan, param_specs(plan), rotations, 64, 4,
                                   prefix)
    return armed


@pytest.mark.parametrize("layout", ["fixture", "served qkv"])
def test_the_renamed_stream_serves_every_layer(mini, mini_qc, layout):
    src = Source(mini, mini_qc) if layout == "fixture" else mixed_source(mini, mini_qc)
    holders, plans = load_model(src)
    armed = arm_model(src, holders, plans)
    for prefix in LAYERS:
        plan = plans[prefix]
        x = torch.randn(3, plan.d_in, generator=torch.Generator().manual_seed(7)).half()
        y = forward(holders[prefix], armed[prefix], x)
        ref = x.double().numpy() @ dense_of(src, plan).astype(np.float64).T
        assert row_error(y, ref) < 1e-2, prefix


def test_the_rotation_tables_land_on_their_owners_unchanged(mini, mini_qc):
    src = Source(mini, mini_qc)
    holders, _ = load_model(src)
    owner = holders["model.layers.0.mlp.gate_up_proj"]
    signs, small = mini.rotation("136_0000000000005eed")
    assert np.array_equal(owner.llvq_rot_136_0000000000005eed_signs.detach().numpy(), signs)
    assert np.array_equal(owner.llvq_rot_136_0000000000005eed_small.detach().numpy(), small)
    assert not hasattr(holders["model.layers.0.self_attn.qkv_proj"],
                       "llvq_rot_136_0000000000005eed_signs")


def test_a_tensor_missing_from_the_stream_is_refused_by_name(mini, mini_qc):
    src = Source(mini, mini_qc)
    holders, plans = load_model(src, drop=("model.layers.0.self_attn.k_proj.row_scales",))
    with pytest.raises(RuntimeError, match=r"qkv_proj\.row_scales: shards \[1\] were never loaded"):
        arm_model(src, holders, plans)


def test_a_tensor_loaded_twice_is_refused(mini, mini_qc):
    src = Source(mini, mini_qc)
    prefix = "model.layers.0.mlp.down_proj"
    plan = layer_plan(src.qc, prefix, PACKED)
    holder = nn.Module()
    register_layer_params(holder, param_specs(plan), prefix)
    t = torch.from_numpy(np.array(mini.tensor(f"{prefix}.scales")))
    holder.scales.weight_loader(holder.scales, t)
    with pytest.raises(ValueError, match="loaded twice"):
        holder.scales.weight_loader(holder.scales, t)


def test_a_tensor_of_the_wrong_shape_is_refused(mini, mini_qc):
    src = Source(mini, mini_qc)
    prefix = "model.layers.0.self_attn.qkv_proj"
    plan = layer_plan(src.qc, prefix, PACKED)
    holder = nn.Module()
    register_layer_params(holder, param_specs(plan), prefix)
    k_tail = torch.from_numpy(np.array(mini.tensor("model.layers.0.self_attn.k_proj.tail")))
    with pytest.raises(ValueError, match=r"checkpoint shape \(64, 16\), want \(128, 16\)"):
        holder.tail.weight_loader(holder.tail, k_tail, "q")


def test_a_fused_layer_refuses_a_tensor_without_a_shard_id(mini, mini_qc):
    src = Source(mini, mini_qc)
    prefix = "model.layers.0.mlp.gate_up_proj"
    plan = layer_plan(src.qc, prefix, PACKED)
    holder = nn.Module()
    register_layer_params(holder, param_specs(plan), prefix)
    t = torch.from_numpy(np.array(mini.tensor("model.layers.0.mlp.up_proj.centroids")))
    with pytest.raises(ValueError, match="without a shard id"):
        holder.centroids.weight_loader(holder.centroids, t)
