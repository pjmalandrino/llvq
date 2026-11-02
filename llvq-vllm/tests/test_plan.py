"""The record plan, on the published 4B's own record table and on the fixture."""

from __future__ import annotations

import copy

import pytest

from conftest import PACKED
from llvq_vllm import plan as P

L0 = "model.layers.0"


def test_both_headers_pass(q4b_qc, mini_qc):
    P.validate_header(q4b_qc)
    P.validate_header(mini_qc)


def test_the_4b_qkv_is_two_tetra_shards_on_one_rotation_and_one_int4(q4b_qc):
    plan = P.layer_plan(q4b_qc, f"{L0}.self_attn.qkv_proj", PACKED)
    assert [s.kind for s in plan.shards] == [P.TETRA, P.TETRA, P.INT4]
    assert [s.d_out for s in plan.shards] == [4096, 1024, 1024]
    assert plan.d_in == 2560
    # q and k read one rotated activation, v the natural one.
    assert plan.rotations == ("2560_000000000110feed",)
    assert plan.shards[2].rotation is None


def test_the_4b_gate_up_shares_one_rotation(q4b_qc):
    plan = P.layer_plan(q4b_qc, f"{L0}.mlp.gate_up_proj", PACKED)
    assert [s.kind for s in plan.shards] == [P.TETRA, P.TETRA]
    assert plan.rotations == ("2560_000000000112feed",)
    assert plan.d_out == 2 * 9728


def test_the_4b_down_proj_is_tetra_or_int4_by_layer(q4b_qc):
    for layer in range(36):
        plan = P.layer_plan(q4b_qc, f"model.layers.{layer}.mlp.down_proj", PACKED)
        (s,) = plan.shards
        if 12 <= layer <= 23:
            assert s.kind == P.INT4 and plan.rotations == ()
        else:
            assert s.kind == P.TETRA
            assert plan.rotations == (f"9728_{layer:08x}0113feed",)
            assert (s.nblocks, s.tail_cols) == (405, 8)


def test_every_linear_layer_of_the_4b_has_a_plan(q4b_qc):
    kinds = {P.TETRA: 0, P.INT4: 0}
    for layer in range(36):
        for leaf in ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj",
                     "mlp.down_proj"):
            for s in P.layer_plan(q4b_qc, f"model.layers.{layer}.{leaf}", PACKED).shards:
                kinds[s.kind] += 1
    assert kinds == {P.TETRA: 168, P.INT4: 84}


def test_the_4b_rotations_each_have_one_owner(q4b_qc):
    owners = P.rotation_owners(q4b_qc, PACKED)
    assert len(owners) == len(q4b_qc["rotations"]) == 96
    for key, owner in owners.items():
        layer = int(key.split("_")[1][:8], 16)
        group = key[-8:]
        want = {"0110feed": "self_attn.qkv_proj", "0112feed": "mlp.gate_up_proj",
                "0113feed": "mlp.down_proj"}[group]
        assert owner == f"model.layers.{layer}.{want}"
    renames = P.rotation_renames(q4b_qc, PACKED)
    key = "2560_000000000110feed"
    assert renames[f"llvq.rotations.{key}."] == f"{L0}.self_attn.qkv_proj.llvq_rot_{key}_"


def test_a_key_read_by_two_layers_is_owned_by_one(mini_qc):
    """On the fixture one 136-wide rotation serves q, k, v, gate and up."""
    owners = P.rotation_owners(mini_qc, PACKED)
    assert owners == {
        "136_0000000000005eed": f"{L0}.mlp.gate_up_proj",
        "128_0000000000005eed": f"{L0}.self_attn.o_proj",
    }
    assert P.owned_keys(mini_qc, PACKED, f"{L0}.self_attn.qkv_proj") == []
    qkv = P.layer_plan(mini_qc, f"{L0}.self_attn.qkv_proj", PACKED)
    assert qkv.rotations == ("136_0000000000005eed",)


def test_an_unused_rotation_table_is_dropped_by_name(mini_qc):
    qc = copy.deepcopy(mini_qc)
    qc["rotations"]["136_00000000deadbeef"] = dict(qc["rotations"]["136_0000000000005eed"])
    assert P.rotation_renames(qc, PACKED)["llvq.rotations.136_00000000deadbeef."] is None


def test_the_mixed_layer_lays_its_slots_out_by_shard(q4b_qc):
    plan = P.layer_plan(q4b_qc, f"{L0}.self_attn.qkv_proj", PACKED)
    specs = {s.name: s for s in P.param_specs(plan)}
    assert set(specs) == {"codes", "row_scales", "centroids", "tail", "qweight", "scales",
                          "biases"}
    codes = specs["codes"].slots
    assert (codes[0].offset, codes[0].shape) == (0, (4096 * 106 * 6,))
    assert (codes[1].offset, codes[1].shape) == (4096 * 106 * 6, (1024 * 106 * 6,))
    assert specs["tail"].slots[1].shape == (1024, 16)
    assert set(specs["qweight"].slots) == {2}
    assert specs["qweight"].slots[2].shape == (1024, 1280)
    assert specs["scales"].slots[2].shape == (1024, 20)
    assert specs["centroids"].numel == 4


@pytest.mark.parametrize("shard_id, index", [("q", 0), ("k", 1), ("v", 2), (0, 0), (1, 1),
                                             (None, None)])
def test_shard_ids_name_positions(shard_id, index):
    assert P.shard_index(shard_id) == index


def test_an_unknown_shard_id_is_refused():
    with pytest.raises(ValueError, match="unknown shard id"):
        P.shard_index("w")


def test_a_layer_without_a_record_is_refused(mini_qc):
    with pytest.raises(KeyError, match="no record for model.layers.0.mlp.fc1"):
        P.layer_plan(mini_qc, f"{L0}.mlp.fc1", PACKED)


def test_the_vocabulary_is_int4_g64_and_a_norm_is_not(q4b_qc):
    assert P.embedding_desc(q4b_qc, "model.embed_tokens") == {
        "rows": 151936, "dim": 2560, "gpr": 40, "prefix": "model.embed_tokens"}
    assert P.embedding_desc(q4b_qc, "model.norm") is None
    assert P.embedding_desc(q4b_qc, "lm_head") is None


@pytest.mark.parametrize("field, value, match", [
    ("code_order", "lsb", "code_order"),
    ("block_dim", 16, "block_dim"),
    ("quant_method", "awq", "quant_method"),
])
def test_a_header_this_reader_does_not_unpack_is_refused(mini_qc, field, value, match):
    qc = copy.deepcopy(mini_qc)
    qc[field] = value
    with pytest.raises(ValueError, match=match):
        P.validate_header(qc)


def test_a_tetra_record_without_a_rotation_is_refused(mini_qc):
    qc = copy.deepcopy(mini_qc)
    del qc["records"][f"{L0}.self_attn.q_proj.weight"]["rotation"]
    with pytest.raises(ValueError, match="without a rotation"):
        P.validate_header(qc)


def test_an_int4_record_of_another_group_is_refused(mini_qc):
    qc = copy.deepcopy(mini_qc)
    qc["records"][f"{L0}.mlp.down_proj.weight"]["group"] = 64
    with pytest.raises(ValueError, match="hard-codes 128"):
        P.validate_header(qc)
