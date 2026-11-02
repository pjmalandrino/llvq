"""Which records of a packed file back which layer of vLLM, in what order.

Pure Python over the `quantization_config` block `hfpack` writes, so every rule
here is checked on a Mac against the real record table of the published 4B.

## The two names of one projection

The file names a record by its `transformers` module, `...self_attn.q_proj`.
vLLM fuses q, k and v into one `QKVParallelLinear` named `...self_attn.qkv_proj`,
and gate and up into one `MergedColumnParallelLinear` named `...mlp.gate_up_proj`.
The model class says which parts make which layer in `packed_modules_mapping`,
and vLLM hands that mapping to the quantization config before any layer is
built (`model_loader/utils.py::configure_quant_config`, vLLM 0.26.0). So one vLLM
layer is a list of records, the shards, in the order of that mapping, which is
the order of the layer's output columns.

## The shards of one layer are not of one kind

On the sealed 4B, `qkv_proj` holds q and k as Tetra records and v as an int4
g128 record. The Tetra shards share a rotation key and read the activation in
the rotated basis; the int4 shard reads it in the natural basis. A plan carries
each shard's kind and rotation, and `layer.py` rotates once per key.

## Where the rotation tables go

The file carries each rotation once, as `llvq.rotations.<key>.signs` and
`.small`, and several records point at it. vLLM's loader has no slot for a
tensor that belongs to no layer. So each key is given an owner: the vLLM layer
of the first record, in sorted order, that uses it. The owner holds the tables
as two parameters of its own, and the weight mapper renames the file's tensors
onto them. A layer that uses a key it does not own reads the owner's tables
after the load.
"""

from __future__ import annotations

from dataclasses import dataclass, field

QUANT_METHOD = "llvq"
TETRA = "tetra"
INT4 = "int4g128"
CODE_ORDER = "msb_first_dense"
BLOCK_DIM = 24
INT4_GROUP = 128
EMBED_GROUP = 64
TETRA_INDEX_BITS = 47
TETRA_GAIN_BITS = 1

# vLLM's two shard-id conventions: `QKVParallelLinear` takes "q", "k", "v" and
# `MergedColumnParallelLinear` takes the shard's position. Both reach a weight
# loader as whatever the model's `hf_to_vllm_mapper` stamped on the tensor.
QKV_IDS = {"q": 0, "k": 1, "v": 2}


def shard_index(shard_id) -> int | None:
    """The position a vLLM shard id names, or `None` for an unfused layer."""
    if shard_id is None:
        return None
    if isinstance(shard_id, str):
        if shard_id not in QKV_IDS:
            raise ValueError(f"unknown shard id {shard_id!r}")
        return QKV_IDS[shard_id]
    if isinstance(shard_id, int):
        return shard_id
    raise ValueError(f"shard id {shard_id!r} is neither a name nor a position")


@dataclass(frozen=True)
class Shard:
    """One record of the file, as one slice of a vLLM layer's output."""

    index: int
    record: str
    prefix: str
    kind: str
    d_out: int
    d_in: int
    desc: dict = field(compare=False, repr=False)

    @property
    def rotation(self) -> str | None:
        return self.desc.get("rotation") if self.kind == TETRA else None

    @property
    def nblocks(self) -> int:
        return int(self.desc["nblocks"])

    @property
    def tail_cols(self) -> int:
        return int(self.desc["tail_cols"])

    @property
    def gpr(self) -> int:
        return int(self.desc["groups_per_row"])


@dataclass(frozen=True)
class LayerPlan:
    prefix: str
    shards: tuple[Shard, ...]

    @property
    def d_in(self) -> int:
        return self.shards[0].d_in

    @property
    def d_out(self) -> int:
        return sum(s.d_out for s in self.shards)

    @property
    def rotations(self) -> tuple[str, ...]:
        """The distinct rotation keys of the Tetra shards, in shard order."""
        seen: list[str] = []
        for s in self.shards:
            if s.rotation and s.rotation not in seen:
                seen.append(s.rotation)
        return tuple(seen)

    def by_kind(self, kind: str) -> tuple[Shard, ...]:
        return tuple(s for s in self.shards if s.kind == kind)


@dataclass(frozen=True)
class Slot:
    """Where one shard's tensor lands inside a layer's flat parameter."""

    offset: int
    shape: tuple[int, ...]

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


@dataclass(frozen=True)
class ParamSpec:
    """One parameter a layer registers: its checkpoint suffix, dtype, and slots.

    The name is the suffix the file gives the tensor, `codes` or `qweight`, and
    not a name of ours: vLLM's fused layers look a loaded tensor up with
    `getattr(layer, name)`, so any other name would land on the layer itself.
    """

    name: str
    dtype: str
    slots: dict[int, Slot]

    @property
    def numel(self) -> int:
        return sum(s.numel for s in self.slots.values())


def validate_header(qc: dict) -> None:
    """Refuse a config block this reader does not understand, before any layer."""
    if qc.get("quant_method") != QUANT_METHOD:
        raise ValueError(f"quant_method is {qc.get('quant_method')!r}, not {QUANT_METHOD!r}")
    if qc.get("code_order") != CODE_ORDER:
        raise ValueError(f"code_order {qc.get('code_order')!r} is not one this reader unpacks")
    if int(qc.get("block_dim", 0)) != BLOCK_DIM:
        raise ValueError(f"block_dim {qc.get('block_dim')} is not {BLOCK_DIM}")
    if not qc.get("records"):
        raise ValueError("the quantization_config carries no record table")
    for name, d in qc["records"].items():
        if d["kind"] == TETRA:
            if (int(d["index_bits"]), int(d["gain_bits"])) != (TETRA_INDEX_BITS, TETRA_GAIN_BITS):
                raise ValueError(f"{name}: {d['index_bits']}+{d['gain_bits']} bits a block, "
                                 "the served layout is 47+1")
            if int(d["n_centroids"]) != 2:
                raise ValueError(f"{name}: {d['n_centroids']} centroids, the kernel binds two")
            if int(d["d_in"]) != int(d["nblocks"]) * BLOCK_DIM + int(d["tail_cols"]):
                raise ValueError(f"{name}: d_in is not nblocks·24 + tail_cols")
            if not d.get("rotation"):
                raise ValueError(f"{name}: a Tetra record without a rotation; the kernel "
                                 "reads rotated weights only")
        elif d["kind"] == INT4:
            if int(d["group"]) != INT4_GROUP:
                raise ValueError(f"{name}: group {d['group']}, tv_q4_h hard-codes 128")
            if int(d["d_in"]) % 8:
                raise ValueError(f"{name}: d_in {d['d_in']} is not a multiple of 8")
        else:
            raise ValueError(f"{name}: unknown record kind {d['kind']!r}")


def layer_plan(qc: dict, vllm_prefix: str, packed: dict[str, list[str]]) -> LayerPlan:
    """The records behind one vLLM linear layer, in output-column order."""
    by_prefix = {d["prefix"]: (name, d) for name, d in qc["records"].items()}
    parent, _, leaf = vllm_prefix.rpartition(".")
    parts = packed.get(leaf, [leaf])
    shards = []
    for i, part in enumerate(parts):
        p = f"{parent}.{part}" if parent else part
        if p not in by_prefix:
            raise KeyError(
                f"{vllm_prefix}: no record for {p} in the file. Every linear layer of the "
                "model must be a record; a dense projection has no tensor to load"
            )
        name, d = by_prefix[p]
        shards.append(Shard(i, name, p, d["kind"], int(d["d_out"]), int(d["d_in"]), d))
    d_ins = {s.d_in for s in shards}
    if len(d_ins) != 1:
        raise ValueError(f"{vllm_prefix}: the shards read different widths {sorted(d_ins)}")
    tails = {s.tail_cols for s in shards if s.kind == TETRA}
    if len(tails) > 1:
        raise ValueError(f"{vllm_prefix}: the Tetra shards carry different tails {sorted(tails)}")
    return LayerPlan(vllm_prefix, tuple(shards))


def param_specs(plan: LayerPlan) -> list[ParamSpec]:
    """The flat parameters a layer registers, one slot per shard that has the tensor."""
    specs: dict[str, dict[int, Slot]] = {}
    dtypes: dict[str, str] = {}
    cursor: dict[str, int] = {}

    def add(name: str, dtype: str, index: int, shape: tuple[int, ...]) -> None:
        slots = specs.setdefault(name, {})
        dtypes[name] = dtype
        off = cursor.get(name, 0)
        slots[index] = Slot(off, shape)
        cursor[name] = off + slots[index].numel

    for s in plan.shards:
        if s.kind == TETRA:
            add("codes", "uint8", s.index, (int(s.desc["code_bytes"]),))
            add("row_scales", "float64", s.index, (s.d_out,))
            add("centroids", "float64", s.index, (2,))
            if s.tail_cols:
                add("tail", "float32", s.index, (s.d_out, s.tail_cols))
        else:
            add("qweight", "uint8", s.index, (s.d_out, s.d_in // 2))
            add("scales", "float16", s.index, (s.d_out, s.gpr))
            add("biases", "float16", s.index, (s.d_out, s.gpr))
    return [ParamSpec(n, dtypes[n], specs[n]) for n in specs]


def vllm_layer_of(record_prefix: str, packed: dict[str, list[str]]) -> str:
    """The vLLM layer prefix a record lands in."""
    inverse = {part: name for name, parts in packed.items() for part in parts}
    parent, _, leaf = record_prefix.rpartition(".")
    layer = inverse.get(leaf, leaf)
    return f"{parent}.{layer}" if parent else layer


def rotation_owners(qc: dict, packed: dict[str, list[str]]) -> dict[str, str]:
    """`key -> vLLM layer prefix` that holds the key's two tables."""
    owners: dict[str, str] = {}
    for name in sorted(qc["records"]):
        d = qc["records"][name]
        key = d.get("rotation") if d["kind"] == TETRA else None
        if key and key not in owners:
            owners[key] = vllm_layer_of(d["prefix"], packed)
    missing = sorted(set(owners) - set(qc.get("rotations", {})))
    if missing:
        raise KeyError(f"records point at rotations the file does not carry: {missing[:3]}")
    return owners


def owned_keys(qc: dict, packed: dict[str, list[str]], vllm_prefix: str) -> list[str]:
    return sorted(k for k, o in rotation_owners(qc, packed).items() if o == vllm_prefix)


def rotation_param(key: str, table: str) -> str:
    """The parameter name the owner gives a table: `llvq_rot_<key>_<table>`."""
    if table not in ("signs", "small"):
        raise ValueError(f"a rotation has signs and small, not {table!r}")
    return f"llvq_rot_{key}_{table}"


def rotation_renames(qc: dict, packed: dict[str, list[str]]) -> dict[str, str | None]:
    """Checkpoint prefix to vLLM prefix for every rotation table in the file.

    `llvq.rotations.<key>.` becomes `<owner>.llvq_rot_<key>_`, so the tensor
    `...<key>.signs` lands on the owner's `llvq_rot_<key>_signs`. A table no
    record uses maps to `None`, which vLLM's `WeightsMapper` reads as "drop".
    """
    owners = rotation_owners(qc, packed)
    out: dict[str, str | None] = {}
    for key in qc.get("rotations", {}):
        src = f"llvq.rotations.{key}."
        out[src] = f"{owners[key]}.llvq_rot_{key}_" if key in owners else None
    return out


def embedding_desc(qc: dict, vllm_prefix: str) -> dict | None:
    """The carried table behind a vocabulary layer, when the file stores it int4.

    `None` when the file stores it in f16, which vLLM then loads unquantized.
    """
    d = qc.get("raw", {}).get(f"{vllm_prefix}.weight")
    if d is None or d.get("encoding") != "quant":
        return None
    if int(d["bits"]) != 4 or int(d["group"]) != EMBED_GROUP:
        raise ValueError(f"{vllm_prefix}: {d['bits']} bits group {d['group']}; the served "
                         "vocabulary is int4 g64")
    rows, dim = (int(x) for x in d["dims"])
    if dim % 8:
        raise ValueError(f"{vllm_prefix}: width {dim} is not a multiple of 8")
    if int(d["groups_per_row"]) != -(-dim // EMBED_GROUP):
        raise ValueError(f"{vllm_prefix}: groups_per_row is not ceil(d / 64)")
    return {"rows": rows, "dim": dim, "gpr": int(d["groups_per_row"]), "prefix": d["prefix"]}
