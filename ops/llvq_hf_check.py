# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "safetensors>=0.4"]
# ///
"""Rebuild every field of a packed LLVQ directory, from the safetensors alone.

The gate of stage 0 of `docs/plan-transformers.md`, preregistered in
`proofs/preregistration-hf-safetensors-2026-09-28.md`. `llvq_llm::hfpack` wrote
the directory and, beside it, one SHA-256 per field computed from the fields as
`llvq-artifact` reads them out of the sealed `.llvq`. This script recomputes
every one of those hashes from the written tensors and from nothing else.

## What makes it a check and not a copy

The two paths share no code. This one reads safetensors, takes the record table
from `config.json`'s `quantization_config` (the metadata a loader will use, so
the check exercises it too), and unpacks the 48-bit words itself. Three of the
hashes are over values this file has to *derive*:

* `indices` and `gains`, MSB-first out of the `codes` bytes. A reader that
  guessed little-endian, or put the gain bit first, fails here.
* `tail`, whose f32 patterns must survive being read as a 2-D tensor.

A hash over the `codes` bytes alone would only have proved a byte array
survived a copy, which is why it is not the only one.

## Usage

    uv run ops/llvq_hf_check.py <directory>

Exit 0 and one line per section when every field agrees. Exit 1 naming the first
field that does not, with the two hashes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

# The convention `llvq_llm::hfpack` fixes: little-endian bit patterns, in the
# order the field stores them. Every dtype below is therefore explicit about
# byte order, and nothing relies on the platform's.
LE = {"f64": "<f8", "f32": "<f4", "f16": "<f2", "u32": "<u4", "u64": "<u8"}

if sys.byteorder != "little":
    raise SystemExit("this reader assumes a little-endian host, as safetensors stores")


class Mismatch(Exception):
    """A field whose recomputed hash differs from the archive's."""


def sha256(data) -> str:
    return hashlib.sha256(memoryview(data)).hexdigest()


def hash_bits(a: np.ndarray, dtype: str) -> str:
    """SHA-256 over the bytes of `a`, which must already be `dtype`.

    No cast, ever. Casting an f16 array to u16 converts the values, so a scale
    of 0.001 hashes as a zero and every digest of a quantized record passes for
    the wrong reason. It cost one red run on 2026-09-28 to find out, and a
    dtype that is not the expected one is now a refusal.
    """
    want = np.dtype(LE[dtype])
    if a.dtype != want:
        raise Mismatch(f"dtype {a.dtype} where {want} was expected")
    return sha256(np.ascontiguousarray(a).tobytes())


def unpack_codes(codes: np.ndarray, nblocks: int, index_bits: int, gain_bits: int):
    """`(indices, gains)` out of an MSB-first dense stream of fixed-width pairs.

    Only the byte-aligned case is implemented, which is every Tetra record: 47
    index bits and 1 gain bit make 48, six bytes a block, so block `b` occupies
    bytes `6b … 6b+6` and nothing straddles a boundary. A Ball record is 47 to
    50 bits and is refused by the packer, so a stream that lands here unaligned
    means the two sides disagree about a width, which is the whole point of
    saying so instead of guessing.
    """
    width = index_bits + gain_bits
    if width % 8:
        raise Mismatch(
            f"{width} bits a block is not byte aligned; this reader handles the "
            "aligned case only, and the packer writes no other kind"
        )
    stride = width // 8
    need = nblocks * stride
    if codes.size != need:
        raise Mismatch(f"{codes.size} code bytes for {nblocks} blocks of {stride}")
    words = codes.reshape(nblocks, stride).astype(np.uint64)
    # MSB first: the first byte carries the highest eight bits.
    value = np.zeros(nblocks, dtype=np.uint64)
    for i in range(stride):
        value = (value << np.uint64(8)) | words[:, i]
    gain_mask = np.uint64((1 << gain_bits) - 1)
    return value >> np.uint64(gain_bits), (value & gain_mask).astype(np.uint32)


def check(field: str, got: str, want: str, checked: list[str]) -> None:
    if got != want:
        raise Mismatch(f"{field}\n  archive  {want}\n  rebuilt  {got}")
    checked.append(field)


def check_records(f, qc: dict, digests: dict, checked: list[str]) -> None:
    for name, desc in qc["records"].items():
        want = digests[name]
        prefix = desc["prefix"]
        if desc["kind"] == "tetra":
            codes = f.get_tensor(f"{prefix}.codes")
            if codes.dtype != np.uint8:
                raise Mismatch(f"{prefix}.codes is {codes.dtype}, not uint8")
            nblocks = desc["d_out"] * desc["nblocks"]
            check(f"{name}.codes", sha256(codes.tobytes()), want["codes"], checked)
            indices, gains = unpack_codes(
                codes, nblocks, desc["index_bits"], desc["gain_bits"]
            )
            check(f"{name}.indices", hash_bits(indices, "u64"), want["indices"], checked)
            check(f"{name}.gains", hash_bits(gains, "u32"), want["gains"], checked)

            scales = f.get_tensor(f"{prefix}.row_scales")
            if scales.shape != (desc["d_out"],):
                raise Mismatch(f"{prefix}.row_scales is {scales.shape}")
            check(f"{name}.row_scales", hash_bits(scales, "f64"), want["row_scales"], checked)
            cent = f.get_tensor(f"{prefix}.centroids")
            if cent.shape != (desc["n_centroids"],):
                raise Mismatch(f"{prefix}.centroids is {cent.shape}")
            check(f"{name}.centroids", hash_bits(cent, "f64"), want["centroids"], checked)
            if desc["tail_cols"]:
                tail = f.get_tensor(f"{prefix}.tail")
                if tail.shape != (desc["d_out"], desc["tail_cols"]):
                    raise Mismatch(f"{prefix}.tail is {tail.shape}")
                check(f"{name}.tail", hash_bits(tail, "f32"), want["tail"], checked)
            elif "tail" in want:
                raise Mismatch(f"{name}: a tail digest for a record with no tail")
        elif desc["kind"] == "int4g128":
            q = f.get_tensor(f"{prefix}.qweight")
            if q.shape != (desc["d_out"], desc["d_in"] // 2):
                raise Mismatch(f"{prefix}.qweight is {q.shape}")
            check(f"{name}.qweight", sha256(q.tobytes()), want["qweight"], checked)
            for field in ("scales", "biases"):
                t = f.get_tensor(f"{prefix}.{field}")
                if t.shape != (desc["d_out"], desc["groups_per_row"]):
                    raise Mismatch(f"{prefix}.{field} is {t.shape}")
                check(f"{name}.{field}", hash_bits(t, "f16"), want[field], checked)
        else:
            raise Mismatch(f"{name}: unknown record kind {desc['kind']}")


def check_rotation_references(qc: dict) -> None:
    """Every record's rotation exists, and every rotation is used.

    No digest covers this: a record pointing at a key with no table would pass
    every hash and reconstruct its weights in the wrong basis, which is the one
    failure mode of this layout that produces plausible output.
    """
    used = {r["rotation"] for r in qc["records"].values() if r.get("rotation")}
    have = set(qc["rotations"])
    if used - have:
        raise Mismatch(f"records point at rotations that are not in the file: {sorted(used - have)}")
    if have - used:
        raise Mismatch(f"rotations no record uses: {sorted(have - used)}")


def check_rotations(f, qc: dict, digests: dict, checked: list[str]) -> None:
    for key, desc in qc["rotations"].items():
        want = digests[key]
        signs = f.get_tensor(f"llvq.rotations.{key}.signs")
        if signs.shape != (desc["n"],):
            raise Mismatch(f"rotation {key}: signs are {signs.shape}, want {desc['n']}")
        if not np.all(np.abs(signs) == 1.0):
            raise Mismatch(f"rotation {key}: signs are not all plus or minus one")
        check(f"rot {key}.signs", hash_bits(signs, "f64"), want["signs"], checked)
        small = f.get_tensor(f"llvq.rotations.{key}.small")
        k = desc["odd"]
        if small.shape != (k, k):
            raise Mismatch(f"rotation {key}: small is {small.shape}, want {k} by {k}")
        # Orthogonality is not in any digest, and a wrong table would still hash
        # to itself. Checked here because it is cheap and it is the property the
        # weights depend on.
        gram = small.astype(np.float64) @ small.astype(np.float64).T
        off = np.abs(gram - np.eye(k)).max() if k else 0.0
        if off > 1e-12:
            raise Mismatch(f"rotation {key}: small is not orthogonal, {off:.3e} off")
        check(f"rot {key}.small", hash_bits(small, "f64"), want["small"], checked)
        if desc["n"] != desc["pow2"] * desc["odd"]:
            raise Mismatch(f"rotation {key}: {desc['pow2']} times {desc['odd']} is not {desc['n']}")


def check_raw(f, qc: dict, digests: dict, checked: list[str]) -> None:
    for name, desc in qc["raw"].items():
        want = digests[name]
        dims = tuple(desc["dims"])
        if desc["encoding"] == "f16":
            t = f.get_tensor(name)
            if t.shape != dims:
                raise Mismatch(f"{name} is {t.shape}, want {dims}")
            check(f"{name}.values", hash_bits(t, "f16"), want["values"], checked)
        elif desc["encoding"] == "quant":
            prefix, rows, gpr = desc["prefix"], desc["rows"], desc["groups_per_row"]
            row_len = dims[-1]
            cols = row_len // 2 if desc["bits"] == 4 else row_len
            q = f.get_tensor(f"{prefix}.qweight")
            if q.shape != (rows, cols):
                raise Mismatch(f"{prefix}.qweight is {q.shape}, want {(rows, cols)}")
            check(f"{name}.qweight", sha256(q.tobytes()), want["qweight"], checked)
            for field in ("scales", "biases"):
                t = f.get_tensor(f"{prefix}.{field}")
                if t.shape != (rows, gpr):
                    raise Mismatch(f"{prefix}.{field} is {t.shape}, want {(rows, gpr)}")
                check(f"{name}.{field}", hash_bits(t, "f16"), want[field], checked)
        else:
            raise Mismatch(f"{name}: unknown raw encoding {desc['encoding']}")


def check_blobs(d: Path, digests: dict, base: dict, written: dict, checked: list[str]) -> None:
    """The carried files. `config.json` is the one that cannot be verbatim.

    Stage 0 puts a `quantization_config` block in it, so the check is that every
    key of the sealed blob is present with an equal value and that the only
    added key is that block. Compared as parsed values, never as text, so no
    number's formatting enters the claim.
    """
    for name, want in digests.items():
        if want["written_verbatim"]:
            got = sha256((d / name).read_bytes())
            check(f"blob {name}", got, want["sha256"], checked)
    added = set(written) - set(base)
    if added != {"quantization_config"}:
        raise Mismatch(f"config.json adds {sorted(added)}, want quantization_config alone")
    for key, value in base.items():
        if key not in written:
            raise Mismatch(f"config.json lost the key {key}")
        if written[key] != value:
            raise Mismatch(f"config.json changed {key}: {value!r} became {written[key]!r}")
    checked.append(f"config.json ({len(base)} keys carried, quantization_config added)")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.split("## Usage")[1].strip(), file=sys.stderr)
        return 2
    d = Path(argv[1])
    digests = json.loads((d / "llvq-digest.json").read_text())
    written = json.loads((d / "config.json").read_text())
    qc = written["quantization_config"]

    if qc["quant_method"] != "llvq":
        raise Mismatch(f"quant_method is {qc['quant_method']!r}")
    if qc["code_order"] != "msb_first_dense":
        raise Mismatch(f"code_order is {qc['code_order']!r}, which this reader does not unpack")
    if qc["artifact_sha256"] != digests["artifact"]["sha256"]:
        raise Mismatch("config.json and llvq-digest.json name two different artifacts")

    counts = digests["counts"]
    kinds = {}
    for desc in qc["records"].values():
        kinds[desc["kind"]] = kinds.get(desc["kind"], 0) + 1
    if kinds.get("tetra", 0) != counts["lattice"] or kinds.get("int4g128", 0) != counts["int4"]:
        raise Mismatch(f"the record table holds {kinds}, the archive counted {counts}")
    if len(qc["records"]) != digests["artifact"]["matrices"]:
        raise Mismatch(f"{len(qc['records'])} records described, {digests['artifact']['matrices']} in the file")
    if len(qc["raw"]) != counts["raw_tensors"] or len(qc["rotations"]) != counts["rotations"]:
        raise Mismatch("the raw or rotation table does not match the archive's count")

    weights = sum(r["d_out"] * r["d_in"] for r in qc["records"].values())
    if weights != counts["quantized_weights"]:
        raise Mismatch(f"{weights} weights described, {counts['quantized_weights']} in the file")
    carried = sum(int(np.prod(r["dims"])) for r in qc["raw"].values())
    if carried != counts["carried_weights"]:
        raise Mismatch(f"{carried} carried weights described, {counts['carried_weights']} in the file")

    checked: list[str] = []
    with safe_open(d / "model.safetensors", framework="np") as f:
        names = set(f.keys())
        check_records(f, qc, digests["records"], checked)
        check_rotation_references(qc)
        check_rotations(f, qc, digests["rotations"], checked)
        check_raw(f, qc, digests["raw"], checked)
        if len(names) != counts["tensors"]:
            raise Mismatch(f"{len(names)} tensors in the file, {counts['tensors']} written")
    check_blobs(d, digests["blobs"], digests["config_base"], written, checked)

    print(f"{d}")
    print(f"  artifact   {digests['artifact']['sha256'][:16]}… v{digests['artifact']['version']}, "
          f"kinds {digests['artifact']['kinds']}")
    print(f"  records    {counts['lattice']} Tetra, {counts['int4']} Int4G128, "
          f"{weights / 1e9:.3f} B weights")
    print(f"  carried    {counts['raw_tensors']} raw tensors, {carried / 1e6:.1f} M weights")
    print(f"  rotations  {counts['rotations']} tables")
    print(f"  {len(checked)} fields rebuilt bit for bit, every digest of the archive matched")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except Mismatch as e:
        print(f"MISMATCH: {e}", file=sys.stderr)
        sys.exit(1)
