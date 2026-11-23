# /// script
# requires-python = ">=3.11"
# dependencies = ["gguf @ git+https://github.com/ggml-org/llama.cpp@690b87d881#subdirectory=gguf-py", "numpy"]
# ///
"""Expert tensors of an Inkling 2 Mini GGUF rung, dequantized for transplant, and their stored bits.

Reference orientation: `w13[e]` is [2304 out, 2304 in] with rows 0..1151 gate and 1152..2303 up, and `w2[e]` is
[2304 out, 1152 in]. Both are used as `F.linear(x, W)`, as in transformers' `modeling_inkling2_loop.py`. GGUF stores
`ne = [in, out, expert]` with `ne[0]` fastest, so the C-order view of one expert's rows is already [out, in]. Validated
against the BF16 GGUF of ggml-org, which was itself checked against the native checkpoint (`eval/ref_bf16/README.md`).

Layer mapping: decoder layer L (0..23, `model.llm.layers.L` in the native checkpoint, `blk.L` in the BF16 GGUF)
is tensor `blk.L.ffn_{gate_up,down}_exps.weight`. The unsloth root files say `inkling2.block_count = 48`: that is the
number of executed blocks, 6 + 12 * 3 (`num_loops` 3 over `loop.core_start` 6 .. `core_end` 18) + 6. Only 24
physical blocks are stored, `blk.0` .. `blk.23`, one expert tensor pair each, so a looped layer (6..17) has one
weight set reused on its three passes. Checked by correlation against the BF16 reference of the same layer index for
layer 0, layer 10 (looped) and layer 23: the matching layer gives 0.87-0.99, the adjacent layer under 0.01
(`ggufx_validate.py`, `eval/gguf_validate.txt`).

Layers 0-19 of the unsloth root files carry a per-expert F32 tensor `<name>.scale` of shape [128]. The stored weight
is multiplied by `scale[e]`; this is the NVFP4 outer factor amax / denominator of the native checkpoint.

Parsing: the header is read with a small structural reader (structure only, no decoding). Dequantization goes through
`gguf.quants.dequantize` only, from the official gguf-py pinned at llama.cpp 690b87d881: PyPI gguf 0.19.0 lacks
types 40 (NVFP4) and 42 (Q2_0), which these files use for most down tensors (operator decision, 2026-11-21). A type that the installed `gguf` cannot dequantize raises `UnsupportedType`; there is
no in-house decoder.

Usage: uv run ggufx.py [--out bits.json] NAME=path.gguf [...]
"""
from __future__ import annotations

import json
import mmap
import struct
import sys

import numpy as np
import gguf
from gguf.quants import dequantize

N_EXPERT, HIDDEN, FF = 128, 2304, 1152
SETS = {"6-8": list(range(6, 9)), "6-11": list(range(6, 12)), "6-17": list(range(6, 18)),
        "0-5": list(range(0, 6)), "18-23": list(range(18, 24)), "all": list(range(24))}
# ggml type id -> (name, block size, bytes per block), from ggml.h on llama.cpp master; used for byte accounting only.
GGML = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 8: ("Q8_0", 32, 34), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
        12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 16: ("IQ2_XXS", 256, 66),
        17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18),
        21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 29: ("IQ1_M", 256, 56),
        30: ("BF16", 1, 2), 39: ("MXFP4", 32, 17), 40: ("NVFP4", 64, 36), 41: ("Q1_0", 128, 18),
        42: ("Q2_0", 64, 18), 43: ("F8_E4M3", 1, 1)}


class UnsupportedType(RuntimeError):
    pass


_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_CACHE: dict[str, tuple] = {}


def _open(path: str):
    """(mmap, tensors by name, data start). Each tensor is {"ne", "type", "offset"}."""
    if path in _CACHE:
        return _CACHE[path]
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    pos = 0

    def rd(fmt):
        nonlocal pos
        v = struct.unpack_from(fmt, mm, pos)[0]
        pos += struct.calcsize(fmt)
        return v

    def rstr():
        nonlocal pos
        n = rd("<Q")
        s = mm[pos:pos + n].decode("utf-8", "replace")
        pos += n
        return s

    def rval(t):
        if t == 8:
            return rstr()
        if t == 9:
            et, n = rd("<I"), rd("<Q")
            return [rval(et) for _ in range(n)]
        return rd(_FMT[t])

    if mm[:4] != b"GGUF":
        raise ValueError(f"{path}: not a GGUF file")
    pos = 4
    version, n_t, n_kv = rd("<I"), rd("<Q"), rd("<Q")
    if version != 3:
        raise ValueError(f"{path}: GGUF version {version}, expected 3")
    kv = {}
    for _ in range(n_kv):
        k = rstr()
        kv[k] = rval(rd("<I"))
    tensors = {}
    for _ in range(n_t):
        name = rstr()
        nd = rd("<I")
        ne = [rd("<Q") for _ in range(nd)]
        tensors[name] = {"ne": ne, "type": rd("<I"), "offset": rd("<Q")}
    align = kv.get("general.alignment", 32)
    data_start = (pos + align - 1) // align * align
    _CACHE[path] = (mm, tensors, data_start)
    return _CACHE[path]


def _nbytes(t) -> tuple[int, int]:
    n = int(np.prod(t["ne"]))
    _, blk, bb = GGML[t["type"]]
    return n, n // blk * bb


def _qtype(type_id: int):
    try:
        q = gguf.GGMLQuantizationType(type_id)
    except ValueError:
        q = None
    if q is None or (q not in gguf.quants._type_traits and q not in (gguf.GGMLQuantizationType.F32,
                                                                       gguf.GGMLQuantizationType.F16)):
        raise UnsupportedType(
            f"ggml type {type_id} ({GGML.get(type_id, ('?',))[0]}) is not dequantizable by gguf "
            f"{_gguf_version()}; llama.cpp master gguf-py has NVFP4 and Q2_0 (commit 495c5805ce, #30630)")
    return q


def _gguf_version() -> str:
    try:
        from importlib.metadata import version
        return version("gguf")
    except Exception:
        return "?"


def _expert_raw(path: str, name: str, e0: int, e1: int):
    mm, tensors, ds = _open(path)
    t = tensors[name]
    n_in, n_out, n_exp = t["ne"]
    assert n_exp == N_EXPERT, (name, t["ne"])
    _, total = _nbytes(t)
    per = total // n_exp
    raw = np.frombuffer(mm, dtype=np.uint8, count=per * (e1 - e0), offset=ds + t["offset"] + e0 * per)
    return raw.reshape(e1 - e0, n_out, per // n_out), t, (n_out, n_in)


def _scale(path: str, name: str):
    mm, tensors, ds = _open(path)
    s = tensors.get(name.replace(".weight", ".scale"))
    if s is None:
        return None
    assert s["type"] == 0 and s["ne"] == [N_EXPERT], s
    return np.frombuffer(mm, dtype="<f4", count=N_EXPERT, offset=ds + s["offset"]).copy()


def _load(path: str, name: str, e0: int, e1: int, apply_scale: bool = True) -> np.ndarray:
    raw, t, (n_out, n_in) = _expert_raw(path, name, e0, e1)
    q = _qtype(t["type"])
    w = dequantize(raw, q).astype(np.float32, copy=False).reshape(e1 - e0, n_out, n_in)
    s = _scale(path, name) if apply_scale else None
    if s is not None:
        w *= s[e0:e1, None, None]
    return w


def load_gguf_experts(path: str, layer: int) -> tuple[np.ndarray, np.ndarray]:
    """Float32 `w13` [128, 2304, 2304] and `w2` [128, 2304, 1152], [expert, out, in], of `layer`."""
    w13 = np.empty((N_EXPERT, 2 * FF, HIDDEN), np.float32)
    w2 = np.empty((N_EXPERT, HIDDEN, FF), np.float32)
    for e0 in range(0, N_EXPERT, 8):
        w13[e0:e0 + 8] = _load(path, f"blk.{layer}.ffn_gate_up_exps.weight", e0, e0 + 8)
        w2[e0:e0 + 8] = _load(path, f"blk.{layer}.ffn_down_exps.weight", e0, e0 + 8)
    return w13, w2


def gguf_expert_bits(path: str, layers: list[int]) -> dict:
    """Types met and expert bits per weight (stored bytes * 8 / weights), per layer and pooled.

    Bytes count `ffn_gate_up_exps` + `ffn_down_exps` and their per-expert F32 `.scale` when present.
    """
    _, tensors, _ = _open(path)
    per, tb, tw = {}, 0, 0
    for L in layers:
        types, b, w = {}, 0, 0
        for kind in ("ffn_gate_up_exps", "ffn_down_exps"):
            t = tensors[f"blk.{L}.{kind}.weight"]
            n, nb = _nbytes(t)
            types[kind] = GGML[t["type"]][0]
            b, w = b + nb, w + n
            s = tensors.get(f"blk.{L}.{kind}.scale")
            if s is not None:
                b += _nbytes(s)[1]
                types[kind + ".scale"] = GGML[s["type"]][0]
        per[L] = {"types": types, "bytes": b, "weights": w, "bits": b * 8 / w}
        tb, tw = tb + b, tw + w
    return {"per_layer": per, "pooled": {"bytes": tb, "weights": tw, "bits": tb * 8 / tw}}


def _main(argv: list[str]) -> None:
    out = None
    if argv[:1] == ["--out"]:
        out, argv = argv[1], argv[2:]
    res = {}
    for a in argv:
        name, p = a.split("=", 1) if "=" in a else (a.rsplit("/", 1)[-1], a)
        res[name] = {"path": p, "sets": {k: gguf_expert_bits(p, v)["pooled"]["bits"] for k, v in SETS.items()},
                     "per_layer": gguf_expert_bits(p, SETS["all"])["per_layer"]}
    print("expert bits per weight (gate_up + down + per-expert F32 scales; stored bytes * 8 / weights)")
    print(f"{'file':10s} " + " ".join(f"{k:>7s}" for k in SETS))
    for n, r in res.items():
        print(f"{n:10s} " + " ".join(f"{r['sets'][k]:7.4f}" for k in SETS))
    print("types per layer, gate_up/down (* = per-expert .scale present):")
    for n, r in res.items():
        cells = []
        for L, v in r["per_layer"].items():
            t = v["types"]
            g = t["ffn_gate_up_exps"] + ("*" if "ffn_gate_up_exps.scale" in t else "")
            d = t["ffn_down_exps"] + ("*" if "ffn_down_exps.scale" in t else "")
            cells.append(f"{L}:{g}/{d}")
        print(f"  {n}: " + " ".join(cells))
    if out:
        json.dump(res, open(out, "w"), indent=1)
        print("wrote", out)


if __name__ == "__main__":
    _main(sys.argv[1:])
