# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "torch==2.5.1",
#   "safetensors==0.4.5",
#   "numpy<2.4",
# ]
# ///
# Two ways to run this file. On the Mac, `uv run ops/hybrid_checkpoint.py --selftest` (CPU, tiny synthetic
# files, a few seconds). In the vLLM job image, `python3 ops/hybrid_checkpoint.py --build ...`; the header is a
# comment there and the image provides torch, safetensors and numpy (safetensors
# serializes BF16 through numpy).
"""One arm of the GSM8K depth restore: the f16 export of the sealed 4B with a window of layers restored to FP16.

Job: ops/jobs/gsm8k-depth-restore-4b.sh. Prereg: proofs/preregistration-gsm8k-depth-restore-4b-2026-11-14.md.

## The arms

  control  the export as it is (hard link to the staged file), no tensor replaced.
  late     the seven projections of layers 25 to 35, inclusive, 0-based: 77 matrices, 1,110,179,840 weights.
  early    the seven projections of layers 0 to 24, inclusive, 0-based: 175 matrices, 2,523,136,000 weights.

These are the windows of `bin/mmlu`'s `LLVQ_RESTORE_F16=25-35` and `0-24` in docs/mesures/depth-restore-4b-2026-11-14.txt.
Every other tensor (embedding, norms, q_norm, k_norm) is the export's, byte for byte.

## The cast

The checkpoint stores BF16. A restored matrix is `t.to(torch.float16)` on the CPU: BF16 widens to f32 exactly, then
f32 narrows to f16 with round to nearest, ties to even, overflow to infinity, gradual underflow. `bin/mmlu`'s f16
restore (`llvq-llm/src/sealed.rs`, `load_sealed`) does `st.load(name, device)?.to_dtype(F16)`: on the CPU candle runs
`f16::from_f32(v.to_f32())` (half 2.7, `f32_to_f16_fallback` without hardware support), on CUDA its `cast_bf16_f16`
kernel goes through `float`, which is the same rounding. vLLM, given the BF16 checkpoint with `dtype=float16`, also
casts with torch. So the three paths hold the same bits for every finite value. This script does not take that on
trust: `cast_reference` ports half's `f32_to_f16_fallback` line by line, the selftest checks torch's cast against it
on all 65,536 BF16 bit patterns but NaN, and every build checks it again on the first replaced matrix through a
lookup table. A NaN in a restored matrix is refused, since NaN payloads are where the two could part.

## What a build checks, by reading the written file back

1. Every replaced tensor is bit-equal to the FP16 tensor cast to f16, read afresh from the shard.
2. Every other tensor is bit-equal to the export's, same dtype, same shape.
3. The counts of replaced matrices and weights are the pinned ones; no key is added or lost.
4. A replaced tensor differs from the export's (a restore that writes the export back is a silent no-op).
5. The export's and the shards' sha256 are the pins of `ops/layer_vs_output.py`; the manifest records them, the
   written file's sha256, and the shards' revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch

PROJ_TYPES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ARMS = {"control": None, "late": (25, 35), "early": (0, 24)}
# Pinned from the shapes of Qwen3-4B (hidden 2560, 32 x 128 query heads, 8 KV heads, intermediate 9728): one layer
# holds 100,925,440 projection weights. Pinned here, not derived from PROJ_TYPES, so a type left out shows.
EXPECTED = {"control": (0, 0), "late": (77, 1_110_179_840), "early": (175, 2_523_136_000)}
N_LAYERS = 36
# The files an arm's directory takes from the FP16 snapshot, beside its own model.safetensors. Not the index:
# it would point vLLM at the FP16 shards.
SIDE_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
              "merges.txt")
CONFIG_KEYS = ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
               "intermediate_size", "vocab_size", "head_dim", "rope_theta", "rms_norm_eps", "tie_word_embeddings")

NAME_RE = re.compile(r"^model\.layers\.(\d+)\.(self_attn|mlp)\.([a-z_]+)\.weight$")


class Refused(Exception):
    """The arm is refused; nothing it wrote may be served."""


# ---------------------------------------------------------------------------
# Window logic.
# ---------------------------------------------------------------------------
def selected(name: str, window) -> bool:
    """True when `name` is one of the seven projections of a layer in `window`, inclusive at both ends, 0-based."""
    if window is None:
        return False
    m = NAME_RE.match(name)
    if not m or m.group(3) not in PROJ_TYPES:
        return False
    lo, hi = window
    layer = int(m.group(1))
    return lo <= layer and layer <= hi


def bits(t: torch.Tensor) -> torch.Tensor:
    """The bytes of a tensor, for a bitwise comparison (torch.equal reads -0 == 0 and NaN != NaN)."""
    return t.detach().contiguous().reshape(-1).view(torch.uint8)


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape) and torch.equal(bits(a), bits(b))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# The cast, and its reference.
# ---------------------------------------------------------------------------
def to_f16(t: torch.Tensor) -> torch.Tensor:
    """The restore's cast: BF16 to f16 through torch, on the CPU."""
    if t.dtype != torch.bfloat16:
        raise Refused(f"the checkpoint tensor is {t.dtype}, expected torch.bfloat16")
    return t.to(torch.float16)


def f32_to_f16_bits(x: int) -> int:
    """half 2.7.1 `f32_to_f16_fallback`, ported line by line: what candle's `f16::from_f32` returns."""
    sign = x & 0x8000_0000
    exp = x & 0x7F80_0000
    man = x & 0x007F_FFFF
    if exp == 0x7F80_0000:
        nan_bit = 0 if man == 0 else 0x0200
        return (sign >> 16) | 0x7C00 | nan_bit | (man >> 13)
    half_sign = sign >> 16
    unbiased_exp = (exp >> 23) - 127
    half_exp = unbiased_exp + 15
    if half_exp >= 0x1F:
        return half_sign | 0x7C00
    if half_exp <= 0:
        if 14 - half_exp > 24:
            return half_sign
        man = man | 0x0080_0000
        half_man = man >> (14 - half_exp)
        round_bit = 1 << (13 - half_exp)
        if (man & round_bit) != 0 and (man & (3 * round_bit - 1)) != 0:
            half_man += 1
        return half_sign | half_man
    half_exp_bits = half_exp << 10
    half_man = man >> 13
    round_bit = 0x0000_1000
    if (man & round_bit) != 0 and (man & (3 * round_bit - 1)) != 0:
        return (half_sign | half_exp_bits | half_man) + 1
    return half_sign | half_exp_bits | half_man


def cast_reference() -> torch.Tensor:
    """Lookup table: BF16 bit pattern (0..65535) to the f16 bit pattern candle would hold, as int32."""
    return torch.tensor([f32_to_f16_bits(b << 16) for b in range(1 << 16)], dtype=torch.int32)


def bf16_is_nan(b: int) -> bool:
    return (b & 0x7F80) == 0x7F80 and (b & 0x007F) != 0


def cast_matches_reference(t_bf16: torch.Tensor, t_f16: torch.Tensor, lut: torch.Tensor) -> bool:
    src = t_bf16.contiguous().reshape(-1).view(torch.int16).to(torch.int32) & 0xFFFF
    got = t_f16.contiguous().reshape(-1).view(torch.int16).to(torch.int32) & 0xFFFF
    return torch.equal(lut[src.long()], got)


def exhaustive_cast_check(lut: torch.Tensor) -> tuple[int, int]:
    """torch's BF16 -> f16 cast against the port on every BF16 bit pattern but NaN: (patterns, mismatches)."""
    pats = [b for b in range(1 << 16) if not bf16_is_nan(b)]
    src = torch.tensor([p - (1 << 16) if p >= 1 << 15 else p for p in pats], dtype=torch.int16).view(torch.bfloat16)
    got = src.to(torch.float16).view(torch.int16).to(torch.int32) & 0xFFFF
    want = lut[torch.tensor(pats, dtype=torch.long)]
    return len(pats), int((got != want).sum())


# ---------------------------------------------------------------------------
# Build and verify.
# ---------------------------------------------------------------------------
def shard_index(fp16_dir: Path, shards) -> dict[str, Path]:
    from safetensors import safe_open

    where = {}
    for s in shards:
        with safe_open(str(fp16_dir / s), framework="pt") as f:
            for k in f.keys():
                where[k] = fp16_dir / s
    return where


def build_file(export: Path, fp16_dir: Path, shards, window, out_file: Path) -> None:
    """Write `out_file`: the export's tensors, the projections of `window` replaced by the FP16 ones cast to f16."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    where = shard_index(fp16_dir, shards)
    tensors = {}
    with safe_open(str(export), framework="pt") as f:
        meta = f.metadata() or {"format": "pt"}
        for k in f.keys():
            t = f.get_tensor(k)
            if selected(k, window):
                if k not in where:
                    raise Refused(f"{k} is not in the FP16 shards")
                with safe_open(str(where[k]), framework="pt") as g:
                    src = g.get_tensor(k)
                if tuple(src.shape) != tuple(t.shape):
                    raise Refused(f"{k}: export {tuple(t.shape)}, FP16 {tuple(src.shape)}")
                if t.dtype != torch.float16:
                    raise Refused(f"{k}: the export stores {t.dtype}, expected torch.float16")
                if bool(torch.isnan(src).any()):
                    raise Refused(f"{k}: the FP16 tensor holds a NaN")
                t = to_f16(src)
            tensors[k] = t
    save_file(tensors, str(out_file), metadata=meta)


def verify_file(written: Path, export: Path, fp16_dir: Path, shards, window, expected, lut=None) -> dict:
    """Read the written file back against the export and the shards. Raises Refused on the first failure."""
    from safetensors import safe_open

    where = shard_index(fp16_dir, shards)
    n_mat = n_w = n_same = 0
    cast_checked = None
    with safe_open(str(written), framework="pt") as w, safe_open(str(export), framework="pt") as e:
        wk, ek = set(w.keys()), set(e.keys())
        if wk != ek:
            raise Refused(f"keys differ from the export: lost {sorted(ek - wk)[:4]}, added {sorted(wk - ek)[:4]}")
        for k in sorted(wk):
            got = w.get_tensor(k)
            if selected(k, window):
                with safe_open(str(where[k]), framework="pt") as g:
                    src = g.get_tensor(k)
                want = to_f16(src)
                if not same_bits(got, want):
                    raise Refused(f"{k}: the written tensor is not the FP16 tensor cast to f16")
                if same_bits(got, e.get_tensor(k)):
                    raise Refused(f"{k}: the restored tensor equals the export's, the restore is a no-op")
                if lut is not None and cast_checked is None:
                    if not cast_matches_reference(src, got, lut):
                        raise Refused(f"{k}: torch's cast differs from half's f32_to_f16_fallback")
                    cast_checked = k
                n_mat += 1
                n_w += got.numel()
            else:
                if not same_bits(got, e.get_tensor(k)):
                    raise Refused(f"{k}: an untouched tensor differs from the export's")
                n_same += 1
    if (n_mat, n_w) != tuple(expected):
        raise Refused(f"replaced {n_mat} matrices, {n_w} weights; expected {expected[0]}, {expected[1]}")
    return {"replaced_matrices": n_mat, "replaced_weights": n_w, "untouched_tensors": n_same,
            "tensors": n_mat + n_same, "cast_checked_on": cast_checked}


def build_arm(args) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import layer_vs_output as lvo  # the pins: export, checkpoint revision, shards, tokenizer

    arm, window, expected = args.arm, ARMS[args.arm], EXPECTED[args.arm]
    export, fp16_dir, out = Path(args.export), Path(args.fp16_dir), Path(args.out)
    t0 = time.time()
    print(f"arm {arm}: window {window} (inclusive, 0-based), types {', '.join(PROJ_TYPES)}, "
          f"expected {expected[0]} matrices, {expected[1]} weights")
    if out.exists():
        raise Refused(f"{out} exists")
    if export.stat().st_size != lvo.EXPORT_BYTES:
        raise Refused(f"{export} is {export.stat().st_size} B, expected {lvo.EXPORT_BYTES}")
    export_sha = sha256_file(export)
    print(f"export {export} sha256 {export_sha}")
    if export_sha != lvo.EXPORT_SHA:
        raise Refused(f"export sha256 {export_sha}, expected {lvo.EXPORT_SHA}")
    shard_sha = {}
    for s, want in lvo.CKPT_SHARDS.items():
        shard_sha[s] = sha256_file(fp16_dir / s)
        print(f"FP16 {s} sha256 {shard_sha[s]}")
        if shard_sha[s] != want:
            raise Refused(f"{s} sha256 {shard_sha[s]}, expected {want}")
    tok_sha = sha256_file(fp16_dir / "tokenizer.json")
    if tok_sha != lvo.TOKENIZER_SHA:
        raise Refused(f"tokenizer.json sha256 {tok_sha}, expected {lvo.TOKENIZER_SHA}")
    cfg = json.loads((fp16_dir / "config.json").read_text())
    exp_cfg = json.loads((export.parent / "config.json").read_text())
    for k in CONFIG_KEYS:
        if exp_cfg.get(k) != cfg.get(k):
            raise Refused(f"export config {k}={exp_cfg.get(k)} against FP16 {cfg.get(k)}")
    if cfg.get("num_hidden_layers") != N_LAYERS:
        raise Refused(f"{cfg.get('num_hidden_layers')} layers, the windows are pinned for {N_LAYERS}")
    print(f"pins: export, {len(shard_sha)} shards at {lvo.CKPT_REV}, tokenizer; config keys agree ({time.time() - t0:.0f} s)")

    out.mkdir(parents=True)
    written = out / "model.safetensors"
    t1 = time.time()
    if window is None:
        os.link(export, written)  # the export itself, no copy
    else:
        build_file(export, fp16_dir, list(lvo.CKPT_SHARDS), window, written)
    for name in SIDE_FILES:
        shutil.copyfile(fp16_dir / name, out / name)
    print(f"built {written} in {time.time() - t1:.0f} s")

    t2 = time.time()
    lut = cast_reference()
    res = verify_file(written, export, fp16_dir, list(lvo.CKPT_SHARDS), window, expected, lut)
    written_sha = sha256_file(written)
    if window is None and written_sha != lvo.EXPORT_SHA:
        raise Refused(f"control: written sha256 {written_sha} is not the export's")
    if window is not None and written_sha == lvo.EXPORT_SHA:
        raise Refused(f"{arm}: the written file is the export")
    have = sorted(p.name for p in out.iterdir())
    if have != sorted(("model.safetensors",) + SIDE_FILES):
        raise Refused(f"{out} holds {have}")
    print(f"verified in {time.time() - t2:.0f} s: {res['replaced_matrices']} matrices, {res['replaced_weights']} "
          f"weights replaced and bit-equal to FP16 cast to f16; {res['untouched_tensors']} tensors bit-equal to the "
          f"export; {res['tensors']} keys, none added or lost; cast checked against half on {res['cast_checked_on']}")
    manifest = {
        "arm": arm, "window_inclusive_0based": list(window) if window else None, "types": list(PROJ_TYPES),
        **res, "cast": "torch bf16.to(float16), CPU, round to nearest even; checked against half f32_to_f16_fallback",
        "written": str(written), "written_sha256": written_sha, "written_bytes": written.stat().st_size,
        "export": str(export), "export_sha256": export_sha, "fp16_repo": lvo.CKPT_REPO, "fp16_revision": lvo.CKPT_REV,
        "fp16_shards_sha256": shard_sha, "tokenizer_sha256": tok_sha,
        "side_files_sha256": {n: sha256_file(out / n) for n in SIDE_FILES}, "directory": have,
        "seconds": round(time.time() - t0, 1),
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"written sha256 {written_sha}; manifest {args.manifest}; {time.time() - t0:.0f} s")
    print(f"arm {arm}: GREEN")
    return 0


# ---------------------------------------------------------------------------
# Selftest: tiny synthetic safetensors, CPU.
# ---------------------------------------------------------------------------
TINY_LAYERS = 6
TINY_SHAPES = {"q_proj": (16, 8), "k_proj": (8, 8), "v_proj": (8, 8), "o_proj": (8, 16), "gate_proj": (12, 8),
               "up_proj": (12, 8), "down_proj": (8, 12)}
TINY_WINDOWS = {"late": (4, 5), "early": (0, 3)}
# 672 projection weights a tiny layer, pinned by hand.
TINY_EXPECTED = {"late": (14, 1344), "early": (28, 2688)}


def proj_name(i: int, p: str) -> str:
    return f"model.layers.{i}.{'mlp' if p in ('gate_proj', 'up_proj', 'down_proj') else 'self_attn'}.{p}.weight"


def tiny_sets(seed: int):
    g = torch.Generator().manual_seed(seed)
    fp16, export = {}, {}
    for i in range(TINY_LAYERS):
        for p, shape in TINY_SHAPES.items():
            fp16[proj_name(i, p)] = (torch.randn(shape, generator=g) * 0.1).to(torch.bfloat16)
            export[proj_name(i, p)] = (torch.randn(shape, generator=g) * 0.1).to(torch.float16)
        for n in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"):
            k = f"model.layers.{i}.{n}.weight"
            fp16[k] = (1 + torch.randn(8, generator=g) * 0.05).to(torch.bfloat16)
            export[k] = fp16[k].to(torch.float16)
    fp16["model.embed_tokens.weight"] = (torch.randn(32, 8, generator=g) * 0.1).to(torch.bfloat16)
    export["model.embed_tokens.weight"] = (torch.randn(32, 8, generator=g) * 0.1).to(torch.float16)
    fp16["model.norm.weight"] = torch.ones(8, dtype=torch.bfloat16)
    export["model.norm.weight"] = torch.ones(8, dtype=torch.float16)
    # Values the cast must round, and subnormals in f16.
    fp16[proj_name(0, "q_proj")][0, :4] = torch.tensor([1.00390625, 3e-6, -7e-8, 1234.5]).to(torch.bfloat16)
    return fp16, export


def selftest() -> int:
    from safetensors import safe_open
    from safetensors.torch import save_file

    t0 = time.time()
    n_ok = n_bad = 0

    def check(ok: bool, what: str):
        nonlocal n_ok, n_bad
        n_ok += bool(ok)
        n_bad += not ok
        print(f"  [{'ok' if ok else 'FAIL'}] {what}")

    def refused(fn) -> str | None:
        try:
            fn()
        except Refused as e:
            return str(e)
        return None

    # 1. Window logic on the real names, inclusive and 0-based.
    names = [proj_name(i, p) for i in range(N_LAYERS) for p in PROJ_TYPES]
    names += [f"model.layers.{i}.input_layernorm.weight" for i in range(N_LAYERS)] + ["model.embed_tokens.weight"]
    late = [n for n in names if selected(n, ARMS["late"])]
    early = [n for n in names if selected(n, ARMS["early"])]
    check(len(late) == EXPECTED["late"][0], f"late selects {len(late)} matrices (77)")
    check(len(early) == EXPECTED["early"][0], f"early selects {len(early)} matrices (175)")
    check(not set(late) & set(early) and len(late) + len(early) == 7 * N_LAYERS, "late and early are disjoint and cover 252")
    check(selected(proj_name(25, "q_proj"), ARMS["late"]) and selected(proj_name(35, "down_proj"), ARMS["late"]),
          "late holds layers 25 and 35")
    check(not selected(proj_name(24, "down_proj"), ARMS["late"]), "late does not hold layer 24")
    check(selected(proj_name(0, "q_proj"), ARMS["early"]) and selected(proj_name(24, "v_proj"), ARMS["early"]),
          "early holds layers 0 and 24")
    check(not selected(proj_name(25, "q_proj"), ARMS["early"]), "early does not hold layer 25")
    check(not any(selected(n, None) for n in names), "control selects nothing")
    check(not selected("model.layers.30.self_attn.q_norm.weight", ARMS["late"]), "a norm is never selected")
    real = {"q_proj": (4096, 2560), "k_proj": (1024, 2560), "v_proj": (1024, 2560), "o_proj": (2560, 4096),
            "gate_proj": (9728, 2560), "up_proj": (9728, 2560), "down_proj": (2560, 9728)}
    w_of = lambda sel: sum(real[NAME_RE.match(n).group(3)][0] * real[NAME_RE.match(n).group(3)][1] for n in sel)
    check(w_of(late) == EXPECTED["late"][1], f"late weights {w_of(late):,} at Qwen3-4B shapes (1,110,179,840)")
    check(w_of(early) == EXPECTED["early"][1], f"early weights {w_of(early):,} at Qwen3-4B shapes (2,523,136,000)")

    # 2. The cast against half's fallback, every BF16 bit pattern but NaN.
    lut = cast_reference()
    n_pat, bad = exhaustive_cast_check(lut)
    check(bad == 0 and n_pat == 65_282,
          f"torch bf16 -> f16 equals half f32_to_f16_fallback on {n_pat:,} non-NaN patterns ({bad} differ)")
    check(lut[0x3F80] == 0x3C00 and lut[0x7F80] == 0x7C00 and lut[0x4780] == 0x7C00,
          "reference: 1.0, +inf, and 65536 overflowing to +inf")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fp16, export = tiny_sets(20261114)
        fdir = td / "fp16"
        fdir.mkdir()
        keys = sorted(fp16)
        shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        save_file({k: fp16[k] for k in keys[: len(keys) // 2]}, str(fdir / shards[0]))
        save_file({k: fp16[k] for k in keys[len(keys) // 2:]}, str(fdir / shards[1]))
        exp_path = td / "export.safetensors"
        save_file(export, str(exp_path), metadata={"format": "pt"})

        # 3. Clean builds verify, and hold what they should.
        for arm in ("late", "early"):
            out = td / f"{arm}.safetensors"
            build_file(exp_path, fdir, shards, TINY_WINDOWS[arm], out)
            err = refused(lambda: verify_file(out, exp_path, fdir, shards, TINY_WINDOWS[arm], TINY_EXPECTED[arm], lut))
            check(err is None, f"{arm}: the clean build verifies{'' if err is None else ': ' + err}")
            with safe_open(str(out), framework="pt") as f:
                lo, hi = TINY_WINDOWS[arm]
                k_in, k_out = proj_name(lo, "v_proj"), proj_name(hi + 1 if hi + 1 < TINY_LAYERS else lo - 1, "v_proj")
                check(f.get_tensor(k_in).dtype == torch.float16
                      and same_bits(f.get_tensor(k_in), fp16[k_in].to(torch.float16)),
                      f"{arm}: {k_in} is the FP16 tensor cast to f16")
                check(same_bits(f.get_tensor(k_out), export[k_out]), f"{arm}: {k_out} is the export's")

        # 4. The verification catches each fault.
        def faulty(tag: str, mutate) -> Path:
            d = {}
            with safe_open(str(td / "late.safetensors"), framework="pt") as f:
                for k in f.keys():
                    d[k] = f.get_tensor(k)
            mutate(d)
            p = td / f"fault-{tag}.safetensors"
            save_file(d, str(p))
            return p

        win, exp_n = TINY_WINDOWS["late"], TINY_EXPECTED["late"]
        shifted = td / "shifted.safetensors"
        build_file(exp_path, fdir, shards, (win[0] - 1, win[1] - 1), shifted)
        cases = {
            "a wrong layer (window shifted down by one)": shifted,
            "a wrong type (v_proj of layer 4 left as the export's)":
                faulty("type", lambda d: d.__setitem__(proj_name(4, "v_proj"), export[proj_name(4, "v_proj")])),
            "an extra type (layer 4's input norm taken from FP16)":
                faulty("norm", lambda d: d.__setitem__("model.layers.4.input_layernorm.weight",
                                                       fp16["model.layers.4.input_layernorm.weight"].to(torch.float16) + 0.5)),
            "an untouched tensor altered (one weight of layer 1)":
                faulty("untouched", lambda d: d[proj_name(1, "up_proj")].view(-1).__setitem__(0, 9.0)),
            "a replaced tensor altered (one weight of layer 5)":
                faulty("replaced", lambda d: d[proj_name(5, "gate_proj")].view(-1).__setitem__(3, 9.0)),
            "a replaced tensor stored in BF16 (cast skipped)":
                faulty("bf16", lambda d: d.__setitem__(proj_name(5, "o_proj"), fp16[proj_name(5, "o_proj")])),
            "a dropped key (model.norm.weight)": faulty("dropped", lambda d: d.pop("model.norm.weight")),
            "an added key": faulty("added", lambda d: d.__setitem__("lm_head.weight", export["model.embed_tokens.weight"])),
        }
        for what, p in cases.items():
            err = refused(lambda: verify_file(p, exp_path, fdir, shards, win, exp_n, lut))
            check(err is not None, f"caught: {what}{'' if err is None else ' -> ' + err}")
        err = refused(lambda: verify_file(td / "late.safetensors", exp_path, fdir, shards, win, (exp_n[0] + 1, exp_n[1]), lut))
        check(err is not None, f"caught: a count that is not the expected one -> {err}")

        # 5. The build refuses a BF16 export projection and a checkpoint NaN.
        bad_exp = dict(export)
        bad_exp[proj_name(4, "k_proj")] = bad_exp[proj_name(4, "k_proj")].to(torch.bfloat16)
        p = td / "bad-export.safetensors"
        save_file(bad_exp, str(p))
        err = refused(lambda: build_file(p, fdir, shards, win, td / "x.safetensors"))
        check(err is not None, f"build refuses an export projection that is not f16 -> {err}")
        nan_fp = {k: fp16[k].clone() for k in keys[len(keys) // 2:]}
        target = next(k for k in nan_fp if selected(k, win))
        nan_fp[target].view(-1)[0] = float("nan")
        ndir = td / "nan"
        ndir.mkdir()
        shutil.copyfile(fdir / shards[0], ndir / shards[0])
        save_file(nan_fp, str(ndir / shards[1]))
        err = refused(lambda: build_file(exp_path, ndir, shards, win, td / "y.safetensors"))
        check(err is not None, f"build refuses a NaN in a restored FP16 tensor -> {err}")

    print(f"selftest: {n_ok} checks passed, {n_bad} failed, {time.time() - t0:.1f} s")
    print(f"selftest: {'GREEN' if n_bad == 0 else 'RED'}")
    return 0 if n_bad == 0 else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selftest", action="store_true")
    mode.add_argument("--build", action="store_true")
    ap.add_argument("--arm", choices=tuple(ARMS))
    ap.add_argument("--export", help="the staged export's model.safetensors (local disk, its config.json beside it)")
    ap.add_argument("--fp16-dir", help="the FP16 snapshot: three shards and the side files")
    ap.add_argument("--out", help="the arm's directory, must not exist")
    ap.add_argument("--manifest", help="where the JSON manifest goes")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not (a.arm and a.export and a.fp16_dir and a.out and a.manifest):
        ap.error("--build needs --arm, --export, --fp16-dir, --out and --manifest")
    try:
        return build_arm(a)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
