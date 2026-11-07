"""The MLX door's served config: read it, refuse what was never measured, by name.

`configs/mlx/*.json` names what the MLX engine runs, the way `configs/*.json`
names what the Rust door runs. It is a separate schema in a separate directory:
`llvq-llm/tests/served_config.rs` parses every file of `configs/` with
`deny_unknown_fields`, and the MLX keys are not the Rust keys.

Every value has exactly one accepted spelling, the one M16 timed and both
perplexity readings scored (mlx-t8 and mlx-pooled, 2026-11-07). Anything else is refused with the key, the value and the reason,
never replaced by a default. A Rust-door key (`fuse`, `decode`, `rows`) is
refused too: the MLX door reads none of them, and silence would let a reader
believe it did.

The environment: the MLX door reads no `LLVQ_*` variable, so any one set is
refused by name. `LLVQ_KV=q8` beside this door would otherwise run an f16 cache
while the shell says int8. `LLVQ_MLX_ABLATE` is the prototype's measurement mode
(`llvq_mlx.py`), and it is refused for the same reason.

No MLX import here: the refusals are tested without a GPU (`test_served.py`).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

# The objects the MLX door may serve, by artifact sha256. One so far: the sealed
# 4B. The 8B and the 14B have never run on MLX.
OBJECTS = {
    "886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8": "qwen3-4b-sealed.bin",
}

# key -> the one accepted value. What M16 timed on the M3 Max.
ACCEPTED = {
    "engine": "mlx",
    "layout": "tetra48",
    "embed": "q4",
    "rot_share": "1",
    "kv": "f16",
    "activations": "f16",
    "attention": "mlx_sdpa",
}

# Why a value other than the accepted one is refused. Read in the refusal.
WHY = {
    "engine": "this door runs MLX only; the Rust engine reads configs/*.json",
    "layout": "the MLX path carries the tetra48 kernels only",
    "embed": "the sealed file stores the table int4 g64 and the MLX path reads it as stored",
    "rot_share": "the MLX path rotates once for q and k, once for gate and up; no other sharing was measured",
    "kv": "mlx-lm's quantized KV cache was never measured on this object",
    "activations": "the MLX kernels take and store f16; nothing else was measured",
    "attention": "the MLX path runs mlx-lm's scaled_dot_product_attention; the Rust forms do not exist here",
}

PINNED = ("mlx", "mlx_lm")  # versions: the installed ones must be these
REQUIRED = ("engine", "object", "layout", "embed", "rot_share", "kv", "activations", "attention", *PINNED)
OPTIONAL = ("note",)
RUST_KEYS = ("fuse", "decode", "rows")


class Refused(ValueError):
    """A config, an environment or an object the MLX door will not serve."""


@dataclass(frozen=True)
class MlxServed:
    path: Path
    sha256: str
    object: str
    mlx: str
    mlx_lm: str
    note: str | None

    def line(self) -> str:
        """The `served config:` line every run prints first."""
        return (f"served config: {self.path} sha256 {self.sha256[:16]} (engine mlx, object "
                f"{OBJECTS[self.object]} {self.object[:8]}, layout tetra48, embed q4, rot_share 1, kv f16, "
                f"activations f16, attention mlx_sdpa, mlx {self.mlx}, mlx-lm {self.mlx_lm})")


def parse(raw: dict, path: Path, sha256: str) -> MlxServed:
    if not isinstance(raw, dict):
        raise Refused(f"{path}: a JSON object expected")
    for k in raw:
        if k in RUST_KEYS:
            raise Refused(f"{path}: key {k!r} belongs to the Rust door (configs/*.json); the MLX door reads none")
        if k not in REQUIRED and k not in OPTIONAL:
            raise Refused(f"{path}: unknown key {k!r}")
    for k in REQUIRED:
        if k not in raw:
            raise Refused(f"{path}: key {k!r} is required and has no default")
        if not isinstance(raw[k], str):
            raise Refused(f"{path}: key {k!r} must be a string, as in configs/*.json")
    for k, want in ACCEPTED.items():
        if raw[k] != want:
            raise Refused(f"{path}: {k} {raw[k]!r} refused: {WHY[k]} (accepted: {want!r})")
    if raw["object"] not in OBJECTS:
        raise Refused(f"{path}: object {raw['object'][:16]}... refused: the MLX door serves "
                      f"{', '.join(OBJECTS.values())} only; no other object has run on MLX")
    return MlxServed(path, sha256, raw["object"], raw["mlx"], raw["mlx_lm"], raw.get("note"))


def load(path: str | Path) -> MlxServed:
    p = Path(path)
    data = p.read_bytes()
    try:
        raw = json.loads(data)
    except json.JSONDecodeError as e:
        raise Refused(f"{p}: not JSON ({e})") from e
    return parse(raw, p, hashlib.sha256(data).hexdigest())


def check_env(environ) -> None:
    """Refuse every LLVQ_* variable: the MLX door reads none of them."""
    set_ = sorted(k for k in environ if k.startswith("LLVQ_"))
    if set_:
        raise Refused(f"{', '.join(set_)} set: the MLX door reads no LLVQ_* variable, so it would be "
                      f"ignored while the shell says otherwise; unset it (LLVQ_MLX_ABLATE is the prototype's "
                      f"measurement mode, llvq_mlx.py)")


def check_object(cfg: MlxServed, packed_config: dict) -> None:
    """The packed directory must hold the config's object, by the sealed file's sha256."""
    q = packed_config.get("quantization_config", {})
    got = q.get("artifact_sha256")
    if got != cfg.object:
        raise Refused(f"the packed directory holds artifact {str(got)[:16]}..., the config serves "
                      f"{cfg.object[:16]}... ({OBJECTS[cfg.object]})")


def check_runtime(cfg: MlxServed, mlx_version: str, mlx_lm_version: str, device: str) -> None:
    if mlx_version != cfg.mlx:
        raise Refused(f"mlx {mlx_version} installed, the config serves mlx {cfg.mlx}: another version "
                      f"compiles other kernels and was not measured")
    if mlx_lm_version != cfg.mlx_lm:
        raise Refused(f"mlx-lm {mlx_lm_version} installed, the config serves mlx-lm {cfg.mlx_lm}")
    if "gpu" not in device:
        raise Refused(f"default device {device}: the Tetra kernels run on the Metal GPU only")
