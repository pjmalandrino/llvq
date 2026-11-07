# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx-lm==0.28.3", "mlx==0.32.4", "numpy==2.5.4", "safetensors==0.8.0", "tokenizers==0.23.3",
#                 "pyarrow==26.0.1", "transformers==5.22.1", "llvq-tetra"]
# [tool.uv.sources]
# llvq-tetra = { path = "../../llvq-tetra" }
# ///
"""The MLX door: the sealed Qwen3-4B on the Mac's GPU, under a served config.

    uv run ops/mlx_tetra/serve.py <config> <packed_dir> chat [temp=0.7] [max_new=512]
    uv run ops/mlx_tetra/serve.py <config> <packed_dir> generate "<message>" [max_new=256]
    uv run ops/mlx_tetra/serve.py <config> <packed_dir> time <n_new> <timed_rounds> [prompt_tokens=0]
    uv run ops/mlx_tetra/serve.py <config> <packed_dir> ppl [ctx=4096] [windows=12] [paths=decode,prefill] [first=0 ref]
    uv run ops/mlx_tetra/serve.py <config> <packed_dir> span

<config> is `configs/mlx/qwen3-4b-tetra-e4-mlx.json`; <packed_dir> the hfpack of
the sealed file (`bin/hfpack`). `ops/llvq-mlx` wraps the first three.

Before the model loads, in this order: no `LLVQ_*` variable is set, the config
parses with every value the one accepted (`served.py`), the packed directory
holds the config's object, and the installed mlx, mlx-lm and device are the
config's. Each refusal names what it refused. Then the first line printed is
the config's, with its sha256.

`chat`: a conversation, Qwen3's format with reasoning off, the cache kept
across turns. `generate`: one such turn, greedy, its text and token digest.
`time`: `metalrun time`'s protocol at this door (2 discarded rounds, then the
timed ones, medians with their ranges). `ppl`: the perplexity readings of `ppl.py`.
`span`: the span gate of that reading alone, without loading the model.

Activations are f16 and the path is not bit-exact with the Rust kernels. Its
gate is the paired perplexity interval of
proofs/preregistration-mlx-pooled-2026-11-07.md, inside +-0.12 % of the Rust
path; the per-window band of the first reading (mlx-t8-2026-11-07) was missed.
"""
from __future__ import annotations

import json
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path

import served as S


def open_door(cfg_path: str, packed: str, load_model: bool = True):
    # The environment first: llvq_mlx reads LLVQ_MLX_ABLATE when it is imported.
    S.check_env(os.environ)
    cfg = S.load(cfg_path)
    cfg_file = Path(packed) / "config.json"
    if not cfg_file.is_file():
        raise S.Refused(f"no packed model at {packed} (no config.json); write it with bin/hfpack")
    S.check_object(cfg, json.loads(cfg_file.read_text()))
    import mlx.core as mx

    S.check_runtime(cfg, version("mlx"), version("mlx-lm"), str(mx.default_device()))
    import llvq_mlx as L

    if L.ABLATE:  # unreachable past check_env; a second lock on the measurement mode
        raise S.Refused(f"ablation {sorted(L.ABLATE)} active")
    print(cfg.line())
    if cfg.note:
        print(f"note: {cfg.note}")
    print(f"packed {packed} | {mx.default_device()}")
    if not load_model:
        return cfg, L, None
    t = time.perf_counter()
    model, _, tetra_bytes = L.build(packed)
    print(f"built in {time.perf_counter() - t:.1f} s | Tetra resident {tetra_bytes / 1e9:.3f} GB | "
          f"active {mx.get_active_memory() / 1e9:.2f} GB")
    return cfg, L, model


def generate(L, model, packed: str, message: str, max_new: int) -> None:
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    fmt = L.ChatFormat(packed)
    ids = fmt.turn(message)
    out, t1 = [], None
    for (tid, _), _ in zip(generate_step(mx.array(ids), model, max_tokens=max_new), range(max_new)):
        tid = int(tid)
        if t1 is None:
            t1 = time.perf_counter()
        if tid in (fmt.im_end, fmt.eot):
            break
        out.append(tid)
    t2 = time.perf_counter()
    print(fmt.tok.decode(out))
    rate = (len(out) - 1) / (t2 - t1) if len(out) > 1 else 0.0
    print(f"--- {len(out)} tokens, sha256 {L.sha_u32(out)}; prompt {len(ids)} tokens sha256 {L.sha_u32(ids)}; "
          f"{rate:.1f} tok/s, one round, not a measurement")


def main(argv: list[str]) -> int:
    usage = "\n".join(line.strip() for line in __doc__.strip().splitlines()[2:7])
    if len(argv) < 4:
        print(usage, file=sys.stderr)
        return 2
    cfg_path, packed, mode, rest = argv[1], argv[2], argv[3], argv[4:]
    if mode not in ("chat", "generate", "time", "ppl", "span"):
        print(f"mode {mode!r} refused\n{usage}", file=sys.stderr)
        return 2
    # Every argument is read before the model loads: a typo costs nothing.
    try:
        if mode == "chat":
            job = ("chat", float(rest[0]) if rest else 0.7, int(rest[1]) if len(rest) > 1 else 512)
        elif mode == "generate":
            if not rest:
                raise ValueError("generate needs a message")
            job = ("generate", rest[0], int(rest[1]) if len(rest) > 1 else 256)
        elif mode == "time":
            if len(rest) < 2:
                raise ValueError("time needs <n_new> <timed_rounds>")
            job = ("time", int(rest[0]), int(rest[1]), int(rest[2]) if len(rest) > 2 else 0)
        else:
            ctx = int(rest[0]) if rest else 4096
            windows = int(rest[1]) if len(rest) > 1 else 12
            paths = rest[2].split(",") if len(rest) > 2 else ["decode", "prefill"]
            bad = [p for p in paths if p not in ("decode", "prefill")]
            if bad or not paths:
                raise ValueError(f"paths {bad} refused: decode, prefill")
            first = int(rest[3]) if len(rest) > 3 else 0
            ref = rest[4] if len(rest) > 4 else None
            if mode == "ppl":
                import ppl  # no MLX import at module level: safe before the door

                ppl.check_args(ctx, windows, first, ref)
            job = (mode, ctx, windows, paths, first, ref)
    except ValueError as e:
        print(f"{e}\n{usage}", file=sys.stderr)
        return 2
    try:
        cfg, L, model = open_door(cfg_path, packed, load_model=mode != "span")
    except S.Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 1
    if mode == "chat":
        L.chat(model, packed, job[1], job[2])
    elif mode == "generate":
        generate(L, model, packed, job[1], job[2])
    elif mode == "time":
        L.time_rounds(model, L.prompt_ids(packed, job[3]), job[1], job[2], "llvq-mlx served")
    else:
        import ppl

        _, ctx, windows, paths, first, ref = job
        if mode == "span":
            _, nwin, digest, n_split = ppl.span(packed, ctx, windows)
            ppl.gate_span(digest, n_split, ctx, nwin)
            return 0
        ppl.run(model, packed, ctx, windows, paths, first, ref)
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
