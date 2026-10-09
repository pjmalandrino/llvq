#!/usr/bin/env python3
"""One round of the speed measurement of the Mac block, one arm, in its own process.

    python ops/hf_speed_mac.py <arm> <model> <device_map> <dtype> <out.jsonl>

Section 6 of `proofs/preregistration-hf-metal-bloc-2026-10-09.md`. Two numbers per arm:

* prefill: the time to the first token of a 1,024-token prompt;
* decode: tokens per second over 128 greedy tokens after a 16-token prompt, taken as
  `127 / (t(128 new) - t(1 new))`, so the prompt's own prefill drops out.

One warm-up generation first, so kernel compilation and caches are not in either number. Every
timing is bracketed by `torch.mps.synchronize()`. The prompt is a fixed text, tokenized and tiled to
length, the same ids for every arm. No number from this file is divided across stacks or cards.
"""

from __future__ import annotations

import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TEXT = (
    "The Leech lattice is a lattice in twenty-four dimensions, the densest packing of spheres there. "
    "Each point has 196,560 nearest neighbours, and its automorphism group is the Conway group Co0. "
)


def sync() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def timed(model, ids, new: int) -> float:
    sync()
    t = time.perf_counter()
    with torch.no_grad():
        model.generate(ids, max_new_tokens=new, min_new_tokens=new, do_sample=False)
    sync()
    return time.perf_counter() - t


def main(argv: list[str]) -> int:
    arm, name, dmap, dtype, out = argv[1:6]
    import importlib

    if arm != "fp16":
        importlib.import_module("llvq_tetra")
    dtype_t = {"float32": torch.float32, "float16": torch.float16}[dtype]
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(name if arm != "fp16" else "Qwen/Qwen3-4B")
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype_t, device_map=dmap)
    sync()
    load = time.perf_counter() - t0
    dev = next(model.parameters()).device
    base = tok(TEXT * 200, return_tensors="pt")["input_ids"][0]
    long_ids = base[:1024].unsqueeze(0).to(dev)
    short_ids = base[:16].unsqueeze(0).to(dev)

    timed(model, short_ids, 4)  # warm-up
    prefill = timed(model, long_ids, 1)
    t1 = timed(model, short_ids, 1)
    t128 = timed(model, short_ids, 128)
    row = {
        "arm": arm,
        "model": name,
        "dtype": dtype,
        "load_s": round(load, 3),
        "prefill_1024_s": round(prefill, 4),
        "decode_tok_s": round(127 / (t128 - t1), 3),
        "t1_s": round(t1, 4),
        "t128_s": round(t128, 4),
        "mps_allocated_gb": round(torch.mps.current_allocated_memory() / 1e9, 3) if dev.type == "mps" else None,
        "torch": torch.__version__,
    }
    with open(out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
