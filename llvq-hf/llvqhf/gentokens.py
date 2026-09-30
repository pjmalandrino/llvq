"""Gate B of stage 1: greedy tokens from a packed directory.

    uv run python -m llvqhf.gentokens <packed directory> [--new 64] [--dtype f32]
                                      [--device cpu] [--dump out.json]

The four prompts are `bin/run`'s, verbatim, and the tokenizer is the one the
sealed file carries. Ids are printed and dumped, not text: a decoded string can
hide a tokenizer difference, and the comparison is about the model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import quantizer  # noqa: F401  (registers "llvq")

PROMPTS = [
    "The capital of France is",
    "In 1969, the first humans landed on",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "Water boils at a temperature of",
]

DTYPES = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}


def generate(model, ids: list[int], n_new: int, device: str) -> list[int]:
    """Greedy, one token at a time, no cache: the reference is a tiny loop.

    `model.generate` would bring a cache, a stopping criterion and a sampler
    between the weights and the ids. This is 64 forward passes over a growing
    prompt, which is what `bin/run`'s uncached path does.
    """
    out: list[int] = []
    tokens = list(ids)
    for _ in range(n_new):
        with torch.no_grad():
            logits = model(torch.tensor([tokens], device=device)).logits
        nxt = int(logits[0, -1].argmax())
        out.append(nxt)
        tokens.append(nxt)
    return out


def report_memory(model, device: str) -> dict:
    """What the loaded model holds, measured rather than computed.

    Three numbers, because they answer different questions: the parameters torch
    knows about, the device bytes the armed projections hold, and what the MPS
    allocator has actually taken. A model whose weights are compressed on the
    device has few parameters and a large allocation, so neither number alone says
    whether the compression is real.
    """
    params = sum(p.numel() * p.element_size() for p in model.parameters())
    params += sum(b.numel() * b.element_size() for b in model.buffers())
    out = {"parameters_and_buffers": params}
    lines = [f"parameters and buffers {params / 1e9:.3f} GB"]
    resident = 0
    try:
        from .fused import FusedTetraLinear

        resident = sum(m.resident_bytes() for m in model.modules()
                       if isinstance(m, FusedTetraLinear))
    except Exception:  # noqa: BLE001  the fused arm is optional
        resident = 0
    if resident:
        out["fused_resident"] = resident
        lines.append(f"fused projections resident {resident / 1e9:.3f} GB")
    if device == "mps":
        import torch as _t

        out["mps_allocated"] = int(_t.mps.current_allocated_memory())
        out["mps_driver"] = int(_t.mps.driver_allocated_memory())
        lines.append(f"mps allocated {out['mps_allocated'] / 1e9:.3f} GB, "
                     f"driver {out['mps_driver'] / 1e9:.3f} GB")
    return {"bytes": out, "lines": lines}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="llvqhf.gentokens")
    ap.add_argument("directory")
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="f32")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dump", default=None)
    a = ap.parse_args(argv[1:])

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.directory)
    # No `device_map`: that argument pulls `accelerate` in, and this package
    # should not need it to answer a question about weights.
    model, info = AutoModelForCausalLM.from_pretrained(
        a.directory, dtype=DTYPES[a.dtype], output_loading_info=True
    )
    if a.device != "cpu":
        model = model.to(a.device)
    model.eval()
    # Part of gate B, and not a diagnostic: a missing key is a weight the model
    # invented, an unexpected one is a weight the file carried and nothing read.
    keys = {k: sorted(v) for k, v in info.items() if v}
    if keys:
        raise SystemExit(f"the load is not clean: { {k: len(v) for k, v in keys.items()} }\n"
                         + "\n".join(f"  {k}: {v[:5]}" for k, v in keys.items()))
    print(f"loaded in {time.time() - t0:.1f} s, dtype {a.dtype}, device {a.device}, "
          f"no missing and no unexpected key", flush=True)
    memory = report_memory(model, a.device)
    for line in memory["lines"]:
        print(f"  {line}", flush=True)

    result = {"directory": str(Path(a.directory)), "dtype": a.dtype, "device": a.device,
              "n_new": a.new, "loading_info": {k: len(v) for k, v in info.items()},
              "memory": memory["bytes"], "prompts": []}
    for p in PROMPTS:
        ids = tok(p, add_special_tokens=False)["input_ids"]
        t = time.time()
        out = generate(model, ids, a.new, a.device)
        result["prompts"].append({"prompt": p, "input_ids": ids, "output_ids": out,
                                  "text": tok.decode(out)})
        print(f"── {p!r}\n   →{tok.decode(out)}\n   {a.new} tokens in {time.time() - t:.1f} s",
              flush=True)
    if a.dump:
        Path(a.dump).write_text(json.dumps(result, indent=2))
        print(f"dumped {a.dump}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
