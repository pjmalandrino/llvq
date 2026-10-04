"""Talk to a Tetra model, which is the test a benchmark cannot do.

    python -m llvq_tetra.chat                       # the published 4B
    python -m llvq_tetra.chat <dir or repo>         # another one
    python -m llvq_tetra.chat --device mps --dtype f16
    python -m llvq_tetra.chat --ask "Explain a hash map to a beginner"

An MMLU score says the model gets 63.37 % of multiple-choice questions right. It
does not say whether the thing is pleasant, whether it repeats itself, whether it
collapses after two turns, or whether 2.73 bits per parameter broke something a
score averages away. Reading its answers does.

Defaults are chosen so that this is usable rather than impressive. **f16 dense on
the accelerator**: the weights are decoded at load into ordinary tensors, which is
8 GB and plain matmuls. That is the fast arm, and the one whose weights are proven
identical to the Rust decoder field by field.

`LLVQ_HF_FUSED=1` keeps the weights compressed instead, 2.75 GB on Apple silicon,
and is far slower here: the kernel takes one activation vector at a time, so every
token costs one dispatch per projection. Fine for checking it works, painful for a
conversation.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from . import quantizer  # noqa: F401  registers the method

DEFAULT = "Pier-Jean/Qwen3-4B-LLVQ-Tetra"
DTYPES = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="llvq_tetra.chat")
    ap.add_argument("model", nargs="?", default=DEFAULT)
    ap.add_argument("--device", default=None, help="default: cuda, else mps, else cpu")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="f16")
    ap.add_argument("--new", type=int, default=512, help="cap on generated tokens")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--ask", action="append", default=None,
                    help="one turn and exit; repeat for several, each on a fresh history")
    # Qwen3 reasons out loud by default, and on a short budget the monologue is
    # all you get: 220 tokens of "let me think about this" and no answer. Off
    # here, because the question this file answers is whether the thing is
    # usable, and a user reads answers.
    ap.add_argument("--think", action="store_true",
                    help="let Qwen3 reason out loud first; it eats the token budget")
    a = ap.parse_args(argv[1:])
    device = a.device or pick_device()

    print(f"loading {a.model}\n  {a.dtype} on {device}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=DTYPES[a.dtype])
    model = model.to(device).eval()
    held = sum(p.numel() * p.element_size() for p in model.parameters())
    held += sum(b.numel() * b.element_size() for b in model.buffers())
    print(f"  ready in {time.time() - t0:.1f} s, holding {held / 1e9:.2f} GB", flush=True)

    if tok.chat_template is None:
        print("REFUSED: this tokenizer has no chat template, so there is no "
              "conversation to have. Check tokenizer_config.json", file=sys.stderr)
        return 1

    def answer(history: list[dict]) -> str:
        kw = {} if a.think else {"enable_thinking": False}
        text = tok.apply_chat_template(history, tokenize=False,
                                       add_generation_prompt=True, **kw)
        ids = tok(text, return_tensors="pt").to(device)
        streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        t = time.time()
        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=a.new, streamer=streamer,
                do_sample=a.temperature > 0, temperature=a.temperature or None,
                top_p=0.8 if a.temperature > 0 else None,
                pad_token_id=tok.eos_token_id,
            )
        n = out.shape[-1] - ids["input_ids"].shape[-1]
        dt = time.time() - t
        # One number, and it is this machine under its own load, not a benchmark.
        print(f"\n  [{n} tokens, {dt:.1f} s, {n / dt:.1f} tok/s]", flush=True)
        return tok.decode(out[0][ids["input_ids"].shape[-1]:], skip_special_tokens=True)

    if a.ask:
        for q in a.ask:
            print(f"\n> {q}")
            answer([{"role": "user", "content": q}])
        return 0

    print("\nType a message. Ctrl-C or an empty line to stop. "
          "`/reset` forgets the conversation.")
    history: list[dict] = []
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            return 0
        if q == "/reset":
            history = []
            print("  forgotten")
            continue
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer(history)})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
