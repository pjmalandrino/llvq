# Runs inside the vLLM job image, with gptqmodel installed at job time.
"""The test that tells "collapsed" from "broken", and nothing else.

The GPTQ 2-bit arm returns **24.74% MMLU**, which is four-choice chance, with a
degenerate output (68.9% of "A", logit spread divided by 3.4). Two hypotheses
produce exactly that shape:

  (a) bare 2-bit GPTQ REALLY collapses, the premise of the whole field;
  (b) the LOADING is faulty, and the number measures nothing.

Free generation settles it: fluent text says (a), garbled text says (b).

## The control is free because the f16 answer is already published

Since 2026-08-17, `ops/awq_speed.py` pins the prompt and the continuation that
the f16 arm of the 4B produces:

    "The capital of France is" -> " Paris. (True or False?)\\nThe statement …"

So we do not need to load a second model to have a control.
"""

from __future__ import annotations

import sys

PROMPT = "The capital of France is"
REF_F16 = (
    ' Paris. (True or False?)\n'
    'The statement "The capital of France is Paris" is **False**.'
)


def main(argv: list[str]) -> int:
    path = argv[0] if argv else "/out/qwen3-4b-gptq2"
    import torch
    from gptqmodel import GPTQModel
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    wrapper = GPTQModel.load(path, device="cuda:0")
    ids = tok(PROMPT, return_tensors="pt").to("cuda:0")
    n = ids["input_ids"].shape[1]

    # ALERT: BOTH PATHS, IN A SINGLE PROCESS.
    #
    # The first pass generated garbled text by calling `.generate()` on
    # `GPTQModel.load(...).model`, the UNWRAPPED HF module. If the wrapper is
    # what wires the quantized kernels, that unwrapping short-circuits them and
    # the model runs on weights that are not the ones we think.
    #
    # Garbled text DOES NOT TELL "collapsed" from "badly loaded": a really
    # destroyed model produces the same. What tells them apart is running BOTH
    # paths side by side:
    #   fluent wrapper + garbled .model  -> our scoring path is at fault, and
    #                                       the 24.74% MMLU is void;
    #   both garbled                     -> the two paths agree, the loss is
    #                                       real.
    outs = {}
    for name, obj in (("wrapper GPTQModel", wrapper), (".model (unwrapped)", wrapper.model)):
        try:
            with torch.inference_mode():
                o = obj.generate(**ids, max_new_tokens=60, do_sample=False)
            outs[name] = tok.decode(o[0][n:], skip_special_tokens=True)
        except Exception as e:
            outs[name] = f"<<FAILED: {type(e).__name__}: {e}>>"

    print("=" * 78)
    print(f"prompt              {PROMPT!r}")
    print(f"f16 (published)     {REF_F16!r}")
    for name, text in outs.items():
        print(f"{name:19s} {text!r}")
    print("=" * 78)
    print("\nReading, set BEFORE looking:")
    print("  fluent wrapper + garbled .model -> our scorer is at fault, MMLU void")
    print("  both garbled                    -> REAL loss, the two paths agree")
    print("  wrapper fails                   -> inconclusive, to be investigated")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
