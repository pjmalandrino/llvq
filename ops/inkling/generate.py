# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "numpy",
#   "safetensors",
#   "tokenizers",
#   "pyarrow",
#   "transformers @ git+https://github.com/huggingface/transformers@046bdb8cc2",
#   "gguf @ git+https://github.com/ggml-org/llama.cpp@690b87d881#subdirectory=gguf-py",
# ]
# ///
"""Greedy generation with Inkling 2 Mini's own per-pass caches, any student, experts loaded lazily.

    uv run generate.py gate [--prompt 32] [--new 32] [--device mps] [--student "<spec>"] [--name N]

Generation runs the library's `Inkling2LoopForCausalLM.forward` with `use_cache=True`: the model builds its
`DynamicCache` and appends one copy of the loop layers' cache per extra pass (`modeling_inkling2_loop.py` 1354-1357),
so each of the 48 layer visits keeps its own keys, values and convolution states. Experts come from
`model_full.ExpertSource`: whole layers for the prompt, single routed experts with an LRU for each decoded token.

`gate`: the prompt is the first `--prompt` tokens of WikiText-2 test window 0 (`data.eval_windows`). `--new` tokens
are generated greedily through the caches; then one forward without cache over the prompt and the first `--new - 1`
generated tokens gives, by causality, the argmax a full forward picks at each of the `--new` positions. The gate passes
when the two token lists are equal. The logits of both paths are compared too (max abs difference), as information.
Output: D/evalfull/generate_<name>.json.
"""

import _threads  # noqa: F401

import argparse
import json
import os
import sys
import time

from _threads import DATA, NATIVE

import executor  # noqa: E402
from executor import torch  # noqa: E402
import model_full  # noqa: E402

OUT = os.path.join(DATA, "evalfull")


def build(student_spec="native", device="mps", cache_gb=12.0):
    """The reference model with a student's tensors, every layer's experts lazy."""
    model = executor.load_model(device)
    model.requires_grad_(False)
    student = model_full.parse_student(student_spec)
    plan = model_full.tensor_plan(student, model_full.tensor_names(model))
    subst = model_full.apply_tensor_plan(model, plan)
    source = model_full.ExpertSource(model_full.expert_arm(student), device=device, cache_gb=cache_gb)
    model_full.install_lazy(model, source)
    return model, source, subst


@torch.inference_mode()
def greedy(model, prompt_ids, max_new, eos=None, use_cache=True):
    """Greedy decode. Returns (generated token list, logits of each step [max_new, V] on the CPU)."""
    dev = next(model.parameters()).device
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=dev)
    out = model(input_ids=ids, use_cache=use_cache)
    cache = out.cache
    step_logits = [out.logits[0, -1].float().cpu()]
    gen = [int(step_logits[-1].argmax())]
    while len(gen) < max_new and (eos is None or gen[-1] != eos):
        nxt = torch.tensor([[gen[-1]]], dtype=torch.long, device=dev)
        out = model(input_ids=nxt, cache=cache, use_cache=use_cache)
        cache = out.cache
        step_logits.append(out.logits[0, -1].float().cpu())
        gen.append(int(step_logits[-1].argmax()))
    return gen, torch.stack(step_logits)


def gate(a):
    from tokenizers import Tokenizer

    import data

    tok = Tokenizer.from_file(os.path.join(NATIVE, "tokenizer.json"))
    wins, _ = data.eval_windows(tok, 1, 2048)
    prompt = wins[0][: a.prompt]
    t0 = time.time()
    model, source, subst = build(a.student, a.device, a.cache_gb)
    tl = time.time() - t0
    t1 = time.time()
    gen, step_logits = greedy(model, prompt, a.new)
    tg = time.time() - t1
    n_cache_layers = None
    t2 = time.time()
    with torch.inference_mode():
        seq = torch.tensor([prompt + gen[:-1]], dtype=torch.long, device=next(model.parameters()).device)
        full = model(input_ids=seq, use_cache=False).logits[0].float().cpu()
    tf = time.time() - t2
    full_steps = full[len(prompt) - 1 :]
    assert full_steps.shape[0] == len(gen), (full_steps.shape, len(gen))
    full_argmax = [int(x) for x in full_steps.argmax(-1)]
    same = [g == f for g, f in zip(gen, full_argmax)]
    d = (step_logits - full_steps).abs()
    res = {
        "gate": "1d generation: cached greedy == full-forward argmax",
        "student": a.student, "device": a.device, "prompt_tokens": len(prompt), "new_tokens": len(gen),
        "generated": gen, "full_forward_argmax": full_argmax, "all_equal": all(same),
        "first_mismatch": same.index(False) if not all(same) else None,
        "max_abs_logit_diff": float(d.max()), "mean_abs_logit_diff": float(d.mean()),
        "per_step_max_abs_logit_diff": [float(x) for x in d.max(-1).values],
        "text_prompt": tok.decode(prompt), "text_generated": tok.decode(gen),
        "expert_source_stats": source.stats, "substituted_tensors": subst,
        "sec_load": tl, "sec_generate": tg, "sec_full_forward": tf, "cache_layers": n_cache_layers,
    }
    os.makedirs(OUT, exist_ok=True)
    name = a.name or "native"
    json.dump(res, open(os.path.join(OUT, f"generate_{name}.json"), "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("all_equal", "first_mismatch", "max_abs_logit_diff", "new_tokens",
                                          "sec_generate", "sec_full_forward", "expert_source_stats")}), flush=True)
    print("generated:", repr(res["text_generated"]), flush=True)
    return res["all_equal"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gate"])
    ap.add_argument("--prompt", type=int, default=32)
    ap.add_argument("--new", type=int, default=32)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--student", default="native")
    ap.add_argument("--name", default=None)
    ap.add_argument("--cache-gb", type=float, default=12.0)
    a = ap.parse_args()
    sys.exit(0 if gate(a) else 1)


if __name__ == "__main__":
    main()
