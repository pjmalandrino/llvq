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
"""Acceptance gates of lot 1, items 1a and 1d (i), on the real model, one model load (heavy: run under heavy.sh).

    uv run gates_a.py [--device mps] [--layers 0,4,5,6,11] [--perturb 0:3:0.5]

1a: `capture_attn` on 2 calibration sequences of 256 tokens, the given layers, into D/attn/gate_2x256/, then
`capture_attn.check` (q = k = v bit for bit on the sliding pre-loop layers, three distinct streams in pass 2 of the loop
layers, tokens = tokens x visits).
1d (i): `evalfull.evaluate` on WikiText-2 window 0 (2,048 tokens), native student: KL must be exactly 0; then one
perturbed expert: KL must be positive.
Writes D/evalfull/gates_a.json.
"""

import _threads  # noqa: F401

import argparse
import json
import os
import time

from _threads import DATA, NATIVE

import capture_attn  # noqa: E402
import evalfull  # noqa: E402
from executor import Teacher, torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--layers", default="0,4,5,6,11")
    ap.add_argument("--perturb", default="0:3:0.5")
    ap.add_argument("--out", default=os.path.join(DATA, "attn", "gate_2x256"))
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    from tokenizers import Tokenizer

    import data

    t0 = time.time()
    res = {"device": a.device}
    tok = Tokenizer.from_file(os.path.join(NATIVE, "tokenizer.json"))
    windows, stats = data.calib_windows(tok, 2, 256)
    ids = torch.tensor(windows, dtype=torch.long)
    teacher = Teacher(device=a.device)
    model = teacher.model
    model.requires_grad_(False)
    layers = [int(x) for x in a.layers.split(",")]
    acc, groups = capture_attn.capture(teacher, ids, layers, group=2, micro=1)
    capture_attn.write(a.out, acc, model.config, 2, 256, {"corpus_stats": stats, "device": a.device,
                                                          "groups": groups, "purpose": "acceptance gate 1a"})
    ok, rep = capture_attn.check(a.out)
    json.dump(rep, open(os.path.join(a.out, "check.json"), "w"), indent=1)
    res["1a"] = {"out": a.out, "ok": ok, "sec": time.time() - t0,
                 "reader_checks": sum(g["reader_checks"] for g in groups),
                 "pre_loop_equal": rep["pre_loop_equal"],
                 "loop_pass2_differ": [{k: r[k] for k in ("layer", "summed_hessians_differ", "ok")}
                                       for r in rep["loop_pass2_differ"]],
                 "tokens_x_visits_ok": all(r["ok"] for r in rep["tokens_x_visits"])}
    print(json.dumps({"1a": res["1a"]}), flush=True)

    r0, _ = evalfull.evaluate("native", a.device, 1, 2048, None, "gate_native_w0", 1000, model=model)
    r1, _ = evalfull.evaluate("native", a.device, 1, 2048, a.perturb, "gate_perturb_w0", 1000, model=model)
    res["1d_kl"] = {"native_kl": r0["kl_mean"], "native_kl_max": r0["kl_max"], "native_top1": r0["top1_agree"],
                    "native_dnll": r0["nll_diff_mean"], "perturb": a.perturb, "perturb_kl": r1["kl_mean"],
                    "perturb_top1": r1["top1_agree"], "perturb_dnll": r1["nll_diff_mean"],
                    "ok": r0["kl_mean"] == 0.0 and r0["kl_max"] == 0.0 and r1["kl_mean"] > 0.0}
    res["sec"] = time.time() - t0
    os.makedirs(evalfull.OUT, exist_ok=True)
    json.dump(res, open(os.path.join(evalfull.OUT, "gates_a.json"), "w"), indent=2)
    print(json.dumps({"1d_kl": res["1d_kl"], "sec": res["sec"]}), flush=True)


if __name__ == "__main__":
    main()
