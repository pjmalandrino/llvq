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
"""KL of an arm against the teacher on raw WikiText-2 test, 8 windows of 2,048 tokens from token 0.

    uv run evalkl.py teacher                         # computes and caches the teacher logits (D/eval/teacher/)
    uv run evalkl.py arm "<spec>" [--name NAME]       # e.g. "6-8=dir:/Users/x/llvq-inkling-mini/hat/tetra"
                                                     #      "6-17=gguf:/path/Inkling-2-Mini-UD-IQ2_XXS.gguf"
The `gguf` git pin in the header is the one ggufx.py uses; only `gguf:` arms import it.
Options: --device cpu|mps, --windows 8, --ctx 2048, --boot 10000.

Per window, over the ctx-1 positions that predict a next token: mean KL(teacher || arm) of the next-token
distributions, top-1 agreement (argmax equal), mean NLL of the true next token for teacher and arm. Bootstrap
(seed 0) over windows, percentile 2.5/97.5, for the mean KL and the paired NLL difference (arm - teacher).
Output: D/eval/arm_<name>.json and one summary line on stdout. Teacher logits are float32, softcapped, as the
model outputs them, one file per window (D/eval/teacher/w{i}.f32, [ctx, 200058]).
"""

import _threads  # noqa: F401

import argparse
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

from _threads import DATA, NATIVE

import executor  # noqa: E402
from executor import Teacher, arm_name, parse_arm, torch  # noqa: E402

EVAL = os.path.join(DATA, "eval")
TDIR = os.path.join(EVAL, "teacher")


def windows(ctx, nwin):
    from tokenizers import Tokenizer

    import data

    tok = Tokenizer.from_file(os.path.join(NATIVE, "tokenizer.json"))
    wins, stats = data.eval_windows(tok, nwin, ctx)
    ids = torch.tensor(wins, dtype=torch.long)
    fp = hashlib.sha256(np.asarray(wins, dtype=np.uint32).tobytes()).hexdigest()[:16]
    return ids, fp, stats


def nll_of(logp, ids):
    return float(-logp[:-1].gather(1, ids[1:, None]).mean())


def boot(x, B, seed=0):
    x = np.asarray(x, np.float64)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(B, len(x)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["teacher", "arm"])
    ap.add_argument("spec", nargs="?", default="native")
    ap.add_argument("--name", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    ids, fp, stats = windows(a.ctx, a.windows)
    t0 = time.time()

    if a.mode == "teacher":
        os.makedirs(TDIR, exist_ok=True)
        t = Teacher(device=a.device)
        nll = {}

        def keep(i, logits):
            lg = logits.float().cpu()
            lg.numpy().tofile(os.path.join(TDIR, f"w{i}.f32"))
            nll[i] = nll_of(torch.log_softmax(lg, -1), ids[i])

        t.run(ids, on_logits=keep)
        meta = {"windows": a.windows, "ctx": a.ctx, "tokens_fingerprint": fp, "corpus": stats,
                "vocab": int(t.model.lm_head.out_features), "nll_per_window": [nll[i] for i in range(a.windows)],
                "ppl_window0": float(np.exp(nll[0])), "ppl_all": float(np.exp(np.mean(list(nll.values())))),
                "device": a.device, "sec": time.time() - t0}
        json.dump(meta, open(os.path.join(TDIR, "teacher.json"), "w"), indent=2)
        print(json.dumps({k: v for k, v in meta.items() if k != "corpus"}))
        return

    meta = json.load(open(os.path.join(TDIR, "teacher.json")))
    if meta["tokens_fingerprint"] != fp or meta["ctx"] != a.ctx or meta["windows"] != a.windows:
        sys.exit("teacher cache does not match these windows; rerun `evalkl.py teacher`")
    V = meta["vocab"]
    arm = parse_arm(a.spec)
    name = a.name or re.sub(r"[^A-Za-z0-9_.+-]+", "_", arm_name(arm))[:120]
    t = Teacher(device=a.device, arm=arm)
    rows = {}

    def score(i, logits):
        la = torch.log_softmax(logits.float().cpu()[:-1], -1)
        lt = torch.log_softmax(torch.from_numpy(np.fromfile(os.path.join(TDIR, f"w{i}.f32"), dtype=np.float32)
                                                .reshape(a.ctx, V))[:-1], -1)
        kl = (lt.exp() * (lt - la)).sum(-1)
        tgt = ids[i, 1:, None]
        rows[i] = {"window": i, "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
                   "top1_agree": float((la.argmax(-1) == lt.argmax(-1)).float().mean()),
                   "nll_teacher": float(-lt.gather(1, tgt).mean()), "nll_arm": float(-la.gather(1, tgt).mean())}

    t.run(ids, on_logits=score)
    R = [rows[i] for i in range(a.windows)]
    kl = [r["kl_mean"] for r in R]
    d = [r["nll_arm"] - r["nll_teacher"] for r in R]
    res = {
        "arm": name, "spec": a.spec, "layers": {str(k): v for k, v in sorted(arm.items())},
        "windows": a.windows, "ctx": a.ctx, "tokens_fingerprint": fp, "positions_per_window": a.ctx - 1,
        "kl_mean": float(np.mean(kl)), "kl_ci95": boot(kl, a.boot),
        "top1_agree": float(np.mean([r["top1_agree"] for r in R])),
        "nll_teacher": float(np.mean([r["nll_teacher"] for r in R])),
        "nll_arm": float(np.mean([r["nll_arm"] for r in R])),
        "nll_diff_mean": float(np.mean(d)), "nll_diff_ci95": boot(d, a.boot),
        "per_window": R, "device": a.device, "sec": time.time() - t0,
        "bootstrap": f"{a.boot} resamples of windows, seed 0, percentile",
    }
    json.dump(res, open(os.path.join(EVAL, f"arm_{name}.json"), "w"), indent=2)
    print(f"ARM {name}  KL {res['kl_mean']:.5f} [{res['kl_ci95'][0]:.5f}, {res['kl_ci95'][1]:.5f}]  "
          f"top1 {100 * res['top1_agree']:.2f}%  dNLL {res['nll_diff_mean']:+.5f} "
          f"[{res['nll_diff_ci95'][0]:+.5f}, {res['nll_diff_ci95'][1]:+.5f}]  ({res['sec']:.0f}s)")


if __name__ == "__main__":
    main()
