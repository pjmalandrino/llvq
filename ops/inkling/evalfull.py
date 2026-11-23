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
"""KL of a whole-model student against the cached teacher logits (D/eval/teacher/), any text tensor substituted.

    uv run evalfull.py "<student>" [--name N] [--device mps] [--windows 8] [--ctx 2048]
                       [--perturb L:E:REL[:SEED]]
    uv run evalfull.py dump "<pattern>" <dir> [--perturb-rel REL]   # write native tensors as raw f32 (dir: sources)

A student is the mapping of `model_full.parse_student`: `experts:<layers>=<source>` and
`<parameter pattern>=<source>` items, `;`-separated, or a JSON file. Unnamed tensors stay native.

Measured exactly as `evalkl.py` (whose `windows`, `boot`, `nll_of` and teacher directory are imported): per window,
over the ctx-1 positions that predict a next token, mean KL(teacher || student), top-1 agreement, mean NLL of the true
next token for teacher and student; bootstrap over windows. The windows are the first `--windows` of the 8 the teacher
cache holds, and the token fingerprint of the 8 is checked against `teacher.json`. The teacher logits were computed on
MPS; on MPS this executor reproduces them exactly, on CPU the floor is KL 1.9e-4 (last night's `arm_native_*`).

`--perturb L:E:REL` adds seeded Gaussian noise of relative size REL (times the expert's own standard deviation) to
expert E of layer L, gate_up and down: the control that shows the measurement is not blind.
Output: D/evalfull/<name>.json and one summary line.
"""

import _threads  # noqa: F401

import argparse
import fnmatch
import json
import os
import re
import sys
import time

import numpy as np

from _threads import DATA

import evalkl  # noqa: E402  (imports executor: torch threads, MoE patch)
from evalkl import TDIR, boot, windows  # noqa: E402
from executor import Teacher, torch  # noqa: E402
import model_full  # noqa: E402

OUT = os.path.join(DATA, "evalfull")


class PerturbProvider:
    """A `Provider` whose layer `layer` has expert `expert` perturbed by seeded Gaussian noise."""

    def __init__(self, base, layer, expert, rel, seed=0):
        self.base, self.layer, self.expert, self.rel, self.seed = base, layer, expert, rel, seed
        self.applied = 0

    def get(self, layer):
        w13, w2 = self.base.get(layer)
        if layer != self.layer:
            return w13, w2
        g = torch.Generator().manual_seed(self.seed)
        out = []
        for w in (w13, w2):
            w = w.clone()
            we = w[self.expert].cpu()
            noise = torch.randn(we.shape, generator=g) * (self.rel * float(we.std()))
            w[self.expert] = (we + noise).to(w.device)
            out.append(w)
        self.applied += 1
        return out[0], out[1]


def score_window(logits, i, ids, ctx, V):
    """evalkl.py's per-window measure, on the student's logits of window i."""
    la = torch.log_softmax(logits.float().cpu()[:-1], -1)
    lt = torch.log_softmax(torch.from_numpy(np.fromfile(os.path.join(TDIR, f"w{i}.f32"), dtype=np.float32)
                                            .reshape(ctx, V))[:-1], -1)
    kl = (lt.exp() * (lt - la)).sum(-1)
    tgt = ids[i, 1:, None]
    return {"window": i, "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
            "top1_agree": float((la.argmax(-1) == lt.argmax(-1)).float().mean()),
            "nll_teacher": float(-lt.gather(1, tgt).mean()), "nll_arm": float(-la.gather(1, tgt).mean())}


def run_rows(student_spec, device="mps", win_idx=(0,), ctx=2048, perturb=None, model=None, restore=False):
    """Per-window rows of a student on the teacher windows `win_idx`. Returns (rows {i: row}, substitution report,
    token fingerprint of the 8 windows, model). With `restore`, the model's substituted tensors are put back."""
    meta = json.load(open(os.path.join(TDIR, "teacher.json")))
    ids_all, fp, _ = windows(ctx, meta["windows"])
    if meta["tokens_fingerprint"] != fp or meta["ctx"] != ctx:
        sys.exit("teacher cache does not match these windows; rerun `evalkl.py teacher`")
    win_idx = list(win_idx)
    if max(win_idx) >= meta["windows"]:
        sys.exit(f"the teacher cache holds {meta['windows']} windows")
    V = meta["vocab"]
    student = model_full.parse_student(student_spec)
    arm = model_full.expert_arm(student)
    teacher = Teacher(device=device, arm=arm, model=model)
    teacher.model.requires_grad_(False)
    teacher.provider = model_full.FastProvider(arm, device)
    plan = model_full.tensor_plan(student, model_full.tensor_names(teacher.model))
    snap = model_full.snapshot(teacher.model, plan) if restore else None
    subst = model_full.apply_tensor_plan(teacher.model, plan, log=None)
    if perturb:
        parts = perturb.split(":")
        L, E, rel = int(parts[0]), int(parts[1]), float(parts[2])
        seed = int(parts[3]) if len(parts) > 3 else 0
        teacher.provider = PerturbProvider(teacher.provider, L, E, rel, seed)
    rows = {}
    ids = ids_all[win_idx]
    try:
        teacher.run(ids, on_logits=lambda j, lg: rows.__setitem__(win_idx[j],
                                                                  score_window(lg, win_idx[j], ids_all, ctx, V)),
                    log=None)
    finally:
        if snap is not None:
            model_full.restore(teacher.model, snap)
    if perturb:
        assert teacher.provider.applied > 0, "perturbed layer never loaded"
    return rows, subst, fp, teacher.model


def aggregate(R, boot_n=10000):
    """evalkl.py's summary over a list of window rows, plus the perplexity ratio exp(mean NLL arm - mean NLL teacher)."""
    kl = [r["kl_mean"] for r in R]
    d = [r["nll_arm"] - r["nll_teacher"] for r in R]
    nt = float(np.mean([r["nll_teacher"] for r in R]))
    na = float(np.mean([r["nll_arm"] for r in R]))
    return {"windows_used": [r["window"] for r in R], "kl_mean": float(np.mean(kl)), "kl_ci95": boot(kl, boot_n),
            "kl_max": float(max(r["kl_max"] for r in R)),
            "top1_agree": float(np.mean([r["top1_agree"] for r in R])), "nll_teacher": nt, "nll_arm": na,
            "nll_diff_mean": float(np.mean(d)), "nll_diff_ci95": boot(d, boot_n),
            "ppl_teacher": float(np.exp(nt)), "ppl_arm": float(np.exp(na)), "ppl_ratio": float(np.exp(na - nt)),
            "bootstrap": f"{boot_n} resamples of windows, seed 0, percentile"}


def evaluate(student_spec, device="mps", nwin=8, ctx=2048, perturb=None, name=None, boot_n=10000, model=None,
             restore=False):
    t0 = time.time()
    rows, subst, fp, model = run_rows(student_spec, device, range(nwin), ctx, perturb, model, restore)
    R = [rows[i] for i in range(nwin)]
    name = name or re.sub(r"[^A-Za-z0-9_.+-]+", "_", student_spec or "native")[:120]
    res = {"student": name, "spec": student_spec,
           "substituted_tensors": {k: v for k, v in list(subst.items())[:50]}, "n_substituted": len(subst),
           "perturb": perturb, "windows": nwin, "ctx": ctx, "tokens_fingerprint_of_8": fp,
           "positions_per_window": ctx - 1, **aggregate(R, boot_n), "per_window": R, "device": device,
           "sec": time.time() - t0}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, f"{name}.json"), "w"), indent=2)
    print(f"STUDENT {name}  KL {res['kl_mean']:.6g} (max {res['kl_max']:.3g})  top1 {100 * res['top1_agree']:.2f}%  "
          f"dNLL {res['nll_diff_mean']:+.6f}  ppl x{res['ppl_ratio']:.5f}  ({res['sec']:.0f}s)", flush=True)
    return res, model


def dump(pattern, d, perturb_rel=0.0, device="cpu"):
    """Write the native tensors a pattern names as raw f32 `<d>/<name>.f32` (optionally perturbed, seed 0)."""
    import executor

    model = executor.load_model(device)
    os.makedirs(d, exist_ok=True)
    tensors = dict(model.named_parameters()) | dict(model.named_buffers())
    names = [n for n in tensors if fnmatch.fnmatchcase(n, pattern)]
    if not names:
        sys.exit(f"{pattern!r} names nothing")
    g = torch.Generator().manual_seed(0)
    for n in names:
        t = tensors[n].detach().float().cpu()
        if perturb_rel:
            t = t + torch.randn(t.shape, generator=g) * (perturb_rel * float(t.std()))
        t.numpy().tofile(os.path.join(d, n + ".f32"))
    print(json.dumps({"dumped": names, "dir": d, "perturb_rel": perturb_rel}))


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "dump":
        ap = argparse.ArgumentParser()
        ap.add_argument("mode")
        ap.add_argument("pattern")
        ap.add_argument("dir")
        ap.add_argument("--perturb-rel", type=float, default=0.0)
        a = ap.parse_args()
        dump(a.pattern, a.dir, a.perturb_rel)
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("student", nargs="?", default="native")
    ap.add_argument("--name", default=None)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--perturb", default=None)
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    evaluate(a.student, a.device, a.windows, a.ctx, a.perturb, a.name, a.boot)


if __name__ == "__main__":
    main()
