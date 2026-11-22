# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "numpy",
#   "safetensors",
#   "tokenizers",
#   "pyarrow",
#   "transformers @ git+https://github.com/huggingface/transformers@046bdb8cc2",
# ]
# ///
"""Real-model gate of the sliced capture (heavy: one model load, about 5 minutes on MPS).

    uv run gate_slices.py [--layers 0,6,11] [--device mps] [--keep]

The input of the 1a gate (calibration windows 0 and 1, 256 tokens). One run of the two sequences in one group,
kept in memory; then two slices of one sequence each through `attn_slices.run_slices`, folded and merged into
D/attn/slicegate/. Every merged file must equal the single run bit for bit, and `capture_attn.check` must pass on the
merged manifest. The merged files are deleted afterwards unless `--keep`. Result: D/evalfull/gate_slices.json.
"""

import _threads  # noqa: F401

import argparse
import json
import os
import shutil
import time
from types import SimpleNamespace

import numpy as np

from _threads import DATA

import attn_slices as S  # noqa: E402
import capture_attn  # noqa: E402
from executor import Teacher, torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,6,11")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    layers = [int(x) for x in a.layers.split(",")]
    out = os.path.join(DATA, "attn", "slicegate")
    if os.path.exists(out):
        shutil.rmtree(out)
    t0 = time.time()
    windows, _, fp = S.corpus(2, 256)
    settings = S.settings_of(SimpleNamespace(n=2, len=256, device=a.device, chunk=8192), fp, layers)
    teacher = Teacher(device=a.device)
    teacher.model.requires_grad_(False)
    acc, _ = capture_attn.capture(teacher, torch.tensor(windows), layers, group=2, micro=1, chunk=8192, log=None)
    ref = {S.stem(L, s) + ".f64": g.ravel() for (L, s), g in acc.gram.items()}
    t_single = time.time() - t0
    S.run_slices(teacher, out, [(0, 1), (1, 2)], windows, settings, log=lambda m: print(m, flush=True))
    S.merge(out, log=lambda m: print(m, flush=True))
    rows = {}
    for fn, r in sorted(ref.items()):
        m = np.fromfile(os.path.join(out, fn), np.float64)
        rows[fn] = {"bit_equal": bool(np.array_equal(m, r)),
                    "max_abs_diff": float(np.abs(m - r).max()), "absmax": float(np.abs(r).max())}
    ok_check, rep = capture_attn.check(out)
    res = {"gate": "sliced capture == single run", "device": a.device, "layers": layers,
           "corpus_fingerprint": fp, "files": rows, "all_bit_equal": all(r["bit_equal"] for r in rows.values()),
           "check_on_merged": ok_check, "pre_loop_equal": rep["pre_loop_equal"],
           "loop_pass2_differ": [{k: r[k] for k in ("layer", "ok")} for r in rep["loop_pass2_differ"]],
           "tokens_x_visits_ok": all(r["ok"] for r in rep["tokens_x_visits"]),
           "sec_single": t_single, "sec_total": time.time() - t0}
    os.makedirs(os.path.join(DATA, "evalfull"), exist_ok=True)
    json.dump(res, open(os.path.join(DATA, "evalfull", "gate_slices.json"), "w"), indent=1)
    if not a.keep:
        for fn in ref:
            os.unlink(os.path.join(out, fn))
    print(json.dumps({k: res[k] for k in ("all_bit_equal", "check_on_merged", "tokens_x_visits_ok",
                                          "loop_pass2_differ", "sec_total")}), flush=True)


if __name__ == "__main__":
    main()
