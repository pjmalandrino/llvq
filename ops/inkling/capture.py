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
"""Teacher capture over the calibration set: MoE inputs and routing of all 48 layer visits.

    uv run capture.py [--n 64] [--len 2048] [--group 32] [--micro 1] [--device cpu] [--out D/calib]
    uv run capture.py --probe 2 [--len 2048]      # speed probe only: no dump

Writes, per visit, `<out>/v{visit:02d}_L{layer:02d}_p{pass}/`:
  x.f16          [n_tokens, 2304] float16, the tensor given to `self.mlp(...)` (flattened, as the experts see it)
  topk_idx.u8    [n_tokens, 8] uint8, the router's experts
  topk_w.f32     [n_tokens, 8] float32, the routing weights as passed to the experts (normalised, times 8.0)
pass = 0 for the visits of layers 0-5 and 18-23, 1..3 for the three passes of layers 6-17. Tokens are in sequence
order then position order, identical across visits. Also `tokens.u32` [n, len], `manifest.json`, `census.json`.
Corpus rule (see data.py): dclm-edu shard 000_00000, rows in order, "\\n\\n"-joined, budget n*len*6 bytes, no BOS,
contiguous windows from token 0.
"""

import _threads  # noqa: F401

import argparse
import json
import os
import sys
import time

import numpy as np

from _threads import DATA, NATIVE, NUM_EXPERTS, HIDDEN, TOPK

import executor  # noqa: E402  (applies torch threads, patches the MoE block)
from executor import Teacher, torch  # noqa: E402


def visit_dir(vi, L, p):
    return f"v{vi:02d}_L{L:02d}_p{p}"


def census(counts, visits, width=HIDDEN):
    """counts [n_visits, 128]: tokens routed to each (visit, expert) cell."""
    per_visit = []
    for vi, (L, p) in enumerate(visits):
        c = counts[vi]
        per_visit.append({"visit": vi, "layer": L, "pass": p, "cells_under_width": int((c < width).sum()),
                          "share_under_width": float((c < width).mean()), "min": int(c.min()),
                          "median": float(np.median(c)), "max": int(c.max())})
    per_layer = []
    for L in sorted({L for L, _ in visits}):
        c = sum(counts[vi] for vi, (LL, _) in enumerate(visits) if LL == L)
        per_layer.append({"layer": L, "visits": sum(1 for LL, _ in visits if LL == L),
                          "share_under_width": float((c < width).mean()), "min": int(c.min()),
                          "median": float(np.median(c))})
    loop = [r for r in per_layer if r["visits"] > 1]
    return {
        "width": width,
        "counts": counts.tolist(),
        "per_visit": per_visit,
        "per_layer_summed_over_passes": per_layer,
        "share_cells_under_width_one_visit_mean": float(np.mean([r["share_under_width"] for r in per_visit])),
        "share_cells_under_width_loop_layers_summed": float(np.mean([r["share_under_width"] for r in loop])) if loop else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--len", type=int, default=2048)
    ap.add_argument("--group", type=int, default=32, help="sequences resident at once (about 190 MB each at 2048)")
    ap.add_argument("--micro", type=int, default=1, help="sequences per layer call")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(DATA, "calib"))
    ap.add_argument("--probe", type=int, default=0, help="speed probe on this many sequences, no dump")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    from tokenizers import Tokenizer

    import data

    tok = Tokenizer.from_file(os.path.join(NATIVE, "tokenizer.json"))
    n = a.probe or a.n
    windows, stats = data.calib_windows(tok, n, a.len)
    ids = torch.tensor(windows, dtype=torch.long)
    teacher = Teacher(device=a.device)
    visits = teacher.visits

    if a.probe:
        t0 = time.time()
        teacher.run(ids, micro=a.micro, on_logits=lambda i, l: None)
        dt = time.time() - t0
        tps = n * a.len / dt
        res = {"probe_seqs": n, "len": a.len, "device": a.device, "threads": _threads.THREADS, "sec": dt,
               "tok_per_s": tps, "projected_capture_h_131072": 131072 / tps / 3600,
               "projected_eval_arm_min_16384": 16384 / tps / 60}
        print(json.dumps(res), flush=True)
        os.makedirs(os.path.join(DATA, "eval"), exist_ok=True)
        json.dump(res, open(os.path.join(DATA, "eval", f"probe_{a.device}_{n}x{a.len}.json"), "w"), indent=2)
        return

    out = a.out
    if os.path.exists(os.path.join(out, "manifest.json")) and not a.force:
        sys.exit(f"{out}/manifest.json exists; --force to overwrite")
    os.makedirs(out, exist_ok=True)
    np.asarray(windows, dtype=np.uint32).tofile(os.path.join(out, "tokens.u32"))
    files = {}
    for vi, (L, p) in enumerate(visits):
        d = os.path.join(out, visit_dir(vi, L, p))
        os.makedirs(d, exist_ok=True)
        files[vi] = tuple(open(os.path.join(d, f), "wb") for f in ("x.f16", "topk_idx.u8", "topk_w.f32"))
    counts = np.zeros((len(visits), NUM_EXPERTS), dtype=np.int64)
    ntok = np.zeros(len(visits), dtype=np.int64)

    def recorder(vi, L, p, ch, x, idx, w):
        fx, fi, fw = files[vi]
        fx.write(x.detach().to(torch.float16).cpu().numpy().tobytes())
        ii = idx.detach().cpu().numpy()
        assert ii.shape[1] == TOPK and ii.max() < 256
        fi.write(ii.astype(np.uint8).tobytes())
        fw.write(w.detach().to(torch.float32).cpu().numpy().tobytes())
        counts[vi] += np.bincount(ii.ravel(), minlength=NUM_EXPERTS)
        ntok[vi] += x.shape[0]

    t0 = time.time()
    groups = []
    for g0 in range(0, n, a.group):
        tg = time.time()
        g = list(range(g0, min(g0 + a.group, n)))
        print(f"group {g0 // a.group}: sequences {g[0]}..{g[-1]}", flush=True)
        teacher.run(ids[g], micro=a.micro, recorder=recorder, on_logits=lambda i, l: None)
        for fs in files.values():
            for f in fs:
                f.flush()
        groups.append({"sequences": [g[0], g[-1]], "sec": time.time() - tg})
        json.dump({"progress_groups_done": groups}, open(os.path.join(out, "progress.json"), "w"))
    for fs in files.values():
        for f in fs:
            f.close()
    dt = time.time() - t0
    assert (ntok == n * a.len).all(), ntok

    import transformers

    manifest = {
        "model": "thinkingmachines/Inkling-2-Mini@ed40e7da625f594fc9c4990f5f66be24873c22d7",
        "transformers": transformers.__version__ + " (git 046bdb8cc2)",
        "arm": "native (float32 dequantized experts, unrounded)",
        "dtype": "float32", "device": a.device, "threads": _threads.THREADS,
        "corpus": "HuggingFaceTB/dclm-edu data/000_00000.parquet, rows in file order joined by blank lines, "
                  "budget n*len*6 UTF-8 bytes, tokenized once with add_special_tokens=False (no BOS), "
                  "contiguous windows from token 0 (llvq-llm corpus.rs rule, seed None)",
        "corpus_stats": stats, "n_seqs": n, "seq_len": a.len, "n_tokens": int(n * a.len),
        "token_order": "sequence-major, then position", "group": a.group, "micro": a.micro,
        "files": {"x.f16": [int(n * a.len), HIDDEN], "topk_idx.u8": [int(n * a.len), TOPK],
                  "topk_w.f32": [int(n * a.len), TOPK]},
        "pass_convention": "0 = layers outside the loop; 1..3 = passes over layers 6-17",
        "visits": [{"visit": vi, "layer": L, "pass": p, "dir": visit_dir(vi, L, p)} for vi, (L, p) in enumerate(visits)],
        "sec": dt, "tok_per_s": n * a.len / dt, "groups": groups,
    }
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=2)
    json.dump(census(counts, visits), open(os.path.join(out, "census.json"), "w"), indent=1)
    print(json.dumps({"done": out, "sec": dt, "tok_per_s": n * a.len / dt}))


if __name__ == "__main__":
    main()
