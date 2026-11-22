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
"""Per-layer encoder inputs: native weights and routing-weighted Hessians of every expert.

    uv run prep_layer.py <layer> [--calib D/calib] [--out D/layers] [--device cpu|mps] [--chunk 8192]

Writes `<out>/L{layer:02d}/`:
  w13.f32  [128, 2304, 2304]  native gate_up_proj (rows 0..1151 gate, 1152..2303 up), [out, in]
  w2.f32   [128, 2304, 1152]  native down_proj, [out, in]
  h13.f64  [128, 2304, 2304]  h13[e] = sum over the layer's visits and the tokens routed to e of w^2 x x^T
  h2.f64   [128, 1152, 1152]  same with a = softcapped SwiGLU(w13[e] x) (the input of w2) in place of x
  manifest.json               shapes, per-expert token counts, sum w and sum w^2 per visit
All raw little-endian, row-major. Sums unnormalised. x is the captured float16 MoE input, widened to float32.
Accumulation: each Gram block is a float32 GEMM over at most `--chunk` tokens of one expert, and the blocks are summed
in float64 (a full float64 GEMM would cost about 40 TFLOP a loop layer on CPU).
The activation uses the library's `Inkling2LoopSoftcappedSwiGLU` with the model's config.
"""

import _threads  # noqa: F401

import argparse
import json
import os
import time

import numpy as np

from _threads import DATA, HIDDEN, INTER, NUM_EXPERTS, TEXTONLY, TOPK, apply_torch

torch = apply_torch()
from transformers.models.inkling2_loop import modeling_inkling2_loop as M  # noqa: E402
from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig  # noqa: E402

import nvfp4  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layer", type=int)
    ap.add_argument("--calib", default=os.path.join(DATA, "calib"))
    ap.add_argument("--out", default=os.path.join(DATA, "layers"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--chunk", type=int, default=8192)
    a = ap.parse_args()
    L, dev = a.layer, a.device
    t0 = time.time()

    man = json.load(open(os.path.join(a.calib, "manifest.json")))
    n = man["n_tokens"]
    visits = [v for v in man["visits"] if v["layer"] == L]
    assert visits, f"no visit of layer {L}"
    od = os.path.join(a.out, f"L{L:02d}")
    os.makedirs(od, exist_ok=True)

    w13, w2 = nvfp4.native_experts(L)
    w13.numpy().tofile(os.path.join(od, "w13.f32"))
    w2.numpy().tofile(os.path.join(od, "w2.f32"))
    del w2
    act = M.Inkling2LoopSoftcappedSwiGLU(Inkling2LoopTextConfig.from_pretrained(TEXTONLY)).to(dev)

    # all visits of this layer, concatenated (x as float16 to bound memory: 0.6 GB a visit)
    xs, idxs, ws = [], [], []
    for v in visits:
        d = os.path.join(a.calib, v["dir"])
        xs.append(np.fromfile(os.path.join(d, "x.f16"), dtype=np.float16).reshape(n, HIDDEN))
        idxs.append(np.fromfile(os.path.join(d, "topk_idx.u8"), dtype=np.uint8).reshape(n, TOPK))
        ws.append(np.fromfile(os.path.join(d, "topk_w.f32"), dtype=np.float32).reshape(n, TOPK))
    nv = len(visits)
    counts = np.zeros((nv, NUM_EXPERTS), np.int64)
    wsum = np.zeros((nv, NUM_EXPERTS), np.float64)
    w2sum = np.zeros((nv, NUM_EXPERTS), np.float64)
    # token lists per (visit, expert)
    per = [[None] * NUM_EXPERTS for _ in range(nv)]
    for vi in range(nv):
        flat_e = idxs[vi].ravel().astype(np.int64)
        order = np.argsort(flat_e, kind="stable")
        bounds = np.searchsorted(flat_e[order], np.arange(NUM_EXPERTS + 1))
        tok_of = order // TOPK
        wflat = ws[vi].ravel()
        for e in range(NUM_EXPERTS):
            sel = order[bounds[e] : bounds[e + 1]]
            per[vi][e] = (tok_of[bounds[e] : bounds[e + 1]], wflat[sel])
            counts[vi, e] = sel.size
            wsum[vi, e] = float(wflat[sel].astype(np.float64).sum())
            w2sum[vi, e] = float((wflat[sel].astype(np.float64) ** 2).sum())

    f13 = open(os.path.join(od, "h13.f64"), "wb")
    f2 = open(os.path.join(od, "h2.f64"), "wb")
    for e in range(NUM_EXPERTS):
        g13 = np.zeros((HIDDEN, HIDDEN), np.float64)
        g2 = np.zeros((INTER, INTER), np.float64)
        we = w13[e].to(dev)
        for vi in range(nv):
            toks, wt = per[vi][e]
            for s in range(0, toks.size, a.chunk):
                t = toks[s : s + a.chunk]
                x = torch.from_numpy(xs[vi][t].astype(np.float32)).to(dev)
                w = torch.from_numpy(wt[s : s + a.chunk].astype(np.float32)).to(dev)[:, None]
                xw = x * w
                g13 += (xw.T @ xw).double().cpu().numpy()
                gu = torch.nn.functional.linear(x, we)
                gate, up = gu.chunk(2, dim=-1)
                aw = act(gate, up) * w
                g2 += (aw.T @ aw).double().cpu().numpy()
        f13.write(g13.tobytes())
        f2.write(g2.tobytes())
        if e % 16 == 15:
            print(f"  L{L:02d} expert {e + 1}/128  {time.time() - t0:.0f}s", flush=True)
    f13.close()
    f2.close()

    json.dump(
        {
            "layer": L,
            "visits": visits,
            "n_tokens_per_visit": n,
            "files": {"w13.f32": [NUM_EXPERTS, 2 * INTER, HIDDEN], "w2.f32": [NUM_EXPERTS, HIDDEN, INTER],
                      "h13.f64": [NUM_EXPERTS, HIDDEN, HIDDEN], "h2.f64": [NUM_EXPERTS, INTER, INTER]},
            "orientation": "w: [expert, out, in]; w13 rows 0..1151 gate, 1152..2303 up; h: [expert, in, in]",
            "hessian": "unnormalised sum over visits and routed tokens of w^2 v v^T; v = x (h13) or "
                       "softcapped SwiGLU(w13[e] x) (h2); float32 Gram blocks of <= chunk tokens summed in float64",
            "chunk": a.chunk, "device": dev,
            "token_counts": counts.tolist(),
            "token_counts_total": counts.sum(0).tolist(),
            "weight_sums": wsum.tolist(),
            "weight_sq_sums": w2sum.tolist(),
            # What `extq` reads: one entry an expert, summed over the layer's visits. `weight_sum` is the sum
            # of the weights the Hessian carries, which are the squared routing weights.
            "num_experts": NUM_EXPERTS,
            "experts": [
                {"tokens": int(t), "weight_sum": float(s)}
                for t, s in zip(counts.sum(0).tolist(), w2sum.sum(0).tolist())
            ],
            "sec": time.time() - t0,
        },
        open(os.path.join(od, "manifest.json"), "w"),
        indent=1,
    )
    print(json.dumps({"layer": L, "out": od, "sec": time.time() - t0}))


if __name__ == "__main__":
    main()
