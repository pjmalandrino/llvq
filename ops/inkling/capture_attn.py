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
"""Attention calibration inputs of Inkling 2 Mini: four Hessians a layer, summed over the layer's visits.

    uv run capture_attn.py [--n 64] [--len 2048] [--layers all] [--group 8] [--device mps] [--out D/attn/<name>]
    uv run capture_attn.py check <out>          # the acceptance checks, on a finished capture

Which matrix reads which stream. Read from transformers 046bdb8cc2 `modeling_inkling2_loop.py`, and checked on every
call of a capture: each `nn.Linear` inside `self_attn` must receive, bit for bit, the tensor of the stream this table
gives it, or the capture stops (`READERS`, `_check_reader`).

| stream    | width | tensor (lines)                                                    | read by (native name -> transformers name)          |
|-----------|-------|-------------------------------------------------------------------|-----------------------------------------------------|
| query     | 2,304 | `hidden_states` argument of `self_attn` (603, 618, 619, 635)      | `wq_du` -> `q_proj.linear`; `wr_du` ->               |
|           |       |                                                                   | `relative_position_bias.linear`; `attn_sink_proj` -> |
|           |       |                                                                   | `sink_proj`; `gate_attn` -> `o_proj.gate_proj`       |
| key       | 2,304 | `key_hidden_states` (604-606; equal to query when not passed, 599) | `wk_dv` -> `k_proj.linear`; full layers also         |
|           |       |                                                                   | `k_sconv.weight_proj` -> `k_proj.sconv.weight_proj`  |
| value     | 2,304 | `value_hidden_states` (607-609; equal to query when not passed)    | `wv_dv` -> `v_proj.linear`; full layers also         |
|           |       |                                                                   | `v_sconv.weight_proj` -> `v_proj.sconv.weight_proj`  |
| gated_out | 3,072 | `attn_output * sigmoid(gate_proj(query))` (540-543)               | `wo_ud` -> `o_proj.linear`                           |

Where the three input streams come from, by layer kind (the code, not the plan, is the authority):
- sliding layers outside the loop (0-4, 18-22): one tensor `input_layernorm(attention_input(...))` passed as
  `hidden_states` only (891-898), so query, key and value are the same tensor;
- sliding loop layers (6-10, 12-16), pass 1: one tensor for all three (961-962); passes 2 and 3: three tensors, each
  through its own message gate then the shared `input_layernorm` (964-967);
- full-attention layers (5, 11, 17, 23): three tensors in every pass, from the three queries of `attention_input`
  (`num_queries=3`, 1006, 1037-1039; layer 0 would have none, but layer 0 is sliding) and three different RMSNorms
  `q/k/v_input_layernorm` (1042-1044); in passes 2 and 3 of layers 11 and 17 the message gates come on top (1115-1119).
  So layer 5, although it precedes the loop, has three different input Hessians; the bit-equality check holds for the
  sliding layers 0-4 only.
The static convolution weights `k_sconv.weight` / `v_sconv.weight` (sliding) and `static_w` (full) act after the
projection, on 1,024 channels, and need no input Hessian; nor do the clip scales and `rel_logits_proj.proj`.

Output, `<out>/`: `L{layer:02d}_{stream}.f64`, the unnormalised sum over the layer's visits and tokens of x x^T,
[width, width] float64, row-major. Each Gram block is a float32 GEMM over at most `--chunk` tokens on the run device,
summed in float64 on the CPU, as `prep_layer.py` does. `manifest.json`: shapes, token counts per (layer, stream) and per
visit, and per (visit, stream) checksums: float64 sum and sum of squares of x, and the sha256 of x's float32 bytes.

Corpus: `data.calib_windows` (dclm-edu, the rule of `capture.py`), first `--n` windows of `--len` tokens. The schedule
is `executor.Teacher.run`, layer-major, native experts, float32.
"""

import _threads  # noqa: F401

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

from _threads import DATA, NATIVE

STREAMS = ("query", "key", "value", "gated_out")
# transformers module path inside `self_attn` -> stream it must read (see the table above)
READERS = {
    "q_proj.linear": "query",
    "relative_position_bias.linear": "query",
    "sink_proj": "query",
    "o_proj.gate_proj": "query",
    "k_proj.linear": "key",
    "k_proj.sconv.weight_proj": "key",
    "v_proj.linear": "value",
    "v_proj.sconv.weight_proj": "value",
    "o_proj.linear": "gated_out",
}


def expected_visits(cfg):
    s, e, n = cfg["loop_layer_index_start"], cfg["loop_layer_index_end"], cfg["loop_num_passes"]
    return {L: (n if s <= L < e else 1) for L in range(cfg["num_hidden_layers"])}


class Accumulator:
    """Per (layer, stream): float64 Gram sum and token count; per (visit, stream): checksums."""

    def __init__(self, layers, chunk=8192, sha=True):
        self.layers = set(layers)
        self.chunk = chunk
        self.sha = sha
        self.gram = {}
        self.tokens = {}
        self.per_visit = {}

    def add(self, layer, pas, stream, x):
        """`x` [..., width] float32, on any device."""
        if layer not in self.layers:
            return
        import torch

        x = x.reshape(-1, x.shape[-1])
        if x.dtype != torch.float32:
            raise TypeError(f"L{layer} {stream}: {x.dtype}, float32 expected")
        key = (layer, stream)
        g = self.gram.get(key)
        if g is None:
            g = self.gram[key] = np.zeros((x.shape[1], x.shape[1]), np.float64)
        for s in range(0, x.shape[0], self.chunk):
            xc = x[s : s + self.chunk]
            g += (xc.T @ xc).cpu().double().numpy()
        self.tokens[key] = self.tokens.get(key, 0) + int(x.shape[0])
        vk = (layer, pas, stream)
        row = self.per_visit.setdefault(vk, {"tokens": 0, "sum": 0.0, "sumsq": 0.0, "h": hashlib.sha256()})
        xd = x.double() if x.device.type == "cpu" else x.cpu().double()
        row["tokens"] += int(x.shape[0])
        row["sum"] += float(xd.sum())
        row["sumsq"] += float((xd * xd).sum())
        if self.sha:
            row["h"].update(x.detach().cpu().contiguous().numpy().tobytes())


class Hooks:
    """Hooks on the decoder layers, `self_attn` and its linears. A visit's pass comes from the per-layer call count
    and the number of micro-batches of the run, and is cross-checked against the presence of `messages`."""

    def __init__(self, model, acc, n_chunks):
        self.model = model
        self.acc = acc
        self.n_chunks = n_chunks
        cfg = model.config
        self.s0, self.e0 = cfg.loop_layer_index_start, cfg.loop_layer_index_end
        self.calls = {}
        self.cur = None  # (layer, pass) of the decoder layer now running
        self.streams = None  # stream tensors of the current self_attn call
        self.handles = []
        self.reader_checks = {}
        for L, layer in enumerate(model.model.layers):
            self.handles.append(layer.register_forward_pre_hook(self._layer_pre(L), with_kwargs=True))
            att = layer.self_attn
            self.handles.append(att.register_forward_pre_hook(self._attn_pre(L), with_kwargs=True))
            import torch.nn as nn

            for name, mod in att.named_modules():
                if isinstance(mod, nn.Linear):
                    if name not in READERS:
                        raise KeyError(f"layer {L}: self_attn.{name} is a linear absent from READERS")
                    self.handles.append(mod.register_forward_pre_hook(self._reader(L, name)))

    def remove(self):
        for h in self.handles:
            h.remove()

    def _layer_pre(self, L):
        def hook(mod, args, kwargs):
            c = self.calls.get(L, 0)
            self.calls[L] = c + 1
            loop = self.s0 <= L < self.e0
            pas = (c // self.n_chunks + 1) if loop else 0
            messages = args[2] if len(args) > 2 else kwargs.get("messages")
            if loop and (messages is not None) != (pas >= 2):
                raise RuntimeError(f"layer {L} call {c}: pass {pas} but messages {'set' if messages is not None else 'absent'}")
            if not loop and c >= self.n_chunks:
                raise RuntimeError(f"layer {L} outside the loop called {c + 1} times for {self.n_chunks} micro-batches")
            self.cur = (L, pas)

        return hook

    def _attn_pre(self, L):
        def hook(mod, args, kwargs):
            assert self.cur is not None and self.cur[0] == L, (self.cur, L)
            q = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            k = kwargs.get("key_hidden_states")
            v = kwargs.get("value_hidden_states")
            k = q if k is None else k
            v = q if v is None else v
            self.streams = {"query": q, "key": k, "value": v}
            _, pas = self.cur
            for s in ("query", "key", "value"):
                self.acc.add(L, pas, s, self.streams[s])

        return hook

    def _reader(self, L, name):
        stream = READERS[name]

        def hook(mod, args):
            x = args[0]
            _, pas = self.cur
            if stream == "gated_out":
                self.acc.add(L, pas, "gated_out", x)
                ok = True
            else:
                ok = bool(x is self.streams[stream] or (x.shape == self.streams[stream].shape
                                                         and bool((x == self.streams[stream]).all())))
                if not ok:
                    raise RuntimeError(f"layer {L} pass {pas}: self_attn.{name} does not read the {stream} stream")
            self.reader_checks[(L, name)] = self.reader_checks.get((L, name), 0) + 1

        return hook


def capture(teacher, ids, layers, group=8, micro=1, chunk=8192, sha=True, log=print):
    """Run `ids` [N, T] through `teacher` (an `executor.Teacher`) group by group with the hooks on.
    Returns (Accumulator, groups)."""
    model = teacher.model
    acc = Accumulator(layers, chunk=chunk, sha=sha)
    groups = []
    n = ids.shape[0]
    for g0 in range(0, n, group):
        g = list(range(g0, min(g0 + group, n)))
        n_chunks = (len(g) + micro - 1) // micro
        hooks = Hooks(model, acc, n_chunks)
        tg = time.time()
        try:
            teacher.run(ids[g], micro=micro, on_logits=lambda i, l: None, log=None)
        finally:
            hooks.remove()
        groups.append({"sequences": [g[0], g[-1]], "sec": time.time() - tg,
                       "reader_checks": sum(hooks.reader_checks.values())})
        if log:
            log(f"group {g0 // group}: sequences {g[0]}..{g[-1]}  {time.time() - tg:.0f}s")
    return acc, groups


def write(out, acc, cfg, n_seqs, seq_len, extra):
    """Hessian files and manifest of a capture."""
    os.makedirs(out, exist_ok=True)
    files = {}
    for (L, s), gm in sorted(acc.gram.items()):
        fn = f"L{L:02d}_{s}.f64"
        gm.tofile(os.path.join(out, fn))
        files[fn] = {"layer": L, "stream": s, "shape": list(gm.shape), "tokens": acc.tokens[(L, s)],
                     "trace": float(np.trace(gm)), "sha256": hashlib.sha256(gm.tobytes()).hexdigest()}
    per_visit = [{"layer": L, "pass": p, "stream": s, "tokens": r["tokens"], "sum": r["sum"], "sumsq": r["sumsq"],
                  "sha256_x_f32": r["h"].hexdigest() if acc.sha else None}
                 for (L, p, s), r in sorted(acc.per_visit.items())]
    cfgd = {k: getattr(cfg, k) for k in ("num_hidden_layers", "loop_layer_index_start", "loop_layer_index_end",
                                         "loop_num_passes", "layer_types")}
    manifest = dict(extra)
    manifest.update({
        "n_seqs": n_seqs, "seq_len": seq_len, "n_tokens": n_seqs * seq_len,
        "layers": sorted(acc.layers), "streams": list(STREAMS), "readers": READERS, "config": cfgd,
        "hessian": "unnormalised sum over visits and tokens of x x^T; float32 Gram blocks of <= chunk tokens on the "
                   "run device, summed in float64 on the CPU",
        "chunk": acc.chunk, "files": files, "per_visit": per_visit,
    })
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    return manifest


def run_capture(a):
    from executor import Teacher, torch  # noqa: E402  (applies torch threads, patches the MoE block)
    from tokenizers import Tokenizer

    import data

    torch.set_grad_enabled(False)
    out = a.out
    if os.path.exists(os.path.join(out, "manifest.json")) and not a.force:
        sys.exit(f"{out}/manifest.json exists; --force to overwrite")
    tok = Tokenizer.from_file(os.path.join(NATIVE, "tokenizer.json"))
    windows, stats = data.calib_windows(tok, a.n, a.len)
    ids = torch.tensor(windows, dtype=torch.long)
    teacher = Teacher(device=a.device)
    teacher.model.requires_grad_(False)
    cfg = teacher.model.config
    layers = list(range(cfg.num_hidden_layers)) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    t0 = time.time()
    acc, groups = capture(teacher, ids, layers, a.group, a.micro, a.chunk, not a.no_sha,
                          log=lambda m: print(m, flush=True))
    dt = time.time() - t0
    import transformers

    write(out, acc, cfg, a.n, a.len, {
        "model": "thinkingmachines/Inkling-2-Mini@ed40e7da625f594fc9c4990f5f66be24873c22d7",
        "transformers": transformers.__version__ + " (git 046bdb8cc2)",
        "arm": "native (float32 dequantized experts, unrounded)", "dtype": "float32", "device": a.device,
        "corpus": "data.calib_windows: dclm-edu 000_00000, rows in order, blank-line joined, budget n*len*6 bytes, "
                  "no BOS, contiguous windows from token 0",
        "corpus_stats": stats, "group": a.group, "micro": a.micro, "groups": groups, "sec": dt,
    })
    print(json.dumps({"done": out, "sec": dt, "files": len(acc.gram)}))
    return out


def check(out):
    """The acceptance checks of item 1a on a finished capture. Returns (ok, report)."""
    man = json.load(open(os.path.join(out, "manifest.json")))
    nv = expected_visits(man["config"])
    s0, e0 = man["config"]["loop_layer_index_start"], man["config"]["loop_layer_index_end"]
    full = {i for i, t in enumerate(man["config"]["layer_types"]) if t != "hybrid_sliding"}
    rep = {"tokens_x_visits": [], "pre_loop_equal": [], "loop_pass2_differ": [], "per_visit_counts": []}
    ok = True

    def load(L, s):
        return np.fromfile(os.path.join(out, f"L{L:02d}_{s}.f64"), np.float64)

    pv = {(r["layer"], r["pass"], r["stream"]): r for r in man["per_visit"]}
    for L in man["layers"]:
        for s in STREAMS:
            f = man["files"].get(f"L{L:02d}_{s}.f64")
            want = man["n_tokens"] * nv[L]
            got = f["tokens"] if f else 0
            passes = sorted(p for (LL, p, ss) in pv if LL == L and ss == s)
            want_passes = list(range(1, nv[L] + 1)) if s0 <= L < e0 else [0]
            good = got == want and passes == want_passes and all(
                pv[(L, p, s)]["tokens"] == man["n_tokens"] for p in passes)
            ok &= good
            rep["tokens_x_visits"].append({"layer": L, "stream": s, "tokens": got, "expected": want,
                                           "passes": passes, "ok": good})
    for L in man["layers"]:
        q, k, v = load(L, "query"), load(L, "key"), load(L, "value")
        eq = bool(np.array_equal(q, k) and np.array_equal(q, v))
        if L < s0 and L not in full:
            ok &= eq
            rep["pre_loop_equal"].append({"layer": L, "q_k_v_bit_equal": eq, "ok": eq})
        elif L < s0:
            rep["pre_loop_equal"].append({"layer": L, "q_k_v_bit_equal": eq, "note": "full attention: three "
                                          "attention_input queries and three norms, not expected equal"})
        if s0 <= L < e0:
            rows = {}
            for p in range(1, nv[L] + 1):
                if any((L, p, s) not in pv for s in ("query", "key", "value")):
                    rows[p] = {"missing": True, "sha_all_equal": False, "sha_all_distinct": False}
                    continue
                hs = [pv[(L, p, s)]["sha256_x_f32"] for s in ("query", "key", "value")]
                sums = [pv[(L, p, s)]["sum"] for s in ("query", "key", "value")]
                rows[p] = {"sha_all_equal": len(set(hs)) == 1, "sha_all_distinct": len(set(hs)) == 3,
                           "sums": sums}
            differ2 = rows.get(2, {}).get("sha_all_distinct", False) and not np.array_equal(q, k) and not np.array_equal(q, v) \
                and not np.array_equal(k, v)
            ok &= differ2
            rep["loop_pass2_differ"].append({"layer": L, "pass_rows": rows, "summed_hessians_differ": differ2,
                                             "ok": differ2,
                                             "pass1_equal_expected": L not in full})
    rep["ok"] = bool(ok)
    return ok, rep


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "check":
        ok, rep = check(sys.argv[2])
        json.dump(rep, open(os.path.join(sys.argv[2], "check.json"), "w"), indent=1)
        print(json.dumps({"check": sys.argv[2], "ok": ok,
                          "tokens_x_visits_ok": all(r["ok"] for r in rep["tokens_x_visits"]),
                          "pre_loop_equal": rep["pre_loop_equal"],
                          "loop_pass2_differ": [{k: r[k] for k in ("layer", "ok")} for r in rep["loop_pass2_differ"]]}))
        sys.exit(0 if ok else 1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--len", type=int, default=2048)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--micro", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--no-sha", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.out = a.out or os.path.join(DATA, "attn", f"n{a.n}x{a.len}")
    run_capture(a)


if __name__ == "__main__":
    main()
