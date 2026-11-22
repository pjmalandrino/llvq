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
"""Non-expert tensors of an Inkling 2 Mini GGUF, mapped to the `transformers` parameters and verified.

    uv run ggufmap.py verify <file.gguf> [<file.gguf> ...]     # heavy: loads the model on the CPU

The name map is a rule table (`RULES`), established from the tensor lists of the four unsloth files and of the
ggml-org BF16 file's header (`D/eval/ref_bf16/bf16_header.json`): the converters disagree on some names (message gate
controllers `loop_attn_down` against `loop_q_ctrl_down`, message queries global against per layer), so a rule may list
several candidate names, and the first one present in the file is taken.

The orientation is not assumed. For each parameter, every candidate transform of the GGUF array (C-order view of
`ne` reversed: identity reshape, transpose, a flip of the last or the first axis, combined; the first `unpadded_vocab_size` rows for the
unembedding) is compared with the native parameter, and the closest by relative Frobenius error is kept. A mapping is
`verified` when: shapes agree after the transform; correlation >= 0.9; relative error <= 0.6; and, for a per-layer
tensor, the native tensor of the same name in the adjacent layer is farther (relative error at least 1.5 times larger;
skipped when the adjacent layer's native tensor equals this one to 1e-6, recorded as `control: equal`). Thresholds are
loose on purpose: they must accept a 2-bit tensor and reject a wrong name, a wrong layer or a wrong orientation.

The report, D/evalfull/ggufmap_<file stem>.json, lists every parameter: GGUF name, type, transform, correlation,
relative error, control, verdict. `verified_map(path)` reads it; `load_for_param` returns a verified tensor and refuses
an unverified one.
"""

import _threads  # noqa: F401

import json
import os
import sys
import time

import numpy as np

from _threads import DATA

OUT = os.path.join(DATA, "evalfull")
_MSG = {"q": ("attn", "q_ctrl", "q"), "k": ("k", "k_ctrl"), "v": ("v", "v_ctrl"), "mlp": ("mlp", "ffn_ctrl")}

# per-layer rules: transformers suffix after `model.layers.{L}.` -> candidate GGUF names after `blk.{L}.`
RULES = {
    "self_attn.q_proj.linear.weight": ["attn_q.weight"],
    "self_attn.q_proj.norm.scale": ["attn_q_clip.weight"],
    "self_attn.k_proj.linear.weight": ["attn_k.weight"],
    "self_attn.v_proj.linear.weight": ["attn_v.weight"],
    "self_attn.k_proj.sconv.conv1d.weight": ["attn_k_conv.weight"],
    "self_attn.v_proj.sconv.conv1d.weight": ["attn_v_conv.weight"],
    "self_attn.k_proj.sconv.static_weight": ["attn_k_conv.weight"],
    "self_attn.v_proj.sconv.static_weight": ["attn_v_conv.weight"],
    "self_attn.k_proj.sconv.weight_proj.weight": ["attn_k_conv_proj.weight"],
    "self_attn.v_proj.sconv.weight_proj.weight": ["attn_v_conv_proj.weight"],
    "self_attn.o_proj.gate_proj.weight": ["attn_gate.weight"],
    "self_attn.o_proj.linear.weight": ["attn_output.weight"],
    "self_attn.sink_proj.weight": ["attn_sink_proj.weight"],
    "self_attn.relative_position_bias.proj": ["attn_rel_bias.weight"],
    "self_attn.relative_position_bias.linear.weight": ["attn_rel.weight"],
    "self_attn.relative_position_bias.norm.scale": ["attn_rel_clip.weight"],
    "attention_input.query": ["attn_res_query.weight"],
    "mlp_input.query": ["ffn_res_query.weight"],
    "input_layernorm.weight": ["attn_norm.weight"],
    "q_input_layernorm.weight": ["attn_q_inp_norm.weight"],
    "k_input_layernorm.weight": ["attn_k_inp_norm.weight"],
    "v_input_layernorm.weight": ["attn_v_inp_norm.weight"],
    "post_attention_layernorm.weight": ["ffn_norm.weight"],
    "post_attention_convolution.conv1d.weight": ["attn_post_conv.weight"],
    "post_mlp_convolution.conv1d.weight": ["ffn_post_conv.weight"],
    "mlp.gate.weight": ["ffn_gate_inp.weight"],
    "mlp.gate.e_score_correction_bias": ["exp_probs_b.bias"],
}
for _g, _alts in _MSG.items():
    for _hf, _gg in (("down_proj", "down"), ("forget_gate_proj", "forget"), ("message_gate_proj", "gain")):
        RULES[f"{_g}_message_gate.{_hf}.weight"] = [f"loop_{a}_{_gg}.weight" for a in _alts]
GLOBAL = {
    "model.embed_tokens.weight": ["token_embd.weight"],
    "model.embed_tokens.embed_norm.weight": ["token_embd_norm.weight"],
    "model.norm.weight": ["output_norm.weight"],
    "model.final_input.query": ["output_res_query.weight"],
    "lm_head.weight": ["output.weight"],
}


def gguf_array(path, name):
    """Float32 array of a GGUF tensor, C-order view of `ne` reversed."""
    import ggufx

    mm, tensors, ds = ggufx._open(path)
    t = tensors[name]
    ne = t["ne"]
    n, nb = ggufx._nbytes(t)
    raw = np.frombuffer(mm, dtype=np.uint8, count=nb, offset=ds + t["offset"])
    tid = t["type"]
    if tid == 0:
        a = raw.view("<f4").astype(np.float32)
    elif tid == 1:
        a = raw.view("<f2").astype(np.float32)
    elif tid == 30:
        a = (raw.view("<u2").astype(np.uint32) << 16).view(np.float32)
    else:
        from gguf.quants import dequantize

        rows = n // ne[0]
        a = dequantize(raw.reshape(rows, -1), ggufx._qtype(tid)).astype(np.float32, copy=False)
    return np.ascontiguousarray(a.reshape(list(reversed(ne)))), ggufx.GGML.get(tid, (str(tid),))[0]


def _candidates(a, shape, rows_cut=None):
    """(transform name, array of `shape`) for each orientation that fits."""
    out = []
    n = int(np.prod(shape))
    bases = [("id", a)]
    if a.ndim == 2:
        bases.append(("T", a.T))
    for bname, b in bases:
        if rows_cut is not None and b.ndim == 2 and b.shape[0] >= rows_cut and b.shape[0] * b.shape[1] != n:
            b = b[:rows_cut]
            bname += f"+rows[:{rows_cut}]"
        if b.size != n:
            continue
        flips = [("", b), ("+flip", b[..., ::-1])]
        if b.ndim >= 2:
            # the dynamic convolutions' static taps are stored in time order, transformers wants backward distance
            flips.append(("+flip0", b[::-1]))
        for fname, c in flips:
            if fname and (c.shape[-1] < 2 if fname == "+flip" else c.shape[0] < 2):
                continue
            out.append((bname + fname, np.ascontiguousarray(c).reshape(shape)))
    return out


def _stats(x, ref):
    x = x.astype(np.float64).ravel()
    r = ref.astype(np.float64).ravel()
    den = np.linalg.norm(r)
    rel = float(np.linalg.norm(x - r) / den) if den > 0 else float(np.linalg.norm(x - r))
    corr = float(np.corrcoef(x, r)[0, 1]) if x.std() > 0 and r.std() > 0 else (1.0 if np.allclose(x, r) else 0.0)
    return rel, corr


def _message_queries(path, tensors, cfg):
    """`model.message_input.query` [loop layers, 4, H] from the GGUF, or None."""
    import torch
    from transformers.core_model_loading import Inkling2LoopMergeMessageQueries

    if "loop_res_query.weight" in tensors and "loop_kv_res_query.weight" in tensors:
        q, _ = gguf_array(path, "loop_res_query.weight")
        kv, _ = gguf_array(path, "loop_kv_res_query.weight")
        m = Inkling2LoopMergeMessageQueries().convert(
            {"q": torch.from_numpy(q), "kv": torch.from_numpy(kv)}, source_patterns=["q", "kv"], target_patterns=["t"]
        )["t"]
        return m.numpy(), "loop_res_query.weight+loop_kv_res_query.weight", "merge(Inkling2LoopMergeMessageQueries)"
    s, e = cfg.loop_layer_index_start, cfg.loop_layer_index_end
    names = [f"blk.{L}.loop_msg_query.weight" for L in range(s, e)]
    if all(n in tensors for n in names):
        return np.stack([gguf_array(path, n)[0] for n in names]), "blk.L.loop_msg_query.weight", "stack over loop layers"
    return None


def plan_names(path, model_names, cfg):
    """{transformers name: candidate GGUF name present in the file, or None}."""
    import ggufx

    _, tensors, _ = ggufx._open(path)
    res = {}
    for n in model_names:
        if ".mlp.experts." in n or n == "model.message_input.query":
            continue
        if n in GLOBAL:
            cands = GLOBAL[n]
        elif n.startswith("model.layers."):
            L, suffix = n[len("model.layers."):].split(".", 1)
            cands = [f"blk.{L}.{c}" for c in RULES.get(suffix, [])]
        else:
            cands = []
        res[n] = next((c for c in cands if c in tensors), None)
    return res


def verify(path, model):
    import ggufx
    import torch

    _, tensors, _ = ggufx._open(path)
    cfg = model.config
    params = {n: p.detach().float().cpu().numpy() for n, p in
              list(model.named_parameters()) + list(model.named_buffers())}
    names = plan_names(path, list(params), cfg)
    rows = {}
    used = set()
    t0 = time.time()
    for n, g in sorted(names.items()):
        ref = params[n]
        row = {"gguf": g}
        if g is None:
            row.update({"verdict": "unmapped", "reason": "no rule names a tensor of this file"})
            rows[n] = row
            continue
        a, ty = gguf_array(path, g)
        used.add(g)
        cut = cfg.unpadded_vocab_size if n == "lm_head.weight" else None
        cands = _candidates(a, ref.shape, cut)
        if not cands:
            row.update({"type": ty, "gguf_shape": list(a.shape), "verdict": "unverified",
                        "reason": f"no orientation of {list(a.shape)} fits {list(ref.shape)}"})
            rows[n] = row
            continue
        scored = sorted(((*_stats(c, ref), tn) for tn, c in cands), key=lambda r: r[0])
        rel, corr, tn = scored[0]
        row.update({"type": ty, "gguf_shape": list(a.shape), "transform": tn, "rel_err": rel, "corr": corr,
                    "runner_up": {"transform": scored[1][2], "rel_err": scored[1][0]} if len(scored) > 1 else None})
        ctrl = "none"
        if n.startswith("model.layers."):
            L, suffix = n[len("model.layers."):].split(".", 1)
            for adj in (int(L) + 1, int(L) - 1):
                an = f"model.layers.{adj}.{suffix}"
                if an in params and params[an].shape == ref.shape:
                    if np.allclose(params[an], ref, atol=1e-6, rtol=0):
                        ctrl = "equal"
                    else:
                        best = dict(cands)[tn]
                        crel, _ = _stats(best, params[an])
                        row["control_layer"] = adj
                        row["control_rel_err"] = crel
                        ctrl = "pass" if crel >= 1.5 * rel else "fail"
                    break
        row["control"] = ctrl
        ok = corr >= 0.9 and rel <= 0.6 and ctrl != "fail"
        row["verdict"] = "verified" if ok else "unverified"
        if not ok:
            row["reason"] = f"corr {corr:.4f}, rel {rel:.4f}, control {ctrl}"
        rows[n] = row
    mq = _message_queries(path, tensors, cfg)
    if mq is not None:
        a, g, how = mq
        ref = params["model.message_input.query"]
        if a.shape == ref.shape:
            rel, corr = _stats(a, ref)
            ok = corr >= 0.9 and rel <= 0.6
            rows["model.message_input.query"] = {"gguf": g, "type": "F32", "transform": how, "rel_err": rel,
                                                 "corr": corr, "control": "none",
                                                 "verdict": "verified" if ok else "unverified"}
            used.update(g.split("+"))
        else:
            rows["model.message_input.query"] = {"gguf": g, "verdict": "unverified",
                                                 "reason": f"shape {list(a.shape)} vs {list(ref.shape)}"}
    else:
        rows["model.message_input.query"] = {"gguf": None, "verdict": "unmapped"}
    non_expert = [k for k in tensors if "_exps." not in k]
    leftover = sorted(k for k in non_expert if k not in used and not any(k.startswith(f"blk.{L}.loop_msg_query")
                                                                          for L in range(64)))
    counts = {}
    for r in rows.values():
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    rep = {"file": path, "counts": counts, "gguf_non_expert_tensors_unused": leftover, "rows": rows,
           "sec": time.time() - t0,
           "criteria": "corr >= 0.9, rel_err <= 0.6, adjacent-layer control rel_err >= 1.5 x (when not equal)"}
    os.makedirs(OUT, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    json.dump(rep, open(os.path.join(OUT, f"ggufmap_{stem}.json"), "w"), indent=1)
    print(json.dumps({"file": os.path.basename(path), "counts": counts, "unused": leftover[:20],
                      "unverified": {k: v.get("reason") for k, v in rows.items() if v["verdict"] != "verified"}}),
          flush=True)
    del torch
    return rep


def verified_map(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    p = os.path.join(OUT, f"ggufmap_{stem}.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p}: run `ggufmap.py verify {path}` first")
    return json.load(open(p))["rows"]


def load_for_param(path, rows, name, shape):
    import torch

    r = rows.get(name)
    if r is None or r["verdict"] != "verified":
        raise ValueError(f"{name}: no verified GGUF mapping in {os.path.basename(path)} ({r and r.get('reason')})")
    if name == "model.message_input.query":
        import ggufx

        _, tensors, _ = ggufx._open(path)
        a = _message_queries(path, tensors, None)[0]
    else:
        a, _ = gguf_array(path, r["gguf"])
        cut = shape[0] if name == "lm_head.weight" else None
        a = dict(_candidates(a, shape, cut))[r["transform"]]
    assert tuple(a.shape) == tuple(shape), (name, a.shape, shape)
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "verify":
        import executor

        model = executor.load_model("cpu")
        for p in sys.argv[2:]:
            verify(p, model)
        return
    print(__doc__)


if __name__ == "__main__":
    main()
