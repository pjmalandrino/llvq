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
"""Float32 dumps of the reference Inkling 2 Mini, layers 0 to 5, for the Rust oracle (`tests/inkling_oracle.rs`).

The reference is the library's own modules (transformers 046bdb8cc2, `modeling_inkling2_loop.py`), loaded by last
night's `executor.load_model` from `D/textonly/` in float32, with the native experts dequantized in float32 by
`nvfp4.native_experts` (the library's `Inkling2LoopDequantizeExperts` and `Interleave(dim=1)`). Both are imported by
path from the main worktree and are not edited.

The schedule is the pre-loop part of `Inkling2LoopTextModel.forward`, lines 1359-1376, without a cache: the
embedding, then layers 0 to 5, each followed by the block-state append of its boundary.

Output, under `D/lot1-c/` (override with `INKLING_LOT1C`):
- `ref_l0_5.safetensors`: `ids` (int64, [T]), `embed` ([T, 2304]), `layer{L}` for L in 0..5 ([T, 2304]), the
  routing of every layer `router{L}.indices` (int64, [T, 8], sorted ascending per token) and `router{L}.weights`
  (aligned with the sorted indices), intermediates of layers 0 and 5 under `L{L}.<module>`, and two experts of layer 0
  as the library dequantizes them, `L0.e{E}.w13` ([2304, 2304], gate rows then up rows) and `L0.e{E}.w2`.
- `ref_l0_5.json`: prompt, revision, per-layer router margin (8th minus 9th of score plus bias, min over tokens).

Usage: `~/.llvq-guard/heavy.sh "<label>" uv run ops/inkling/dump_ref.py [ntok]` (default 64 tokens). Autograd off.

`dump_ref.py full [ntok]` (lot 2, default 32 tokens): the whole text model, all 48 layer visits, the routing of each,
and the final logits, for the prompt of `D/evalfull/generate_native.json`; see `main_full`.
"""

import os
import sys
import time

# The reference tools (`executor.py`, `nvfp4.py`, `data.py`, `_threads.py`) sit beside this file; `INKLING_TOOLS` points
# elsewhere when they do not.
W_INKLING = os.environ.get("INKLING_TOOLS") or os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(W_INKLING, "executor.py")):
    sys.exit(f"dump_ref: no executor.py in {W_INKLING}; set INKLING_TOOLS to the directory that holds the tools")
sys.path.insert(0, W_INKLING)

import _threads  # noqa: E402,F401  (BLAS threads before numpy/torch)

import json  # noqa: E402

from _threads import DATA, NATIVE, apply_torch  # noqa: E402

torch = apply_torch()
torch.set_grad_enabled(False)

import executor  # noqa: E402  (patches the MoE block to take provided experts)
import nvfp4  # noqa: E402
from executor import M  # noqa: E402

REVISION = "ed40e7da625f594fc9c4990f5f66be24873c22d7"
TRANSFORMERS = "046bdb8cc2"
PROMPT = (
    "The lighthouse keeper climbed the spiral stairs at dusk, counting each of the one hundred and twelve steps as "
    "his father had taught him. Outside, the storm gathered over the northern cliffs, and the fishing boats were "
    "still three miles from the harbour. He lit the lamp, wound the clockwork that turned the great lens, and wrote "
    "the time in the logbook: seven minutes past six, wind from the west, rising."
)
OUT = os.path.expanduser(os.environ.get("INKLING_LOT1C", os.path.join(DATA, "lot1-c")))
DUMP_EXPERTS = (0, 77)
INTERMEDIATE_LAYERS = (0, 5)


def _store(dumps, name, value):
    if isinstance(value, (tuple, list)):
        for i, v in enumerate(value):
            if torch.is_tensor(v):
                _store(dumps, f"{name}.{i}", v)
        return
    if not torch.is_tensor(value):
        return
    v = value.detach()
    if v.dim() >= 1 and v.shape[0] == 1:
        v = v[0]
    v = v.to(torch.int64) if not v.is_floating_point() else v.to(torch.float32)
    dumps[name] = v.contiguous().clone()


def main(ntok=64):
    from safetensors.torch import save_file
    from tokenizers import Tokenizer

    os.makedirs(OUT, exist_ok=True)
    tok = Tokenizer.from_file(os.path.join(NATIVE, "tokenizer.json"))
    ids = tok.encode(PROMPT, add_special_tokens=False).ids
    if len(ids) < ntok:
        raise RuntimeError(f"prompt has {len(ids)} tokens, {ntok} asked")
    ids = ids[:ntok]

    t0 = time.time()
    model = executor.load_model("cpu")
    for p in model.parameters():
        p.requires_grad_(False)
    t_load = time.time() - t0
    tm = model.model
    cfg = model.config
    provider = executor.Provider({}, "cpu")
    for layer in tm.layers:
        layer.mlp.experts.provider = provider

    dumps = {}
    meta = {"revision": REVISION, "transformers": TRANSFORMERS, "prompt": PROMPT, "ntok": ntok, "ids": ids,
            "dtype": "float32", "device": "cpu", "router_margin": {}}

    # Hooks: every routing, and the intermediates of layers 0 and 5.
    hooks = []
    for L, layer in enumerate(tm.layers[: cfg.loop_layer_index_start]):
        def on_gate(mod, inp, out, _L=L):
            logits, weights, indices = out
            order = torch.sort(indices, dim=-1)
            _store(dumps, f"router{_L}.indices", order.values)
            _store(dumps, f"router{_L}.weights", weights.gather(-1, order.indices))
            s = torch.sigmoid(logits.float()) + mod.e_score_correction_bias
            top = torch.topk(s, mod.top_k + 1, dim=-1).values
            meta["router_margin"][str(_L)] = float((top[:, mod.top_k - 1] - top[:, mod.top_k]).min())
            if _L in INTERMEDIATE_LAYERS:
                _store(dumps, f"L{_L}.mlp.gate.logits", logits)

        hooks.append(layer.mlp.gate.register_forward_hook(on_gate))
        if L in INTERMEDIATE_LAYERS:
            for name, mod in layer.named_modules():
                if name in ("", "mlp.gate", "mlp.experts"):
                    continue
                if name.count(".") > 2:
                    continue

                def on_mod(m, inp, out, _n=f"L{L}.{name}"):
                    _store(dumps, _n, out)

                hooks.append(mod.register_forward_hook(on_mod))

    ids_t = torch.tensor([ids], dtype=torch.long)
    position_ids = torch.arange(ntok).unsqueeze(0)
    t1 = time.time()
    emb = tm.embed_tokens(ids_t)
    mk = {"config": cfg, "inputs_embeds": emb, "attention_mask": None, "cache": None, "position_ids": position_ids}
    masks = {
        "full_attention": M.create_causal_mask(**mk),
        "sliding_attention": M.create_sliding_window_causal_mask(**mk),
        "linear_attention": M.create_recurrent_attention_mask(**mk),
    }
    rel = {
        "full_attention": M.get_relative_distance(emb, None, is_sliding=False),
        "sliding_attention": M.get_relative_distance(emb, None, is_sliding=True),
    }
    meta["conv_mask_is_none"] = masks["linear_attention"] is None
    _store(dumps, "ids", ids_t)
    _store(dumps, "embed", emb)

    # lines 1359-1376
    hidden_states = emb
    block_states = emb.new_zeros(*emb.shape[:-1], 0, emb.shape[-1])
    for L, layer in enumerate(tm.layers[: cfg.loop_layer_index_start]):
        ex = layer.mlp.experts
        ex.w13, ex.w2 = provider.get(L)
        if L == 0:
            for e in DUMP_EXPERTS:
                dumps[f"L0.e{e}.w13"] = ex.w13[e].clone()
                dumps[f"L0.e{e}.w2"] = ex.w2[e].clone()
        new_hidden_states = layer(
            hidden_states,
            block_states,
            attention_mask=masks[layer.attention_type],
            relative_distance=rel[layer.attention_type],
            conv_mask=masks["linear_attention"],
            cache=None,
            position_ids=position_ids,
        )
        ex.w13 = ex.w2 = None
        if layer.is_block_boundary:
            block_states = torch.cat([block_states, hidden_states.unsqueeze(-2)], dim=-2)
        hidden_states = new_hidden_states
        _store(dumps, f"layer{L}", hidden_states)
        print(f"layer {L} ({layer.attention_type}) done, |h|max {float(hidden_states.abs().max()):.4f}", flush=True)
    for h in hooks:
        h.remove()
    meta["seconds_load"] = t_load
    meta["seconds_forward"] = time.time() - t1
    meta["nvfp4_module"] = nvfp4.__file__
    meta["tensors"] = {k: list(v.shape) for k, v in sorted(dumps.items())}
    save_file(dumps, os.path.join(OUT, "ref_l0_5.safetensors"), metadata={"format": "pt"})
    json.dump(meta, open(os.path.join(OUT, "ref_l0_5.json"), "w"), indent=1)
    print(json.dumps({k: meta[k] for k in ("router_margin", "seconds_load", "seconds_forward", "conv_mask_is_none")}))


def gen_prompt_ids(tok, n=32):
    """The prompt of `D/evalfull/generate_native.json`: the first `n` tokens of WikiText-2 test window 0, built as
    `W/ops/inkling/generate.py` builds it (`data.eval_windows(tok, 1, 2048)[0][:n]`), and checked against the file's
    `text_prompt`."""
    import data

    wins, _ = data.eval_windows(tok, 1, 2048)
    ids = list(wins[0][:n])
    ref = json.load(open(os.path.join(DATA, "evalfull", "generate_native.json")))
    if tok.decode(ids) != ref["text_prompt"]:
        raise RuntimeError("the rebuilt prompt does not decode to generate_native.json's text_prompt")
    return ids, ref


def main_full(ntok=32):
    """Every layer visit of the whole text model, lines 1350-1443, for the 32-token prompt of generate_native.json.

    Output under `D/lot2-d/` (override with `INKLING_LOT2D`): `ref_full.safetensors` with `ids`, `embed`, `visit{V}`
    for V in 0..47 (the hidden state after visit V, [T, 2304]), `visit{V}.indices` / `visit{V}.weights` (routing,
    sorted ascending per token), `logits` ([T, 200058], softcapped), and two experts of layer 20 (MXFP8) as the
    library dequantizes them (`L20.e{E}.w13`, `L20.e{E}.w2`). `ref_full.json`: the visit schedule, per-visit router
    margins, and the cross-check of this schedule's logits against the library's own `Inkling2LoopForCausalLM.forward`.
    """
    from safetensors.torch import save_file
    from tokenizers import Tokenizer

    out_dir = os.path.expanduser(os.environ.get("INKLING_LOT2D", os.path.join(DATA, "lot2-d")))
    os.makedirs(out_dir, exist_ok=True)
    tok = Tokenizer.from_file(os.path.join(NATIVE, "tokenizer.json"))
    ids, ref = gen_prompt_ids(tok, ntok)

    t0 = time.time()
    model = executor.load_model("cpu")
    t_load = time.time() - t0
    tm = model.model
    cfg = model.config
    provider = executor.Provider({}, "cpu")
    for layer in tm.layers:
        layer.mlp.experts.provider = provider
    visits = executor.visit_schedule(cfg)

    dumps = {}
    meta = {"revision": REVISION, "transformers": TRANSFORMERS, "ntok": ntok, "ids": ids, "dtype": "float32",
            "device": "cpu", "visits": visits, "router_margin": []}
    routed = []

    def on_gate(mod, inp, out):
        logits, weights, indices = out
        order = torch.sort(indices, dim=-1)
        v = len(routed)
        _store(dumps, f"visit{v}.indices", order.values)
        _store(dumps, f"visit{v}.weights", weights.gather(-1, order.indices))
        s = torch.sigmoid(logits.float()) + mod.e_score_correction_bias
        top = torch.topk(s, mod.top_k + 1, dim=-1).values
        meta["router_margin"].append(float((top[:, mod.top_k - 1] - top[:, mod.top_k]).min()))
        routed.append(v)

    hooks = [layer.mlp.gate.register_forward_hook(on_gate) for layer in tm.layers]

    ids_t = torch.tensor([ids], dtype=torch.long)
    position_ids = torch.arange(ntok).unsqueeze(0)
    t1 = time.time()
    emb = tm.embed_tokens(ids_t)
    mk = {"config": cfg, "inputs_embeds": emb, "attention_mask": None, "cache": None, "position_ids": position_ids}
    masks = {
        "full_attention": M.create_causal_mask(**mk),
        "sliding_attention": M.create_sliding_window_causal_mask(**mk),
        "linear_attention": M.create_recurrent_attention_mask(**mk),
    }
    rel = {
        "full_attention": M.get_relative_distance(emb, None, is_sliding=False),
        "sliding_attention": M.get_relative_distance(emb, None, is_sliding=True),
    }
    _store(dumps, "ids", ids_t)
    _store(dumps, "embed", emb)

    s0, e0 = cfg.loop_layer_index_start, cfg.loop_layer_index_end
    h = emb
    b = emb.new_zeros(*emb.shape[:-1], 0, emb.shape[-1])
    for vi, (L, p) in enumerate(visits):
        layer = tm.layers[L]
        ex = layer.mlp.experts
        ex.w13, ex.w2 = provider.get(L)
        if L == 20:
            for e in DUMP_EXPERTS:
                dumps[f"L20.e{e}.w13"] = ex.w13[e].clone()
                dumps[f"L20.e{e}.w2"] = ex.w2[e].clone()
        if p >= 2 and L == s0:
            # lines 1398-1402: restart from the pre-loop states, keep the previous pass for the messages
            pb, ph = b[..., npre + 1 :, :], h
            h, b = preh, preb
        common = dict(attention_mask=masks[layer.attention_type], relative_distance=rel[layer.attention_type],
                      conv_mask=masks["linear_attention"], cache=None, position_ids=position_ids)
        if p >= 2:
            new_h = layer(h, b, tm.message_input(pb, ph, L - s0), **common)  # line 1407
        else:
            new_h = layer(h, b, **common)
        ex.w13 = ex.w2 = None
        if layer.is_block_boundary:
            b = torch.cat([b, h.unsqueeze(-2)], dim=-2)
        h = new_h
        if p == 0 and L == s0 - 1:
            preh, preb, npre = h, b, b.shape[-2]  # line 1378
        _store(dumps, f"visit{vi}", h)
        print(f"visit {vi:02d} L{L:02d} p{p} |h|max {float(h.abs().max()):.4f}", flush=True)
    hf = tm.final_input(b, h)[0]  # line 1442
    hf = tm.norm(hf).to(torch.float32)
    logits = model.logit_softcap(model.lm_head(hf).float())
    _store(dumps, "logits", logits)
    t_fwd = time.time() - t1
    for hk in hooks:
        hk.remove()
    if len(routed) != len(visits):
        raise RuntimeError(f"{len(routed)} router calls for {len(visits)} visits")

    # Cross-check against the library's own forward (experts fetched lazily by the provider).
    t2 = time.time()
    lib = model(input_ids=ids_t, use_cache=False).logits[0].float()
    d = (lib - logits[0]).abs()
    meta["library_forward"] = {"max_abs_diff_logits": float(d.max()), "seconds": time.time() - t2,
                               "argmax_equal": bool((lib.argmax(-1) == logits[0].argmax(-1)).all())}
    meta["argmax_last_equals_generated0"] = int(logits[0, -1].argmax()) == ref["generated"][0]
    meta["seconds_load"] = t_load
    meta["seconds_forward"] = t_fwd
    meta["tensors"] = {k: list(v.shape) for k, v in sorted(dumps.items())}
    save_file(dumps, os.path.join(out_dir, "ref_full.safetensors"), metadata={"format": "pt"})
    json.dump(meta, open(os.path.join(out_dir, "ref_full.json"), "w"), indent=1)
    print(json.dumps({k: meta[k] for k in ("library_forward", "argmax_last_equals_generated0", "seconds_load",
                                           "seconds_forward")}))
    print("min router margin per visit:", [f"{m:.2e}" for m in meta["router_margin"]])


def main_window(ntok=2048):
    """The first `ntok` tokens of WikiText-2 test window 0 (`data.eval_windows(tok, 1, 2048)`, the windows of
    `W/ops/inkling/evalfull.py`), as `D/lot2-d/wikitext_w0_{ntok}.json` for gate 3c. No model is loaded."""
    from tokenizers import Tokenizer
    import data

    tok = Tokenizer.from_file(os.path.join(NATIVE, "tokenizer.json"))
    wins, info = data.eval_windows(tok, 1, 2048)
    ids = list(wins[0][:ntok])
    out_dir = os.path.expanduser(os.environ.get("INKLING_LOT2D", os.path.join(DATA, "lot2-d")))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"wikitext_w0_{ntok}.json")
    json.dump({"ids": ids, "source": info, "window": 0, "ctx": 2048, "ntok": ntok}, open(path, "w"))
    print(path, len(ids))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "window":
        main_window(int(sys.argv[2]) if len(sys.argv) > 2 else 2048)
    elif len(sys.argv) > 1 and sys.argv[1] == "full":
        main_full(int(sys.argv[2]) if len(sys.argv) > 2 else 32)
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 64)
