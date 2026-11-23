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
"""Layer-major executor of Inkling 2 Mini (transformers 046bdb8cc2 modules, float32, no KV cache).

Every sequence goes through one layer visit before the next visit starts, so one layer's experts are resident at a
time (4.1 GB in float32). Inside a layer everything is the library's own module (decoder layers, attention, norms,
convolutions, message gates, `message_input`, masks, `final_input`, `norm`, `lm_head`, softcap). Only the top-level
schedule of `Inkling2LoopTextModel.forward` (lines 1350-1443) is reproduced, in `Teacher.run`.

The experts module is replaced, at construction, by `ProviderExperts`: same forward as `Inkling2LoopExperts` (the
library's per-expert loop, copied verbatim) but its weights come from a provider instead of 3-D parameters, so the
65 GB of float32 expert parameters are never allocated.

Arms: `parse_arm("6-8=dir:/path,20=gguf:/file.gguf")` maps layers to sources; any layer not named is `native`.
Sources: `native` (nvfp4.native_experts), `dir:<path>` (raw little-endian `w13_hat.f32`/`w2_hat.f32`, or `.f16`, in
`<path>/L{layer:02d}/`), `gguf:<file>` (`ggufx.load_gguf_experts(path, layer)`, imported lazily).

Weights: the non-expert tensors are loaded by the library's own `from_pretrained` and its checkpoint conversion
mapping (`inkling2_loop_text`), from `D/textonly/`, a copy of the native file without the experts and the vision
tower (`uv run executor.py build-textonly`).

Numerics: float32 everywhere (the library's native dtype for this checkpoint is bfloat16; its fp32 islands, router,
convolutions, residual attention, stay fp32). Experts are the unrounded float32 dequantization (see nvfp4.py).

Self-test (gate G2): `uv run executor.py g2 [cpu|mps] [ntok]`: one sequence, this executor against the library's
`Inkling2LoopForCausalLM.forward` on the same weights.
"""

import _threads  # noqa: F401  (BLAS threads before numpy/torch)

import json
import os
import sys
import time

import numpy as np

from _threads import DATA, HIDDEN, INTER, NATIVE, NATIVE_FILE, NUM_EXPERTS, TEXTONLY, apply_torch

torch = apply_torch()
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers.models.inkling2_loop import modeling_inkling2_loop as M  # noqa: E402
from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig  # noqa: E402


# ---------------------------------------------------------------- arms and the expert provider


def _parse_layers(s):
    out = []
    for part in s.split("+"):
        if part == "all":
            out += list(range(24))
        elif "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def parse_arm(spec):
    """`"6-8=dir:/p,20+21=gguf:/f.gguf"` -> {6: "dir:/p", 7: ..., 20: "gguf:/f.gguf", 21: ...}. Empty -> native."""
    arm = {}
    if not spec or spec == "native":
        return arm
    if os.path.isfile(spec) and spec.endswith(".json"):
        return {int(k): v for k, v in json.load(open(spec)).items()}
    for item in spec.split(","):
        layers, src = item.split("=", 1)
        if not (src == "native" or src.startswith("dir:") or src.startswith("gguf:")):
            raise ValueError(f"unknown source {src!r}")
        for L in _parse_layers(layers):
            arm[L] = src
    return arm


def arm_name(arm):
    if not arm:
        return "native"
    by_src = {}
    for L, s in sorted(arm.items()):
        by_src.setdefault(s, []).append(L)
    return ";".join(f"{','.join(map(str, Ls))}={s}" for s, Ls in by_src.items())


def _read_raw(path_noext, shape):
    for ext, dt in ((".f32", np.float32), (".f16", np.float16)):
        p = path_noext + ext
        if os.path.exists(p):
            a = np.fromfile(p, dtype=dt)
            if a.size != int(np.prod(shape)):
                raise ValueError(f"{p}: {a.size} values, expected {shape}")
            return torch.from_numpy(a.reshape(shape).astype(np.float32, copy=False))
    raise FileNotFoundError(path_noext + ".{f32,f16}")


class Provider:
    def __init__(self, arm=None, device="cpu"):
        self.arm = arm or {}
        self.device = device

    def source(self, layer):
        return self.arm.get(layer, "native")

    def get(self, layer):
        src = self.source(layer)
        if src == "native":
            import nvfp4

            w13, w2 = nvfp4.native_experts(layer)
        elif src.startswith("dir:"):
            d = os.path.join(src[4:], f"L{layer:02d}")
            w13 = _read_raw(os.path.join(d, "w13_hat"), (NUM_EXPERTS, 2 * INTER, HIDDEN))
            w2 = _read_raw(os.path.join(d, "w2_hat"), (NUM_EXPERTS, HIDDEN, INTER))
        elif src.startswith("gguf:"):
            import ggufx

            w13, w2 = ggufx.load_gguf_experts(src[5:], layer)
            w13 = torch.as_tensor(np.asarray(w13)) if not torch.is_tensor(w13) else w13
            w2 = torch.as_tensor(np.asarray(w2)) if not torch.is_tensor(w2) else w2
            w13, w2 = w13.float(), w2.float()
        else:
            raise ValueError(src)
        assert tuple(w13.shape) == (NUM_EXPERTS, 2 * INTER, HIDDEN), (src, layer, w13.shape)
        assert tuple(w2.shape) == (NUM_EXPERTS, HIDDEN, INTER), (src, layer, w2.shape)
        return w13.contiguous().to(self.device), w2.contiguous().to(self.device)


class ProviderExperts(nn.Module):
    """`Inkling2LoopExperts` without parameters: weights are set per visit (`w13`, `w2`), or fetched lazily from
    `provider` when unset. `recorder(x, top_k_index, top_k_weights)` sees exactly the experts' inputs."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.act_fn = M.Inkling2LoopSoftcappedSwiGLU(config)
        self.layer_idx = None
        self.provider = None
        self.w13 = None
        self.w2 = None
        self.recorder = None

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if self.recorder is not None:
            self.recorder(hidden_states, top_k_index, top_k_weights)
        if self.w13 is not None:
            gate_up_proj, down_proj = self.w13, self.w2
        else:
            gate_up_proj, down_proj = self.provider.get(self.layer_idx)
        # verbatim: Inkling2LoopExperts.forward (modeling_inkling2_loop.py 787-807), parameters -> provided tensors
        final_hidden_states = torch.zeros_like(hidden_states, dtype=torch.float32)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            current_hidden_states = self._apply_gate(F.linear(current_state, gate_up_proj[expert_idx]))
            current_hidden_states = F.linear(current_hidden_states, down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states.to(hidden_states.dtype)

    def _apply_gate(self, gate_up):
        gate, up = gate_up.chunk(2, dim=-1)
        return self.act_fn(gate, up)


def _moe_init(self, config):
    nn.Module.__init__(self)
    self.experts = ProviderExperts(config)
    self.gate = M.Inkling2LoopTopKRouter(config)


# Patch the MoE block (not the experts class: `_init_weights` matches on `Inkling2LoopExperts` by name).
M.Inkling2LoopSparseMoeBlock.__init__ = _moe_init


# ---------------------------------------------------------------- weights


def build_textonly(src=NATIVE_FILE, out=TEXTONLY):
    """Copy every `model.llm.*` tensor except the experts into D/textonly/model.safetensors, with a text config."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    os.makedirs(out, exist_ok=True)
    cfg = json.load(open(os.path.join(NATIVE, "config.json")))
    tc = dict(cfg["text_config"])
    tc["architectures"] = ["Inkling2LoopForCausalLM"]
    tc["eos_token_id"] = cfg.get("eos_token_id")
    json.dump(tc, open(os.path.join(out, "config.json"), "w"), indent=2)
    tensors = {}
    with safe_open(src, framework="pt") as f:
        for k in f.keys():
            if k.startswith("model.llm.") and ".mlp.experts." not in k:
                tensors[k] = f.get_tensor(k)
    save_file(tensors, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    print(f"textonly: {len(tensors)} tensors -> {out}")


def load_model(device="cpu"):
    if not os.path.exists(os.path.join(TEXTONLY, "model.safetensors")):
        build_textonly()
    config = Inkling2LoopTextConfig.from_pretrained(TEXTONLY)
    model, info = M.Inkling2LoopForCausalLM.from_pretrained(
        TEXTONLY, config=config, dtype=torch.float32, attn_implementation="eager", output_loading_info=True
    )
    bad = {k: v for k, v in info.items() if v and k != "mismatched_keys"} | (
        {"mismatched_keys": info["mismatched_keys"]} if info.get("mismatched_keys") else {}
    )
    if bad:
        raise RuntimeError(f"loading info not clean: {json.dumps({k: list(v)[:20] for k, v in bad.items()})}")
    for i, layer in enumerate(model.model.layers):
        layer.mlp.experts.layer_idx = i
    model.eval()
    # No autograd, ever: with grad on, every F.linear keeps its (lazily provided, 4 GB a layer) expert weight
    # alive for backward, and a library forward then holds all 48 visits' experts (about 196 GB).
    model.requires_grad_(False)
    torch.set_grad_enabled(False)
    return model.to(device)


# ---------------------------------------------------------------- the layer-major schedule


def visit_schedule(config):
    """[(layer, pass)] in execution order; pass 0 for layers outside the loop, 1..num_passes inside it."""
    s, e, n = config.loop_layer_index_start, config.loop_layer_index_end, config.loop_num_passes
    v = [(L, 0) for L in range(s)]
    for p in range(1, n + 1):
        v += [(L, p) for L in range(s, e)]
    v += [(L, 0) for L in range(e, config.num_hidden_layers)]
    return v


class Teacher:
    def __init__(self, device="cpu", arm=None, model=None):
        self.device = device
        self.model = model if model is not None else load_model(device)
        self.cfg = self.model.config
        self.set_arm(arm)
        self.visits = visit_schedule(self.cfg)

    def set_arm(self, arm):
        self.provider = Provider(arm or {}, self.device)
        for layer in self.model.model.layers:
            layer.mlp.experts.provider = self.provider

    @torch.inference_mode()
    def run(self, ids, micro=1, recorder=None, on_logits=None, log=print):
        """Run sequences `ids` [N, T] layer-major. `recorder(visit, layer, pass, seq_indices, x, idx, w)` gets the
        experts' inputs of each visit, micro-batch by micro-batch (tokens seq-major). `on_logits(i, logits[T, V])` is
        called for every sequence at the end. All N sequences' states stay resident (about 190 MB a sequence at
        T=2048); split N outside for larger sets."""
        tm = self.model.model
        cfg = self.cfg
        dev = self.device
        ids = ids.to(dev)
        N, T = ids.shape
        s0, e0 = cfg.loop_layer_index_start, cfg.loop_layer_index_end
        chunks = [list(range(i, min(i + micro, N))) for i in range(0, N, micro)]
        position_ids = torch.arange(T, device=dev).unsqueeze(0)

        states = []
        masks = {}
        for ch in chunks:
            emb = tm.embed_tokens(ids[ch])
            states.append({"h": emb, "b": emb.new_zeros(*emb.shape[:-1], 0, emb.shape[-1])})
            b = len(ch)
            if b not in masks:
                mk = {"config": cfg, "inputs_embeds": emb, "attention_mask": None, "cache": None,
                      "position_ids": position_ids}
                masks[b] = (
                    {
                        "full_attention": M.create_causal_mask(**mk),
                        "sliding_attention": M.create_sliding_window_causal_mask(**mk),
                        "linear_attention": M.create_recurrent_attention_mask(**mk),
                    },
                    {
                        "full_attention": M.get_relative_distance(emb, None, is_sliding=False),
                        "sliding_attention": M.get_relative_distance(emb, None, is_sliding=True),
                    },
                )

        t0 = time.time()
        for vi, (L, p) in enumerate(self.visits):
            tv = time.time()
            layer = tm.layers[L]
            ex = layer.mlp.experts
            ex.w13, ex.w2 = self.provider.get(L)
            tload = time.time() - tv
            for ci, ch in enumerate(chunks):
                st = states[ci]
                cmask, rel = masks[len(ch)]
                if p >= 2 and L == s0:
                    # lines 1398-1402: restart from the pre-loop states, keep the previous pass for the messages
                    st["pb"] = st["b"][..., st["npre"] + 1 :, :]
                    st["ph"] = st["h"]
                    st["h"], st["b"] = st["preh"], st["preb"]
                if recorder is not None:
                    ex.recorder = lambda x, i, w, _vi=vi, _L=L, _p=p, _ch=ch: recorder(_vi, _L, _p, _ch, x, i, w)
                common = dict(
                    attention_mask=cmask[layer.attention_type],
                    relative_distance=rel[layer.attention_type],
                    conv_mask=cmask["linear_attention"],
                    cache=None,
                    position_ids=position_ids,
                )
                if p >= 2:
                    messages = tm.message_input(st["pb"], st["ph"], L - s0)
                    new_h = layer(st["h"], st["b"], messages, **common)
                else:
                    new_h = layer(st["h"], st["b"], **common)
                ex.recorder = None
                if layer.is_block_boundary:
                    st["b"] = torch.cat([st["b"], st["h"].unsqueeze(-2)], dim=-2)
                st["h"] = new_h
                if p == 0 and L == s0 - 1:
                    # line 1378
                    st["preh"], st["preb"] = st["h"], st["b"]
                    st["npre"] = st["b"].shape[-2]
                if p == cfg.loop_num_passes and L == e0 - 1:
                    for k in ("preh", "preb", "pb", "ph"):
                        st.pop(k, None)
            ex.w13 = ex.w2 = None
            if log:
                log(f"  visit {vi:02d} L{L:02d} p{p}  load {tload:5.1f}s  total {time.time() - tv:6.1f}s  "
                    f"({N * T / max(time.time() - tv, 1e-9):7.1f} tok/s this visit)")
        out = []
        for ci, ch in enumerate(chunks):
            st = states[ci]
            h = tm.final_input(st["b"], st["h"])[0]
            h = tm.norm(h).to(torch.float32)
            logits = self.model.logit_softcap(self.model.lm_head(h).float())
            for j, i in enumerate(ch):
                if on_logits is not None:
                    on_logits(i, logits[j])
                else:
                    out.append(logits[j].cpu())
            states[ci] = None
        if log:
            dt = time.time() - t0
            log(f"  run: {N} x {T} tokens in {dt:.1f}s = {N * T / dt:.2f} tok/s")
        return out if on_logits is None else None


# ---------------------------------------------------------------- G2


def g2(device="cpu", ntok=256):
    from tokenizers import Tokenizer

    import data

    tok = Tokenizer.from_file(os.path.join(NATIVE, "tokenizer.json"))
    ids = data.wikitext_ids(tok)[:ntok]
    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    t = Teacher(device=device)
    t0 = time.time()
    mine = t.run(ids, micro=1)[0]
    t_mine = time.time() - t0
    t0 = time.time()
    with torch.inference_mode():
        lib = t.model(input_ids=ids.to(device), use_cache=False).logits[0].float().cpu()
    t_lib = time.time() - t0
    d = (mine - lib).abs()
    top_same = (mine.argmax(-1) == lib.argmax(-1))
    res = {
        "gate": "G2",
        "device": device,
        "ntok": ntok,
        "max_abs_diff_logits": float(d.max()),
        "mean_abs_diff_logits": float(d.mean()),
        "top1_same_all": bool(top_same.all()),
        "top1_same_count": int(top_same.sum()),
        "logit_absmax": float(lib.abs().max()),
        "sec_executor": t_mine,
        "sec_library": t_lib,
    }
    print(json.dumps(res), flush=True)
    os.makedirs(os.path.join(DATA, "eval"), exist_ok=True)
    json.dump(res, open(os.path.join(DATA, "eval", f"g2_{device}.json"), "w"), indent=2)
    return res


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "build-textonly":
        build_textonly()
    elif len(sys.argv) >= 2 and sys.argv[1] == "g2":
        g2(sys.argv[2] if len(sys.argv) > 2 else "cpu", int(sys.argv[3]) if len(sys.argv) > 3 else 256)
    else:
        print(__doc__)
