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
"""Light tests of items 1a and 1d on a tiny random Inkling2Loop (CPU, seconds, no checkpoint).

    INKLING_THREADS=2 uv run test_lot1_tiny.py [scratch_dir]

The tiny model keeps the real structure: 8 layers, blocks of 2, layers 2-5 looped 3 times, sliding layers 0, 1, 2, 4,
6 and full layers 3, 5, 7, so it has sliding and full layers before, inside and after the loop. Every parameter is
random. It runs the real code paths: `executor.Teacher.run`, the capture hooks and `check`, `LazyExperts` on an
`ExpertSource`, the library's per-pass caches, the student mapping.

Tests: (1) capture: check() passes; layers 0-1 have q = k = v bit for bit; loop layers differ in pass 2; tokens equal
tokens x visits; the reader table holds on every call. (2) dropping pass 3 from the accumulator fails check().
(3) cached greedy == full-forward argmax, through single experts and through full layers. (4) a decode that forgets the
cache does not pass (3), so (3) can fail. (5) `dir:` substitution of a non-expert tensor: an equal copy leaves the
logits unchanged bit for bit, a perturbed copy changes them; experts: per-expert values equal the full layer's.
"""

import _threads  # noqa: F401

import json
import os
import sys
import tempfile

import numpy as np

import executor  # noqa: E402
from executor import Teacher, torch  # noqa: E402
from transformers.models.inkling2_loop import modeling_inkling2_loop as M  # noqa: E402
from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig  # noqa: E402

import capture_attn  # noqa: E402
import generate  # noqa: E402
import model_full  # noqa: E402

torch.set_grad_enabled(False)


def tiny_config():
    return Inkling2LoopTextConfig(
        vocab_size=101, unpadded_vocab_size=97, hidden_size=48, num_hidden_layers=8,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        swa_num_attention_heads=4, swa_num_key_value_heads=2, swa_head_dim=8,
        sliding_window_size=16, d_rel=4, rel_extent=16, log_scaling_n_floor=8,
        layer_types=["hybrid_sliding", "hybrid_sliding", "hybrid_sliding", "hybrid",
                     "hybrid_sliding", "hybrid", "hybrid_sliding", "hybrid"],
        moe_intermediate_size=24, n_routed_experts=8, num_experts_per_tok=2,
        n_layers_per_attn_res_block=2, loop_layer_index_start=2, loop_layer_index_end=6, loop_num_passes=3,
        loop_gate_rank=4, eos_token_id=None,
    )


class TinySource(model_full.ExpertSource):
    """Random experts of the tiny model, served by layer or by expert."""

    def __init__(self, cfg, seed=1):
        super().__init__({}, device="cpu", cache_gb=1.0)
        g = torch.Generator().manual_seed(seed)
        E, H, I = cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size
        self.w13 = [torch.randn(E, 2 * I, H, generator=g) * 0.2 for _ in range(cfg.num_hidden_layers)]
        self.w2 = [torch.randn(E, H, I, generator=g) * 0.2 for _ in range(cfg.num_hidden_layers)]

    def layer(self, layer):
        self.stats["full_layer_loads"] += 1
        return self.w13[layer], self.w2[layer]

    def expert(self, layer, e):
        self.stats["expert_misses"] += 1
        return self.w13[layer][e].clone(), self.w2[layer][e].clone()


def tiny_model(seed=0):
    cfg = tiny_config()
    cfg._attn_implementation = "eager"
    torch.manual_seed(seed)
    model = M.Inkling2LoopForCausalLM(cfg).eval()
    g = torch.Generator().manual_seed(seed)
    for n, p in model.named_parameters():
        if n.endswith("query") or "message" in n or "static_weight" in n or n.endswith("proj"):
            p.data.copy_(torch.randn(p.shape, generator=g) * 0.3)
    for i, layer in enumerate(model.model.layers):
        layer.mlp.experts.layer_idx = i
    model.requires_grad_(False)
    return model


def test_capture(tmp):
    model = tiny_model()
    src = TinySource(model.config)
    teacher = Teacher(device="cpu", model=model)
    teacher.provider = src
    ids = torch.randint(0, 97, (3, 20), generator=torch.Generator().manual_seed(3))
    acc, groups = capture_attn.capture(teacher, ids, list(range(8)), group=2, micro=1, chunk=7, log=None)
    out = os.path.join(tmp, "cap")
    capture_attn.write(out, acc, model.config, 3, 20, {})
    ok, rep = capture_attn.check(out)
    assert sum(g["reader_checks"] for g in groups) > 0
    assert ok, json.dumps(rep, indent=1)[:3000]
    # the Hessian is X^T X: compare one against a direct float64 product of the captured stream
    return rep


def test_capture_drops_pass3_fails(tmp):
    orig = capture_attn.Accumulator.add

    def mutant(self, layer, pas, stream, x):
        if pas == 3:
            return
        return orig(self, layer, pas, stream, x)

    capture_attn.Accumulator.add = mutant
    try:
        model = tiny_model()
        teacher = Teacher(device="cpu", model=model)
        teacher.provider = TinySource(model.config)
        ids = torch.randint(0, 97, (2, 20), generator=torch.Generator().manual_seed(3))
        acc, _ = capture_attn.capture(teacher, ids, list(range(8)), group=2, log=None)
        out = os.path.join(tmp, "cap_mutant")
        capture_attn.write(out, acc, model.config, 2, 20, {})
        ok, rep = capture_attn.check(out)
    finally:
        capture_attn.Accumulator.add = orig
    bad = [r for r in rep["tokens_x_visits"] if not r["ok"]]
    assert not ok and bad and all(2 <= r["layer"] <= 5 for r in bad), bad
    return len(bad)


def test_generation():
    res = {}
    for name, thresh in (("single experts", 10**9), ("full layers", -1)):
        model = tiny_model()
        src = TinySource(model.config)
        model_full.install_lazy(model, src)
        old = model_full.FULL_LAYER_TOKENS
        model_full.FULL_LAYER_TOKENS = thresh
        try:
            prompt = [int(x) for x in torch.randint(0, 97, (12,), generator=torch.Generator().manual_seed(5))]
            gen, steps = generate.greedy(model, prompt, 10)
            full = model(input_ids=torch.tensor([prompt + gen[:-1]]), use_cache=False).logits[0, len(prompt) - 1 :]
            fa = [int(x) for x in full.argmax(-1)]
            res[name] = {"equal": gen == fa, "max_abs": float((steps - full).abs().max()), "stats": dict(src.stats)}
            assert gen == fa, (name, gen, fa)
            assert float((steps - full).abs().max()) < 1e-4
        finally:
            model_full.FULL_LAYER_TOKENS = old
    assert res["single experts"]["stats"]["expert_misses"] > 0 and res["full layers"]["stats"]["full_layer_loads"] > 0
    return res


def test_generation_without_cache_fails():
    model = tiny_model()
    model_full.install_lazy(model, TinySource(model.config))
    prompt = [int(x) for x in torch.randint(0, 97, (12,), generator=torch.Generator().manual_seed(5))]
    gen, _ = generate.greedy(model, prompt, 10, use_cache=False)
    full = model(input_ids=torch.tensor([prompt + gen[:-1]]), use_cache=False).logits[0, len(prompt) - 1 :]
    fa = [int(x) for x in full.argmax(-1)]
    assert gen != fa, "a decode without cache passed the generation check"
    return sum(a != b for a, b in zip(gen, fa))


def test_substitution(tmp):
    model = tiny_model()
    src = TinySource(model.config)
    teacher = Teacher(device="cpu", model=model)
    teacher.provider = src
    ids = torch.randint(0, 97, (1, 20), generator=torch.Generator().manual_seed(4))
    base = teacher.run(ids, log=None)[0]
    name = "model.layers.3.self_attn.q_proj.linear.weight"
    p = dict(model.named_parameters())[name]
    d_eq, d_pert = os.path.join(tmp, "eq"), os.path.join(tmp, "pert")
    os.makedirs(d_eq), os.makedirs(d_pert)
    p.detach().numpy().tofile(os.path.join(d_eq, name + ".f32"))
    (p.detach() + 0.05 * torch.randn(p.shape)).numpy().tofile(os.path.join(d_pert, name + ".f32"))
    names = model_full.tensor_names(model)
    plan = model_full.tensor_plan(model_full.parse_student(f"model.layers.3.self_attn.q_proj.linear.*=dir:{d_eq}"), names)
    assert list(plan) == [name], plan
    model_full.apply_tensor_plan(model, plan, log=None)
    same = teacher.run(ids, log=None)[0]
    assert torch.equal(same, base)
    plan = model_full.tensor_plan(model_full.parse_student(f"{name}=dir:{d_pert}"), names)
    model_full.apply_tensor_plan(model, plan, log=None)
    moved = teacher.run(ids, log=None)[0]
    assert float((moved - base).abs().max()) > 0
    st = model_full.parse_student("experts:6-17=dir:/x;experts:8=native;model.norm.weight=native")
    assert model_full.expert_arm(st) == {L: "dir:/x" for L in range(6, 18) if L != 8}
    try:
        model_full.tensor_plan(model_full.parse_student("model.layers.*.mlp.experts.*=dir:/x"), names)
        raise AssertionError("experts accepted through a parameter pattern")
    except KeyError:
        pass
    return float((moved - base).abs().max())


def main():
    tmp = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="lot1tiny-")
    os.makedirs(tmp, exist_ok=True)
    results = {}
    for name, fn in (("capture", lambda: test_capture(tmp)),
                     ("capture_drops_pass3_fails", lambda: test_capture_drops_pass3_fails(tmp)),
                     ("generation", test_generation),
                     ("generation_without_cache_fails", test_generation_without_cache_fails),
                     ("substitution", lambda: test_substitution(tmp))):
        try:
            r = fn()
            results[name] = "PASS"
            print(f"PASS {name}: {json.dumps(r, default=str)[:600]}", flush=True)
        except Exception as e:  # noqa: BLE001
            results[name] = f"FAIL {type(e).__name__}: {e}"
            print(f"FAIL {name}: {type(e).__name__}: {str(e)[:2000]}", flush=True)
    n_fail = sum(1 for v in results.values() if v != "PASS")
    print(f"test result: {len(results) - n_fail} passed, {n_fail} failed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
