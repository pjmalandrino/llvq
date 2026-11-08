# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx-lm==0.28.3", "mlx==0.32.4", "numpy==2.5.4", "safetensors==0.8.0", "tokenizers==0.23.3",
#                 "pyarrow==26.0.1", "transformers==5.22.1", "huggingface_hub", "llvq-tetra"]
# [tool.uv.sources]
# llvq-tetra = { path = "../llvq-tetra" }
# ///
"""EAGLE-3 acceptance on the sealed Tetra 4B, on the Mac, at $0 (2026-11-08, prereg spec-final section 6 D).

    uv run ops/eagle_alpha.py <packed_dir> [n_gsm8k=50]

The target is the sealed 4B through the MLX door (`ops/mlx_tetra`, its hfpack): its greedy completions, then one
forward over prompt and completion for the hidden states EAGLE-3 reads, the inputs of layers 2, 18 and 33 (the
EAGLE-3 convention, HF `hidden_states` indices [2, L/2, L-3]). The head is AngelSlim/Qwen3-4B_eagle3 at a pinned
revision (Apache-2.0), SpecForge's layout: fc over the three states, one Llama layer whose attention reads the
normed token embedding and the normed state side by side, a 32,000-token head mapped to the target's vocabulary
by d2t.

Greedy decoding makes acceptance a function of the greedy text: a round whose pending token is x[m] drafts from the
head's cache over positions 0..m-1, built from the target's states (what EAGLE-3 holds after a verify), then its own
steps. Draft j is kept when it equals x[m + j] and every draft before it was kept. The rounds are replayed as the
loop runs them, for k = 1, 2, 3, and the tokens a verify are reported with the per-token alpha they imply.

What this does not measure: the card, the head's cost, the MLX door's tokens against the CUDA door's (they part at
token 243 of the house prompt, configs/mlx), a quantized head.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "mlx_tetra"))

HEAD_REPO = "AngelSlim/Qwen3-4B_eagle3"
HEAD_REV = "fd331e59626c8e95c392381a16ee59d518727fbb"
AUX = (2, 18, 33)
GSM8K_REV = "740312add88f781978c0658806c59bc2815b9866"
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
SAMPLE_SEED = 0x0065_736D_386B
M64 = (1 << 64) - 1


def splitmix_select(n: int, limit: int) -> list[int]:
    """`llvq_llm::gsm8k::select`: a seeded Fisher-Yates, the first `limit`, sorted."""
    if limit >= n:
        return list(range(n))
    state = SAMPLE_SEED
    idx = list(range(n))

    def nxt() -> int:
        nonlocal state
        state = (state + 0x9E3779B97F4A7C15) & M64
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)

    for i in range(n - 1, 0, -1):
        j = nxt() % (i + 1)
        idx[i], idx[j] = idx[j], idx[i]
    return sorted(idx[:limit])


def specrun_prompts() -> dict[str, str]:
    """The five chat prompts of `llvq-llm/src/bin/specrun.rs`, read from its source."""
    src = (HERE.parent / "llvq-llm/src/bin/specrun.rs").read_text()

    def const(name: str) -> str:
        m = re.search(r'const %s: &str = "(.*?)";' % name, src, re.S)
        return m.group(1).replace("\\\n", "").replace('\\"', '"')

    return {
        "chat": const("CHAT"),
        "code": const("CODE"),
        "summary": const("SUMMARY") + "\n\n" + const("SUMMARY_TEXT"),
        "edit": const("EDIT") + "\n\n" + const("EDIT_CODE"),
        "rag": const("RAG") + "\n\n" + const("SUMMARY_TEXT"),
    }


class Head:
    """SpecForge's Eagle3 draft model, f32 on the GPU."""

    def __init__(self, path: str):
        import mlx.core as mx

        w = mx.load(path)
        self.d2t = w["d2t"].astype(mx.int32)
        f = lambda k: w[k].astype(mx.float32)
        self.fc = f("fc.weight")
        self.lm = f("lm_head.weight")
        self.norm = f("norm.weight")
        p = "midlayer."
        self.hn, self.inl, self.post = f(p + "hidden_norm.weight"), f(p + "input_layernorm.weight"), f(p + "post_attention_layernorm.weight")
        self.q, self.k, self.v, self.o = (f(p + "self_attn.%s_proj.weight" % n) for n in "qkvo")
        self.gate, self.up, self.down = (f(p + "mlp.%s_proj.weight" % n) for n in ("gate", "up", "down"))
        self.hq, self.hk, self.dh = 32, 8, 128
        self.eps = 1e-6

    def rope(self, x, offset: int):
        import mlx.core as mx

        return mx.fast.rope(x, self.dh, traditional=False, base=1e6, scale=1.0, offset=offset)

    def qkv(self, emb, hid, offset: int):
        """emb, hid: [T, H]. Returns the residual, q [Hq, T, D], k and v [Hk, T, D]."""
        import mlx.core as mx

        rms = lambda x, g: mx.fast.rms_norm(x, g, self.eps)
        x = mx.concatenate([rms(emb, self.inl), rms(hid, self.hn)], axis=-1)
        t = x.shape[0]
        q = (x @ self.q.T).reshape(t, self.hq, self.dh).transpose(1, 0, 2)
        k = (x @ self.k.T).reshape(t, self.hk, self.dh).transpose(1, 0, 2)
        v = (x @ self.v.T).reshape(t, self.hk, self.dh).transpose(1, 0, 2)
        return hid, self.rope(q, offset), self.rope(k, offset), v

    def finish(self, resid, ctx):
        """ctx [Hq, T, D] -> the layer's output [T, H]."""
        import mlx.core as mx

        t = ctx.shape[1]
        h = resid + ctx.transpose(1, 0, 2).reshape(t, self.hq * self.dh) @ self.o.T
        m = mx.fast.rms_norm(h, self.post, self.eps)
        return h + (nn_silu(m @ self.gate.T) * (m @ self.up.T)) @ self.down.T

    def tokens(self, out):
        """The draft token of each row, in the target's vocabulary."""
        import mlx.core as mx

        d = mx.argmax(mx.fast.rms_norm(out, self.norm, self.eps) @ self.lm.T, axis=-1).astype(mx.int32)
        return d + self.d2t[d]

    def attend(self, q, kb, vb, extra_k, extra_v):
        """Row i attends to base keys 0..i and to its own extra keys (one a step past the first)."""
        import mlx.core as mx

        rep = self.hq // self.hk
        kb, vb = mx.repeat(kb, rep, axis=0), mx.repeat(vb, rep, axis=0)
        t = q.shape[1]
        scale = self.dh ** -0.5
        s = (q @ kb.transpose(0, 2, 1)) * scale
        causal = mx.triu(mx.full((t, t), -mx.inf, dtype=mx.float32), k=1)
        cols = [s + causal]
        ek = [mx.repeat(k, rep, axis=0) for k in extra_k]
        ev = [mx.repeat(v, rep, axis=0) for v in extra_v]
        cols += [(q * k).sum(-1, keepdims=True) * scale for k in ek]
        p = mx.softmax(mx.concatenate(cols, axis=-1), axis=-1, precise=True)
        out = p[..., :t] @ vb
        for j, v in enumerate(ev):
            out = out + p[..., t + j : t + j + 1] * v
        return out


def nn_silu(x):
    import mlx.core as mx

    return x * mx.sigmoid(x)


def target_states(model, x: list[int]):
    """One forward of the target over x: the inputs of the AUX layers [T, H] each, and the greedy argmax a row."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner = model.model
    h = inner.embed_tokens(mx.array(x)[None])
    mask = create_attention_mask(h, None)
    aux = []
    for idx, layer in enumerate(inner.layers):
        if idx in AUX:
            aux.append(h[0].astype(mx.float32))
        h = layer(h, mask, None)
    logits = model.lm_head(inner.norm(h)) if hasattr(model, "lm_head") else inner.embed_tokens.as_linear(inner.norm(h))
    return aux, mx.argmax(logits[0], axis=-1)


def drafts(head: Head, model, x: list[int], aux):
    """Draft tokens 1..3 for every row i: d_j[i] proposes x[i + 1 + j] from the state at i and token x[i + 1]."""
    import mlx.core as mx

    t = len(x) - 1
    emb = model.model.embed_tokens(mx.array(x[1:])).astype(mx.float32)
    fused = mx.concatenate([a[:t] for a in aux], axis=-1) @ head.fc.T
    r1, q1, k1, v1 = head.qkv(emb, fused, 0)
    o1 = head.finish(r1, head.attend(q1, k1, v1, [], []))
    d1 = head.tokens(o1)
    e2 = model.model.embed_tokens(d1).astype(mx.float32)
    r2, q2, k2, v2 = head.qkv(e2, o1, 1)
    o2 = head.finish(r2, head.attend(q2, k1, v1, [k2], [v2]))
    d2 = head.tokens(o2)
    e3 = model.model.embed_tokens(d2).astype(mx.float32)
    r3, q3, k3, v3 = head.qkv(e3, o2, 2)
    o3 = head.finish(r3, head.attend(q3, k1, v1, [k2, k3], [v2, v3]))
    d3 = head.tokens(o3)
    mx.eval(d1, d2, d3)
    return [d.tolist() for d in (d1, d2, d3)]


def alpha_of(tpv: float, k: int) -> float:
    """The per-token alpha whose geometric round of k drafts emits tpv tokens: (1 - a^(k+1)) / (1 - a) = tpv."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        a = (lo + hi) / 2
        t = k + 1 if a >= 1 else (1 - a ** (k + 1)) / (1 - a)
        lo, hi = (a, hi) if t < tpv else (lo, a)
    return (lo + hi) / 2


def replay(x: list[int], p: int, d: list[list[int]], k: int) -> tuple[int, int]:
    """The rounds of a k-draft loop over the completion x[p:]: (rounds, tokens emitted after x[p])."""
    m, rounds, n = p, 0, len(x)
    while m < n - 1:
        i = m - 1
        a = 0
        while a < k and m + 1 + a < n and d[a][i] == x[m + 1 + a]:
            a += 1
        e = min(a + 1, n - 1 - m)
        rounds += 1
        m += e
    return rounds, n - 1 - p


def main(argv: list[str]) -> int:
    import mlx.core as mx
    from huggingface_hub import hf_hub_download
    from mlx_lm.generate import generate_step

    packed = argv[1]
    n_gsm = int(argv[2]) if len(argv) > 2 else 50
    import llvq_mlx as L

    t0 = time.perf_counter()
    model, _, _ = L.build(packed)
    fmt = L.ChatFormat(packed)
    hp = hf_hub_download(HEAD_REPO, "model.safetensors", revision=HEAD_REV)
    head = Head(hp)
    print(f"target: the sealed 4B through the MLX door ({packed}); head {HEAD_REPO}@{HEAD_REV[:8]}; "
          f"aux layers {AUX}; loaded in {time.perf_counter() - t0:.1f} s")

    work: list[tuple[str, str, int]] = [(n, m, 256) for n, m in specrun_prompts().items()]
    if n_gsm > 0:
        import pyarrow.parquet as pq

        gp = hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset", revision=GSM8K_REV)
        qs = pq.read_table(gp).column("question").to_pylist()
        for j in splitmix_select(len(qs), n_gsm):
            work.append((f"gsm8k/{j}", qs[j].strip() + "\n" + INSTRUCTION, 1024))

    stops = (fmt.im_end, fmt.eot)
    tot: dict[str, list[int]] = {}
    rows = []
    for name, msg, max_new in work:
        ids = fmt.turn(msg)
        gen = []
        for (tid, _), _ in zip(generate_step(mx.array(ids), model, max_tokens=max_new), range(max_new)):
            tid = int(tid)
            if tid in stops:
                break
            gen.append(tid)
        x = ids + gen
        p = len(ids)
        aux, am = target_states(model, x)
        tf = am.tolist()
        agree = sum(1 for i in range(p - 1, len(x) - 1) if tf[i] == x[i + 1]) / max(1, len(x) - p)
        d = drafts(head, model, x, aux)
        # conditional acceptance over every pending position of the completion
        cond = []
        for j in range(3):
            ok = tried = 0
            for m in range(p, len(x) - 1 - j):
                i = m - 1
                if all(d[q][i] == x[m + 1 + q] for q in range(j)):
                    tried += 1
                    ok += d[j][i] == x[m + 1 + j]
            cond.append(ok / tried if tried else float("nan"))
        line = {"name": name, "prompt": p, "tokens": len(gen), "teacher_forced_agree": agree, "cond": cond}
        for k in (1, 2, 3):
            r, e = replay(x, p, d, k)
            line[f"k{k}"] = (r, e)
            key = "gsm8k" if name.startswith("gsm8k") else name
            acc = tot.setdefault(f"{key}/k{k}", [0, 0])
            acc[0] += r
            acc[1] += e
        rows.append(line)
        if not name.startswith("gsm8k"):
            print(json.dumps(line))
    print(f"\nper workload: tokens a verify (rounds) and the alpha it implies, k = 1 / 2 / 3; "
          f"GSM8K over {n_gsm} problems")
    for key in ["chat", "code", "summary", "edit", "rag", "gsm8k"]:
        cells = []
        for k in (1, 2, 3):
            r, e = tot.get(f"{key}/k{k}", [0, 0])
            if r == 0:
                continue
            tpv = e / r  # tokens emitted a round, the target's own token included
            cells.append(f"{tpv:.2f} (a {alpha_of(tpv, k):.2f})")
        print(f"  {key:8s} " + "   ".join(cells))
    gs = [r for r in rows if r["name"].startswith("gsm8k")]
    if gs:
        ag = sum(r["teacher_forced_agree"] for r in gs) / len(gs)
        c = [sum(r["cond"][j] for r in gs) / len(gs) for j in range(3)]
        print(f"  gsm8k teacher-forced agreement {ag:.4f}; conditional acceptance of drafts 1/2/3 {c[0]:.3f} / {c[1]:.3f} / {c[2]:.3f}")
    print(f"done in {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
