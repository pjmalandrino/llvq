# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx-lm==0.28.3", "mlx==0.32.4", "numpy==2.5.4", "safetensors==0.8.0", "tokenizers==0.23.3",
#                 "pyarrow==26.0.1", "transformers==5.22.1", "huggingface_hub", "llvq-tetra"]
# [tool.uv.sources]
# llvq-tetra = { path = "../llvq-tetra" }
# ///
"""The fixture of the Rust DFlash drafter (llvq-llm/src/dflash.rs), from the MLX reference, on the Mac, at $0.

    uv run ops/dflash_fixture.py <packed_dir> <completions.json> <out.safetensors>

<packed_dir> is the sealed 4B through the MLX door (hfpack, e.g. ~/q4b-hf-2026-09-28); <completions.json> is the
eagle-tree greedy text the reference ran on (docs/mesures/dflash-alpha-2026-11-09-brut/completions.json). The
drafter forward below is copied from ops/dflash_alpha.py (branch of the dflash-alpha measurement, 2026-11-09), with
one change: it also returns the final normed hidden states and the raw logits. The weights are read from the
Hugging Face cache only (no download).

Two short prompts ("chat" and "gsm8k/33"), the prompt plus 20 greedy tokens, three rounds each at pending positions
p, p + 4, p + 9. For each prompt the file holds the token ids and the target's tap states at positions 0..p+8 (the
inputs of layers 2, 10, 18, 26, 34, concatenated in that order, f32); for each round the 15 drafted positions'
argmax, top 64 raw logits and their ids, logsumexp, final normed hidden state, and the tree of 15 nodes built best
first over the top 4 candidates a position (ops/dflash_alpha.py tree_accepts, its pop order) as (token, parent).
It also holds a few dequantized rows of the target's table, to check the Rust dequantization of the int4 g64 table.
"""
from __future__ import annotations

import heapq
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "mlx_tetra"))

DRAFT_REPO = "z-lab/Qwen3-4B-DFlash-b16"
DRAFT_REV = "b74e3a329c4d963783143b1e970d95b002be72bd"
CTX_LAYERS = (2, 10, 18, 26, 34)
BLOCK = 16
MASK_ID = 151669
TOPK = 4
TREE_NODES = 15
KEEP = 64
PROMPTS = ("chat", "gsm8k/33")
GEN = 20
ROUNDS = (0, 4, 9)


def silu(x):
    import mlx.core as mx

    return x * mx.sigmoid(x)


class Drafter:
    """DFlashDraftModel, f32 on the GPU; ops/dflash_alpha.py's Drafter with the hidden states returned."""

    def __init__(self, path: str):
        import mlx.core as mx

        w = mx.load(path)
        f = lambda k: w[k].astype(mx.float32)
        self.fc, self.hn, self.norm = f("fc.weight"), f("hidden_norm.weight"), f("norm.weight")
        names = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj",
                 "qn": "self_attn.q_norm", "kn": "self_attn.k_norm", "ln1": "input_layernorm",
                 "ln2": "post_attention_layernorm", "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}
        self.layers = [{s: f(f"layers.{i}.{n}.weight") for s, n in names.items()} for i in range(5)]
        assert len(w) == 3 + 5 * len(names), len(w)
        self.hq, self.hk, self.dh, self.eps, self.base = 32, 8, 128, 1e-6, 1e6
        mx.eval(self.fc, self.hn, self.norm, [list(l.values()) for l in self.layers])

    def rope(self, x, pos):
        import mlx.core as mx

        half = self.dh // 2
        inv = 1.0 / (self.base ** (mx.arange(0, self.dh, 2, dtype=mx.float32) / self.dh))
        ang = pos.astype(mx.float32)[..., None] * inv
        c, s = mx.cos(ang), mx.sin(ang)
        x1, x2 = x[..., :half], x[..., half:]
        return mx.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)

    def context(self, feats):
        import mlx.core as mx

        c = mx.fast.rms_norm(feats @ self.fc.T, self.hn, self.eps)
        t = c.shape[0]
        pos = mx.arange(t)
        ks, vs = [], []
        for l in self.layers:
            k = mx.fast.rms_norm((c @ l["k"].T).reshape(t, self.hk, self.dh), l["kn"], self.eps).transpose(1, 0, 2)
            ks.append(self.rope(k, pos))
            vs.append((c @ l["v"].T).reshape(t, self.hk, self.dh).transpose(1, 0, 2))
        mx.eval(ks, vs)
        return ks, vs

    def block(self, emb, head_w, x: list[int], m: int, ks, vs):
        """One pending position m: (final normed hidden [15, H], logits [15, V]), f32."""
        import mlx.core as mx

        blk = np.full((1, BLOCK), MASK_ID, dtype=np.int32)
        blk[:, 0] = x[m]
        h = emb[mx.array(blk)]
        msa = mx.array(np.asarray([m], dtype=np.int32))
        pos = (msa[:, None] + mx.arange(BLOCK)[None, :])[:, None, :]
        t = ks[0].shape[1]
        cmask = mx.where(mx.arange(t)[None, :] < msa[:, None], 0.0, -mx.inf)[:, None, None, :]
        rep = self.hq // self.hk
        scale = self.dh ** -0.5
        b = 1
        for li, l in enumerate(self.layers):
            hn = mx.fast.rms_norm(h, l["ln1"], self.eps)
            q = mx.fast.rms_norm((hn @ l["q"].T).reshape(b, BLOCK, self.hq, self.dh), l["qn"], self.eps)
            q = self.rope(q.transpose(0, 2, 1, 3), pos)
            kb = mx.fast.rms_norm((hn @ l["k"].T).reshape(b, BLOCK, self.hk, self.dh), l["kn"], self.eps)
            kb = mx.repeat(self.rope(kb.transpose(0, 2, 1, 3), pos), rep, axis=1)
            vb = mx.repeat((hn @ l["v"].T).reshape(b, BLOCK, self.hk, self.dh).transpose(0, 2, 1, 3), rep, axis=1)
            kc = mx.repeat(ks[li], rep, axis=0)[None]
            vc = mx.repeat(vs[li], rep, axis=0)[None]
            sc = (q @ kc.transpose(0, 1, 3, 2)) * scale + cmask
            sb = (q @ kb.transpose(0, 1, 3, 2)) * scale
            p = mx.softmax(mx.concatenate([sc, sb], axis=-1), axis=-1, precise=True)
            att = p[..., :t] @ vc + p[..., t:] @ vb
            h = h + att.transpose(0, 2, 1, 3).reshape(b, BLOCK, self.hq * self.dh) @ l["o"].T
            m2 = mx.fast.rms_norm(h, l["ln2"], self.eps)
            h = h + (silu(m2 @ l["gate"].T) * (m2 @ l["up"].T)) @ l["down"].T
        out = mx.fast.rms_norm(h, self.norm, self.eps)[0, 1:, :]
        z = out @ head_w.T
        mx.eval(out, z)
        return np.asarray(out), np.asarray(z)


def capture(model, x: list[int]):
    """ops/dflash_alpha.py capture: the inputs of CTX_LAYERS concatenated [T, 5H] f32."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner = model.model
    h = inner.embed_tokens(mx.array(x)[None])
    mask = create_attention_mask(h, None)
    feats = []
    for idx, layer in enumerate(inner.layers):
        if idx in CTX_LAYERS:
            feats.append(h[0].astype(mx.float32))
        h = layer(h, mask, None)
    f = mx.concatenate(feats, axis=-1)
    mx.eval(f)
    return np.asarray(f)


def tree(ti: np.ndarray, tl: np.ndarray, nodes: int):
    """ops/dflash_alpha.py tree_accepts, its pop order kept: nodes (token, parent), parent -1 = the root x_m."""
    heap = [(-float(tl[0, k]), (k,)) for k in range(TOPK)]
    heapq.heapify(heap)
    rank_of, toks, parents = {}, [], []
    while heap and len(rank_of) < nodes:
        s, path = heapq.heappop(heap)
        rank_of[path] = len(rank_of)
        toks.append(int(ti[len(path) - 1, path[-1]]))
        parents.append(rank_of[path[:-1]] if len(path) > 1 else -1)
        if len(path) < BLOCK - 1:
            d = len(path)
            for k in range(TOPK):
                heapq.heappush(heap, (s - float(tl[d, k]), path + (k,)))
    return np.asarray(toks, np.int32), np.asarray(parents, np.int32)


def main(argv: list[str]) -> int:
    import mlx.core as mx
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import save_file

    import llvq_mlx as L

    packed, comp_path, out_path = argv[1], Path(argv[2]), Path(argv[3])
    model, _, _ = L.build(packed)
    qe = model.model.embed_tokens
    table = mx.dequantize(qe.weight, qe.scales, qe.biases, group_size=qe.group_size, bits=qe.bits).astype(mx.float32)
    mx.eval(table)
    drafter = Drafter(hf_hub_download(DRAFT_REPO, "model.safetensors", revision=DRAFT_REV, local_files_only=True))
    comp = json.loads(comp_path.read_text())
    t: dict[str, np.ndarray] = {}
    rows = {MASK_ID}
    for i, name in enumerate(PROMPTS):
        e = comp[name]
        x = e["prompt"] + e["gen"][:GEN]
        p = len(e["prompt"])
        ms = [p + r for r in ROUNDS]
        feats = capture(model, x)[: max(ms)]
        t[f"p{i}.ids"] = np.asarray(x, np.int32)
        t[f"p{i}.taps"] = np.ascontiguousarray(feats.astype(np.float32))
        ks, vs = drafter.context(mx.array(feats))
        for j, m in enumerate(ms):
            # the context mask of the reference keeps positions < m only
            out, z = drafter.block(table, table, x, m, ks, vs)
            lse = np.log(np.exp(z.astype(np.float64) - z.max(-1, keepdims=True)).sum(-1)) + z.max(-1)
            order = np.argsort(-z, axis=-1, kind="stable")[:, :KEEP]
            top = np.take_along_axis(z, order, axis=-1)
            tl = top[:, :TOPK] - lse[:, None].astype(np.float32)
            tt, tp = tree(order[:, :TOPK], tl, TREE_NODES)
            k = f"p{i}.r{j}"
            t[f"{k}.m"] = np.asarray([m], np.int32)
            t[f"{k}.argmax"] = np.argmax(z, axis=-1).astype(np.int32)
            t[f"{k}.top_ids"] = order.astype(np.int32)
            t[f"{k}.top_logits"] = np.ascontiguousarray(top.astype(np.float32))
            t[f"{k}.lse"] = lse.astype(np.float32)
            t[f"{k}.out"] = np.ascontiguousarray(out.astype(np.float32))
            t[f"{k}.tree_tok"] = tt
            t[f"{k}.tree_parent"] = tp
            rows.update(int(v) for v in order[:, 0])
            rows.add(x[m])
            print(f"{name} m={m}: drafts {t[f'{k}.argmax'].tolist()}", flush=True)
    ids = np.asarray(sorted(rows), np.int32)
    t["table.ids"] = ids
    t["table.rows"] = np.ascontiguousarray(np.asarray(table[mx.array(ids)]).astype(np.float32))
    save_file(t, str(out_path), metadata={"drafter": f"{DRAFT_REPO}@{DRAFT_REV}", "packed": packed,
                                          "prompts": ",".join(PROMPTS), "gen": str(GEN),
                                          "rounds": ",".join(map(str, ROUNDS))})
    print(f"wrote {out_path} ({len(t)} tensors)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
