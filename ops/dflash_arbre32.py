# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx-lm==0.28.3", "mlx==0.32.4", "numpy==2.5.4", "safetensors==0.8.0", "tokenizers==0.23.3",
#                 "pyarrow==26.0.1", "transformers==5.22.1", "huggingface_hub", "llvq-tetra"]
# [tool.uv.sources]
# llvq-tetra = { path = "../llvq-tetra" }
# ///
"""DFlash tree acceptance at 15, 23 and 31 nodes on the sealed Tetra 4B, on the Mac, at $0 (2026-11-11).

    uv run ops/dflash_arbre32.py <packed_dir> <house.jsonl> <gsm.jsonl> <sp8_brut_dir> <out_dir> [n_gsm]

The question: is a verify pass of 32 rows worth building. The card loop (llvq_llm::dflash_spec, Tree(15)) builds a
tree of 15 nodes from the drafter's single block pass; this replays that construction offline at N in {15, 23, 31}
against the target's greedy text, the method of ops/dflash_alpha.py (dflash-alpha-2026-11-09, branch
claude/dflash-sass-2026-11-09), whose drafter forward this file copies.

Text. GSM8K problems and the house set's 20 math prompts (GSM8K rows 0-19): the card's own greedy text, the `pass`
decoder of SP8's dump gsm200.jsonl (768 tokens after `first` at most; the house runs stop at 256). The dump holds the
tokens after `first`; `first` is the MLX door's argmax at the prompt's end, checked by the teacher-forced agreement.
The house set's chat, code and rag prompts have no token dump from the card: their text is the MLX door's own greedy
decode, 256 tokens after `first`, stopped at <|im_end|> included, as specbench does.

Prompt ids: llvq_llm::chatfmt as specbench builds them (each message `<|im_start|>` role `\\n` body `<|im_end|>` `\\n`,
then the assistant's opening with an empty reasoning block); their lengths are checked against the card's log.

Tree: llvq_llm::dflash::draft_tree_scored's construction, top 4 a position, the first N nodes popped best first by
summed log probability. Accepted a pass: the longest prefix of the greedy tokens whose rank path is in the tree, at
most the tokens left; tokens a pass = accepted + 1. The card reports "accepted a pass" the same way (tokens over
passes, minus one).
"""
from __future__ import annotations

import gzip
import heapq
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "mlx_tetra"))

DRAFT_REPO = "z-lab/Qwen3-4B-DFlash-b16"
DRAFT_REV = "b74e3a329c4d963783143b1e970d95b002be72bd"
CTX_LAYERS = (2, 10, 18, 26, 34)
BLOCK = 16
MASK_ID = 151669
BATCH = 24
TOPK = 4
NODES = (15, 16, 23, 31)
HOUSE_BUDGET = 256
GSM_BUDGET = 768
IM_START, IM_END, NL, NL2, THINK, ENDTHINK = 151644, 151645, 198, 271, 151667, 151668


def silu(x):
    import mlx.core as mx

    return x * mx.sigmoid(x)


class Drafter:
    """DFlashDraftModel, f32 on the GPU (copied from ops/dflash_alpha.py)."""

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

    def drafts(self, emb, head_w, x: list[int], ms: list[int], ks, vs):
        """Top-4 ids and log probabilities [B, 15, 4] for the pending positions ms of x."""
        import mlx.core as mx

        b = len(ms)
        blk = np.full((b, BLOCK), MASK_ID, dtype=np.int32)
        blk[:, 0] = [x[m] for m in ms]
        h = emb[mx.array(blk)]
        msa = mx.array(np.asarray(ms, dtype=np.int32))
        pos = (msa[:, None] + mx.arange(BLOCK)[None, :])[:, None, :]
        t = ks[0].shape[1]
        cmask = mx.where(mx.arange(t)[None, :] < msa[:, None], 0.0, -mx.inf)[:, None, None, :]
        rep = self.hq // self.hk
        scale = self.dh ** -0.5
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
        out = mx.fast.rms_norm(h, self.norm, self.eps)[:, 1:, :]
        z = out @ head_w.T
        lse = mx.logsumexp(z, axis=-1, keepdims=True)
        ti = mx.argpartition(-z, kth=TOPK - 1, axis=-1)[..., :TOPK].astype(mx.int32)
        tl = mx.take_along_axis(z, ti, axis=-1) - lse
        mx.eval(ti, tl)
        return np.asarray(ti), np.asarray(tl).astype(np.float32)


def capture(model, x: list[int]):
    """One forward of the target over x: the inputs of CTX_LAYERS [T, 5H] f32, and the argmax a row."""
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
    hn = inner.norm(h)
    logits = inner.embed_tokens.as_linear(hn)
    f = mx.concatenate(feats, axis=-1)
    am = mx.argmax(logits[0], axis=-1)
    mx.eval(f, am)
    return f, am.tolist()


def tree_depths(x: list[int], p: int, ti: np.ndarray, tl: np.ndarray) -> dict[int, list[int]]:
    """Accepted depth at every pending position for each tree size in NODES (draft_tree_scored's construction)."""
    n = len(x)
    out = {nb: [] for nb in NODES}
    big = max(NODES)
    for r, m in enumerate(range(p, n - 1)):
        cand, lp = ti[r], tl[r]
        heap = [(-float(lp[0, k]), (k,)) for k in range(TOPK)]
        heapq.heapify(heap)
        rank_of = {}
        while heap and len(rank_of) < big:
            s, path = heapq.heappop(heap)
            rank_of[path] = len(rank_of)
            if len(path) < BLOCK - 1:
                d = len(path)
                for k in range(TOPK):
                    heapq.heappush(heap, (s - float(lp[d, k]), path + (k,)))
        idx, path, a = [], (), 0
        while a < BLOCK - 1 and m + 1 + a < n:
            hit = np.nonzero(cand[a] == x[m + 1 + a])[0]
            if not len(hit) or path + (int(hit[0]),) not in rank_of:
                break
            path = path + (int(hit[0]),)
            idx.append(rank_of[path])
            a += 1
        for nb in NODES:
            depth = 0
            while depth < len(idx) and idx[depth] < nb:
                depth += 1
            out[nb].append(depth)
    return out


def chain_depths(x: list[int], p: int, ti: np.ndarray) -> list[int]:
    n = len(x)
    out = []
    for r, m in enumerate(range(p, n - 1)):
        a = 0
        while a < BLOCK - 1 and m + 1 + a < n and int(ti[r, a, 0]) == x[m + 1 + a]:
            a += 1
        out.append(a)
    return out


def replay(x: list[int], p: int, acc: list[int]) -> list[int]:
    """Accepted drafts of every round of the loop over x[p:], a round at m emitting min(a + 1, left) tokens."""
    m, n, rounds = p, len(x), []
    while m < n - 1:
        e = min(acc[m - p] + 1, n - 1 - m)
        rounds.append(e - 1)
        m += e
    return rounds


def prompt_ids(tok, messages: list[tuple[str, str]]) -> list[int]:
    enc = lambda t: tok.encode(t, add_special_tokens=False).ids
    ids: list[int] = []
    for role, content in messages:
        ids += [IM_START] + enc(role) + [NL] + enc(content) + [IM_END, NL]
    return ids + [IM_START] + enc("assistant") + [NL, THINK, NL2, ENDTHINK, NL2]


def load_prompts(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        if "id" not in o:
            continue
        msgs = [("user", o["prompt"])] if "prompt" in o else [(m["role"], m["content"]) for m in o["messages"]]
        out.append({"id": o["id"], "kind": o["kind"], "messages": msgs})
    return out


def card_log(path: Path) -> dict[str, dict]:
    """Per prompt from a specbench log: prompt length, and per drafter (tokens, accepted a pass, passes)."""
    out: dict[str, dict] = {}
    cur = None
    for line in path.read_text().splitlines():
        m = re.match(r"== (\S+) \((\w+)\): prompt (\d+) tokens", line)
        if m:
            cur = out.setdefault(m.group(1), {"kind": m.group(2), "prompt": int(m.group(3))})
            continue
        m = re.match(r"\s*(dflash:\w+): (\d+) tokens, .*?; ([\d.]+) drafted tokens accepted a pass, (\d+) passes", line)
        if m and cur is not None:
            cur[m.group(1)] = (int(m.group(2)), float(m.group(3)), int(m.group(4)))
    return out


def main(argv: list[str]) -> int:
    import mlx.core as mx
    from huggingface_hub import hf_hub_download
    from mlx_lm.generate import generate_step
    from tokenizers import Tokenizer

    import llvq_mlx as L

    packed, house_p, gsm_p, brut, out_dir = argv[1], Path(argv[2]), Path(argv[3]), Path(argv[4]), Path(argv[5])
    n_gsm = int(argv[6]) if len(argv) > 6 else 200
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    model, _, _ = L.build(packed)
    qe = model.model.embed_tokens
    table = mx.dequantize(qe.weight, qe.scales, qe.biases, group_size=qe.group_size, bits=qe.bits).astype(mx.float32)
    mx.eval(table)
    tok = Tokenizer.from_file(str(Path(packed) / "tokenizer.json"))
    drafter = Drafter(hf_hub_download(DRAFT_REPO, "model.safetensors", revision=DRAFT_REV))
    print(f"target tetra ({packed}); drafter {DRAFT_REPO}@{DRAFT_REV[:8]}; nodes {NODES}; loaded in "
          f"{time.perf_counter() - t0:.1f} s", flush=True)

    dump = {}
    for line in (brut / "gsm200.jsonl").read_text().splitlines():
        o = json.loads(line)
        if o["decoder"] == "pass":
            dump[o["id"]] = o["tokens"]
    card_house, card_gsm = card_log(brut / "spec-house.txt"), card_log(brut / "spec-gsm200.txt")

    jobs = []  # (set, id, kind, budget, prompt record, card log)
    for r in load_prompts(house_p):
        jobs.append(("house", r["id"], r["kind"], HOUSE_BUDGET, r, card_house.get(r["id"], {})))
    for r in load_prompts(gsm_p)[:n_gsm]:
        jobs.append(("gsm", r["id"], "gsm8k", GSM_BUDGET, r, card_gsm.get(r["id"], {})))

    rows, raw = [], {}
    gen_cache = out_dir / "mac-greedy.json"
    mac_gen = json.loads(gen_cache.read_text()) if gen_cache.exists() else {}
    for set_name, pid, kind, budget, rec, card in jobs:
        ids = prompt_ids(tok, rec["messages"])
        if card.get("prompt") is not None and card["prompt"] != len(ids):
            raise SystemExit(f"{pid}: prompt {len(ids)} tokens here, {card['prompt']} on the card")
        if pid in dump:
            first = None
            for (t, _), _ in zip(generate_step(mx.array(ids), model, max_tokens=1), range(1)):
                first = int(t)
            gen = [first] + dump[pid][:budget]
            source = "card"
        else:
            if pid not in mac_gen:
                g = []
                for (t, _), _ in zip(generate_step(mx.array(ids), model, max_tokens=budget + 1), range(budget + 1)):
                    g.append(int(t))
                    if int(t) == IM_END and len(g) > 1:
                        break
                mac_gen[pid] = g
                gen_cache.write_text(json.dumps(mac_gen))
            gen = mac_gen[pid]
            source = "mac"
        x, p = ids + gen, len(ids)
        feats, am = capture(model, x)
        agree = sum(1 for i in range(p - 1, len(x) - 1) if am[i] == x[i + 1]) / (len(x) - p)
        first_div = next((i - p + 1 for i in range(p - 1, len(x) - 1) if am[i] != x[i + 1]), None)
        ks, vs = drafter.context(feats)
        ms = list(range(p, len(x) - 1))
        parts = [drafter.drafts(table, table, x, ms[i:i + BATCH], ks, vs) for i in range(0, len(ms), BATCH)]
        ti = np.concatenate([q[0] for q in parts])
        tl = np.concatenate([q[1] for q in parts])
        order = np.argsort(-tl, axis=-1, kind="stable")
        ti = np.take_along_axis(ti, order, axis=-1)
        tl = np.take_along_axis(tl, order, axis=-1)
        trees = tree_depths(x, p, ti, tl)
        chain = chain_depths(x, p, ti)
        line = {"set": set_name, "id": pid, "kind": kind, "text": source, "prompt": p, "tokens": len(gen) - 1,
                "first": gen[0], "tf_agree": round(agree, 4), "tf_first_div": first_div}
        raw[f"{set_name}/{pid}"] = {"chain15": chain, **{f"tree{nb}": trees[nb] for nb in NODES}}
        for name, acc in [("chain15", chain)] + [(f"tree{nb}", trees[nb]) for nb in NODES]:
            rr = replay(x, p, acc)
            line[name] = {"passes": len(rr), "accepted": sum(rr), "hist": np.bincount(rr, minlength=16).tolist()}
        for arm in ("dflash:tree", "dflash:chain"):
            if arm in card:
                line["card_" + arm.split(":")[1]] = {"tokens": card[arm][0], "acc": card[arm][1], "passes": card[arm][2]}
        rows.append(line)
        short = {k: (v if not isinstance(v, dict) or "hist" not in v else round(v["accepted"] / v["passes"], 3))
                 for k, v in line.items()}
        print(json.dumps(short), flush=True)
        del feats, ks, vs
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=1))
    (out_dir / "accept-depths.json.gz").write_bytes(gzip.compress(json.dumps(raw).encode(), mtime=0))
    print(f"done in {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
