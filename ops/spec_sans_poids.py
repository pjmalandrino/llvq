# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx-lm==0.28.3", "mlx==0.32.4", "numpy==2.5.4", "safetensors==0.8.0", "tokenizers==0.23.3",
#                 "pyarrow==26.0.1", "transformers==5.22.1", "huggingface_hub", "llvq-tetra"]
# [tool.uv.sources]
# llvq-tetra = { path = "../llvq-tetra" }
# ///
"""Weightless drafters on a 16-row tree verify pass of the sealed Tetra 4B, on the Mac, at $0 (2026-11-11).

    uv run ops/spec_sans_poids.py <packed_dir> <house.jsonl> <gsm.jsonl> <sp8_brut_dir> <arbre32_brut_dir> \
        <out_dir> --mode search|frozen [--gsm LO:HI] [--house] [--rec SHAPE] [--comb SHAPE:Q]

Drafters with no weights, fed only by the text so far and by the target's logits on the rows of earlier verify passes:
  rec   token recycling (arXiv 2408.08696): a table [vocab][k] of the target's top-k next tokens, row t written from
        the last verify row whose token was t, every row of every pass, accepted or not. Tree of 15 grown best first
        from the root by a static rank priority p[r] (child of rank r scores parent * p[r]).
  look  prompt lookup: the longest n-gram (8 down to 1) suffix of prompt + text found earlier, most recent
        occurrence, its continuation as a chain of up to 15.
  comb  the lookup chain as the spine when its match is at least 2 tokens (spine child scores parent * q), the other
        nodes from the recycling table by p, best first, 15 nodes in all.
Tables are cold at each prompt or warm across the prompts of a set, in file order (causal).

Exact verify rows. A tree node's row must be the target conditioned on its path. Rows on the text (path = the greedy
continuation) are the teacher-forced rows of one forward over prompt + text. A node off the text has a canonical key
(j, tail): j the position of its first token off the text, tail its tokens from there; its row is the target over
text[:j] + tail. Those rows are computed here by a tree forward of the target (the same layers, dense f16 weights
decoded once from the sealed words, the prefix K/V of the teacher-forced pass, per-row positions and an
ancestor mask), cached per prompt by key, and shared across the arms. Gate at the first prompt: the tree forward on
text rows against the teacher-forced rows, and on a few off-text keys against a fresh forward over text[:j] + tail.

Text, prompts, `first`, replay convention: those of ops/dflash_arbre32.py (accepted a pass = tokens over passes
minus one; a pass at m emits min(a + 1, left)).
"""
from __future__ import annotations

import argparse
import gzip
import heapq
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "mlx_tetra"))
import dflash_arbre32 as A  # noqa: E402

VOCAB = 151936
KMAX = 8
NODES = 15
CHUNK_ROWS = 192
SHAPES = {  # static rank priorities p[r], r = 0..7
    "deep": (0.70, 0.12, 0.06, 0.04, 0.03, 0.02, 0.015, 0.01),
    "mid": (0.55, 0.18, 0.09, 0.06, 0.04, 0.03, 0.02, 0.015),
    "wide": (0.45, 0.22, 0.13, 0.09, 0.06, 0.045, 0.035, 0.03),
    "flat": (0.35, 0.22, 0.16, 0.12, 0.09, 0.07, 0.06, 0.05),
}


# ---------------------------------------------------------------- drafters

def lookup(arr: np.ndarray, m: int, nmax: int = 8, lmax: int = NODES) -> tuple[list[int], int]:
    """Longest suffix of arr[:m+1] (nmax down to 1) seen before; its most recent continuation, up to lmax."""
    L = m + 1
    ctx = arr[:L]
    for n in range(min(nmax, L - 1), 0, -1):
        if L - 1 - n + 1 <= 0:
            continue
        win = np.lib.stride_tricks.sliding_window_view(ctx[:L - 1], n)
        hit = np.nonzero((win == ctx[L - n:]).all(axis=1))[0]
        if len(hit):
            i = int(hit[-1])
            return ctx[i + n:min(i + n + lmax, L)].tolist(), n
    return [], 0


class Arm:
    def __init__(self, name: str, kind: str, k: int = 8, warm: bool = False, shape: str = "mid", q: float = 0.0):
        self.name, self.kind, self.k, self.warm, self.shape, self.q = name, kind, k, warm, shape, q
        self.p = SHAPES[shape][:k]
        self.table = np.full((VOCAB, k), -1, dtype=np.int32) if kind != "look" else None

    def start(self):
        if self.table is not None and not self.warm:
            self.table.fill(-1)

    def tree(self, arr: np.ndarray, m: int) -> list[tuple[int, int]]:
        """Nodes (token, parent index), root first (token arr[m], parent -1), at most 1 + NODES."""
        root = int(arr[m])
        if self.kind == "look":
            chain, _ = lookup(arr, m)
            return [(root, -1)] + [(t, i) for i, t in enumerate(chain)]
        chain: list[int] = []
        if self.kind == "comb":
            c, n = lookup(arr, m)
            if n >= 2:
                chain = c
        nodes = [(root, -1)]
        kids: list[set[int]] = [set()]
        heap: list = []
        seq = 0

        def push(i: int, score: float, spine: int):
            nonlocal seq
            if spine >= 0 and spine < len(chain):
                heapq.heappush(heap, (-score * self.q, seq, i, chain[spine], spine + 1))
                seq += 1
            row = self.table[nodes[i][0]]
            for r in range(self.k):
                t = int(row[r])
                if t < 0:
                    break
                heapq.heappush(heap, (-score * self.p[r], seq, i, t, -1))
                seq += 1

        push(0, 1.0, 0 if chain else -1)
        while heap and len(nodes) < 1 + NODES:
            s, _, par, t, sp = heapq.heappop(heap)
            if t in kids[par]:
                continue
            kids[par].add(t)
            nodes.append((t, par))
            kids.append(set())
            push(len(nodes) - 1, -s, sp)
        return nodes

    def update(self, nodes, tops):
        for (t, _), top in zip(nodes, tops):
            self.table[t] = top[:self.k]


# ---------------------------------------------------------------- target

def patch_dense(L, mx):
    """Every Tetra projection, for 2 rows or more, as x_rot @ W^T with W decoded once to f16 and kept."""
    cache = {}
    orig = L.TetraLinear.__call__

    def call(self, x):
        lead = x.shape[:-1]
        if math.prod(lead) < 2:
            return orig(self, x)
        xr = self.rot(x).reshape(-1, self.d_in).astype(mx.float16)
        w = cache.get(id(self))
        if w is None:
            w = self.dense(xr)
            mx.eval(w)
            cache[id(self)] = w
        return (xr @ w.T).reshape(*lead, self.d_out)

    L.TetraLinear.__call__ = call
    return cache


class Target:
    def __init__(self, model, mx):
        self.model, self.mx = model, mx
        a = model.model.layers[0].self_attn
        self.hq, self.hk, self.dh, self.scale = a.n_heads, a.n_kv_heads, 128, a.scale
        self.base = 1e6
        self.inv = mx.array(1.0 / (self.base ** (np.arange(0, self.dh, 2, dtype=np.float64) / self.dh)),
                            dtype=mx.float32)

    def top8(self, logits):
        mx = self.mx
        z = logits.astype(mx.float32)
        idx = mx.argpartition(-z, kth=KMAX - 1, axis=-1)[..., :KMAX]
        val = mx.take_along_axis(z, idx, axis=-1)
        order = mx.argsort(-val, axis=-1)
        out = mx.take_along_axis(idx, order, axis=-1).astype(mx.int32)
        mx.eval(out)
        return np.array(out)

    def capture(self, x: list[int]):
        """Teacher-forced pass over x: per-layer K (roped) and V [1, hk, T, dh], and the top 8 of every row."""
        mx = self.mx
        from mlx_lm.models.cache import make_prompt_cache

        cache = make_prompt_cache(self.model)
        hn = self.model.model(mx.array(x)[None], cache)
        T = len(x)
        kv = [(c.keys[:, :, :T, :], c.values[:, :, :T, :]) for c in cache]
        mx.eval(hn, kv)
        tops = []
        for i in range(0, T, 256):
            tops.append(self.top8(self.model.model.embed_tokens.as_linear(hn[0, i:i + 256])))
        return kv, np.concatenate(tops)

    def rope(self, x, cos, sin):
        mx = self.mx
        half = self.dh // 2
        xf = x.astype(mx.float32)
        x1, x2 = xf[..., :half], xf[..., half:]
        return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)

    def rows(self, kv, toks: list[int], pos: list[int], plen: list[int], anc: list[list[int]]):
        """Top 8 of rows: token toks[i] at position pos[i], attending to the prefix [0, plen[i]) and to rows anc[i]."""
        mx = self.mx
        R0 = len(toks)
        if R0 == 1:  # keep the multi-row path
            toks, pos, plen, anc = toks * 2, pos * 2, plen * 2, anc + [[1]]
        R = len(toks)
        T = max(plen)
        mask = np.zeros((R, T + R), dtype=bool)
        for i in range(R):
            mask[i, :plen[i]] = True
            mask[i, [T + a for a in anc[i]]] = True
        maskm = mx.array(mask)[None, None]
        ang = mx.array(np.asarray(pos, dtype=np.float32))[:, None] * self.inv[None, :]
        cos, sin = mx.cos(ang)[None, None], mx.sin(ang)[None, None]
        model = self.model
        h = model.model.embed_tokens(mx.array(toks)[None])
        for li, layer in enumerate(model.model.layers):
            at = layer.self_attn
            xn = layer.input_layernorm(h)
            q, k, v = at.q_proj(xn), at.k_proj(xn), at.v_proj(xn)
            q = at.q_norm(q.reshape(1, R, self.hq, self.dh)).transpose(0, 2, 1, 3)
            k = at.k_norm(k.reshape(1, R, self.hk, self.dh)).transpose(0, 2, 1, 3)
            v = v.reshape(1, R, self.hk, self.dh).transpose(0, 2, 1, 3)
            q, k = self.rope(q, cos, sin), self.rope(k, cos, sin)
            K = mx.concatenate([kv[li][0][:, :, :T], k.astype(kv[li][0].dtype)], axis=2)
            V = mx.concatenate([kv[li][1][:, :, :T], v.astype(kv[li][1].dtype)], axis=2)
            o = mx.fast.scaled_dot_product_attention(q, K, V, scale=self.scale, mask=maskm)
            h = h + at.o_proj(o.transpose(0, 2, 1, 3).reshape(1, R, -1))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        hn = model.model.norm(h)
        return self.top8(model.model.embed_tokens.as_linear(hn[0]))[:R0]

    def keys(self, kv, keys: list[tuple[int, tuple]]) -> dict:
        """Rows for canonical keys (j, tail), closed under prefix by the caller; chunked without splitting a j."""
        out = {}
        byj: dict[int, list] = {}
        for key in keys:
            byj.setdefault(key[0], []).append(key)
        groups = [sorted(g, key=lambda t: len(t[1])) for g in byj.values()]
        chunk: list = []
        for g in groups + [None]:
            if g is None or (chunk and len(chunk) + len(g) > CHUNK_ROWS):
                if chunk:
                    at = {key: i for i, key in enumerate(chunk)}
                    toks, pos, plen, anc = [], [], [], []
                    for j, tail in chunk:
                        toks.append(tail[-1])
                        pos.append(j + len(tail) - 1)
                        plen.append(j)
                        anc.append([at[(j, tail[:l])] for l in range(1, len(tail) + 1)])
                    tops = self.rows(kv, toks, pos, plen, anc)
                    for key, top in zip(chunk, tops):
                        out[key] = top
                chunk = []
            if g is not None:
                chunk.extend(g)
        return out


# ---------------------------------------------------------------- loop

def node_keys(nodes, x, m):
    """Per node: ('tf', row) if its path is the text, else ('key', (j, tail))."""
    n = len(x)
    paths = [()]
    out = [("tf", m)]
    for t, par in nodes[1:]:
        path = paths[par] + (t,)
        paths.append(path)
        r = 0
        while r < len(path) and m + 1 + r < n and path[r] == x[m + 1 + r]:
            r += 1
        out.append(("tf", m + len(path)) if r == len(path) else ("key", (m + 1 + r, path[r:])))
    return out


def accept(nodes, x, m) -> int:
    kids: dict[int, dict[int, int]] = {}
    for i, (t, par) in enumerate(nodes[1:], start=1):
        kids.setdefault(par, {}).setdefault(t, i)
    cur, a, n = 0, 0, len(x)
    while a < NODES and m + 1 + a < n:
        c = kids.get(cur, {}).get(x[m + 1 + a])
        if c is None:
            break
        cur, a = c, a + 1
    return a


def run_prompt(tg, arms, x, p, stats):
    kv, tf = tg.capture(x)
    arr = np.asarray(x, dtype=np.int64)
    n = len(x)
    cache: dict = {}
    state = {a.name: {"m": p, "rounds": []} for a in arms}
    for a in arms:
        a.start()
    while True:
        live = [a for a in arms if state[a.name]["m"] < n - 1]
        if not live:
            break
        trees, need = {}, set()
        for a in live:
            m = state[a.name]["m"]
            nodes = a.tree(arr, m)
            ks = node_keys(nodes, x, m) if a.kind != "look" else None
            trees[a.name] = (nodes, ks)
            if ks:
                for kind, key in ks:
                    if kind == "key" and key not in cache:
                        j, tail = key
                        for l in range(1, len(tail) + 1):  # ancestors too, cached or not: their K/V
                            need.add((j, tail[:l]))
        if need:
            t0 = time.perf_counter()
            cache.update(tg.keys(kv, sorted(need)))
            stats["fwd_s"] += time.perf_counter() - t0
            stats["rows"] += len(need)
            stats["fwds"] += 1
        for a in live:
            st = state[a.name]
            m = st["m"]
            nodes, ks = trees[a.name]
            acc = accept(nodes, x, m)
            e = min(acc + 1, n - 1 - m)
            st["rounds"].append(e - 1)
            st["m"] = m + e
            if ks:
                a.update(nodes, [tf[key] if kind == "tf" else cache[key] for kind, key in ks])
            stats["nodes"] += len(nodes) - 1
    del kv
    return {a.name: state[a.name]["rounds"] for a in arms}, tf


def gate(tg, model, mx, x, p, tf, kv_tf):
    """Tree forward against teacher forcing on text rows, and against a fresh forward on off-text keys."""
    rng = np.random.default_rng(0)
    n = len(x)
    js = sorted(rng.choice(np.arange(p, n - 6), size=6, replace=False).tolist())
    keys = []
    for j in js:
        for l in range(1, 5):
            keys.append((j, tuple(x[j:j + l])))
    got = tg.keys(kv_tf, keys)
    top1 = sum(int(got[(j, t)][0] == tf[j + len(t) - 1][0]) for j, t in keys)
    ov = np.mean([len(set(got[(j, t)].tolist()) & set(tf[j + len(t) - 1].tolist())) / KMAX for j, t in keys])
    # off-text keys: second-ranked token at j, then a free continuation of 2
    offs = []
    for j in js[:3]:
        alt = int(tf[j - 1][1])
        t2 = int(tf[j - 1][2])
        offs.append((j, (alt,)))
        offs.append((j, (alt, t2)))
        offs.append((j, (alt, t2, int(tf[j][0]))))
    got2 = tg.keys(kv_tf, offs)
    agree1, ov2 = 0, []
    for j, tail in offs:
        seq = list(x[:j]) + list(tail)
        _, top = tg.capture(seq)
        ref = top[-1]
        agree1 += int(ref[0] == got2[(j, tail)][0])
        ov2.append(len(set(ref.tolist()) & set(got2[(j, tail)].tolist())) / KMAX)
    return {"text_rows": len(keys), "text_top1_agree": top1, "text_top8_overlap": round(float(ov), 4),
            "off_rows": len(offs), "off_top1_agree": agree1, "off_top8_overlap": round(float(np.mean(ov2)), 4)}


def make_arms(mode: str, rec: str, comb: str) -> list[Arm]:
    if mode == "search":
        arms = [Arm(f"rec8w-{s}", "rec", 8, True, s) for s in SHAPES]
        for s in ("mid", "wide"):
            for q in (0.6, 0.9):
                arms.append(Arm(f"comb8w-{s}-q{q}", "comb", 8, True, s, q))
        return arms
    cs, cq = comb.split(":")
    return [Arm("rec4c", "rec", 4, False, rec), Arm("rec4w", "rec", 4, True, rec),
            Arm("rec8c", "rec", 8, False, rec), Arm("rec8w", "rec", 8, True, rec),
            Arm("look", "look"),
            Arm("comb8c", "comb", 8, False, cs, float(cq)), Arm("comb8w", "comb", 8, True, cs, float(cq))]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    for a in ("packed", "house", "gsm", "sp8", "a32", "out"):
        ap.add_argument(a)
    ap.add_argument("--mode", required=True, choices=("search", "frozen"))
    ap.add_argument("--gsm-range", default="0:40")
    ap.add_argument("--with-house", dest="with_house", action="store_true")
    ap.add_argument("--rec", default="mid")
    ap.add_argument("--comb", default="mid:0.9")
    args = ap.parse_args(argv[1:])

    import mlx.core as mx
    from tokenizers import Tokenizer

    import llvq_mlx as L

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    model, _, _ = L.build(args.packed)
    patch_dense(L, mx)
    tg = Target(model, mx)
    tok = Tokenizer.from_file(str(Path(args.packed) / "tokenizer.json"))
    arms = make_arms(args.mode, args.rec, args.comb)
    print(f"target tetra ({args.packed}); mode {args.mode}; arms "
          f"{[(a.name, a.kind, a.k, a.warm, a.shape, a.q) for a in arms]}; loaded in {time.perf_counter() - t0:.1f} s",
          flush=True)

    sp8, a32 = Path(args.sp8), Path(args.a32)
    dump = {}
    for line in (sp8 / "gsm200.jsonl").read_text().splitlines():
        o = json.loads(line)
        if o["decoder"] == "pass":
            dump[o["id"]] = o["tokens"]
    first = {(r["set"], r["id"]): r["first"] for r in json.loads((a32 / "rows.json").read_text())}
    mac_gen = json.loads((a32 / "mac-greedy.json").read_text())
    card_house, card_gsm = A.card_log(sp8 / "spec-house.txt"), A.card_log(sp8 / "spec-gsm200.txt")

    lo, hi = (int(v) for v in args.gsm_range.split(":"))
    sets = []
    if args.with_house:
        sets.append(("house", [(r, A.HOUSE_BUDGET, card_house.get(r["id"], {}))
                               for r in A.load_prompts(Path(args.house))]))
    sets.append(("gsm", [(r, A.GSM_BUDGET, card_gsm.get(r["id"], {})) for r in A.load_prompts(Path(args.gsm))[lo:hi]]))

    rows, raw, gated = [], {}, None
    stats = {"fwd_s": 0.0, "rows": 0, "fwds": 0, "nodes": 0}
    for set_name, recs in sets:
        for a in arms:  # warm tables are per set
            if a.table is not None:
                a.table.fill(-1)
        for rec, budget, card in recs:
            pid = rec["id"]
            ids = A.prompt_ids(tok, rec["messages"])
            if card.get("prompt") is not None and card["prompt"] != len(ids):
                raise SystemExit(f"{pid}: prompt {len(ids)} tokens here, {card['prompt']} on the card")
            if pid in dump:
                gen, source = [first[(set_name, pid)]] + dump[pid][:budget], "card"
            else:
                gen, source = mac_gen[pid], "mac"
            x, p = ids + gen, len(ids)
            tp = time.perf_counter()
            if gated is None:
                kv, tf = tg.capture(x)
                gated = gate(tg, model, mx, x, p, tf, kv)
                del kv
                print("gate " + json.dumps(gated), flush=True)
            r0 = dict(stats)
            res, _ = run_prompt(tg, arms, x, p, stats)
            line = {"set": set_name, "id": pid, "kind": rec["kind"], "text": source, "prompt": p,
                    "tokens": len(gen) - 1, "rows": stats["rows"] - r0["rows"], "fwds": stats["fwds"] - r0["fwds"],
                    "s": round(time.perf_counter() - tp, 1)}
            raw[f"{set_name}/{pid}"] = res
            for name, rr in res.items():
                line[name] = {"passes": len(rr), "accepted": sum(rr), "hist": np.bincount(rr, minlength=16).tolist()}
            if "dflash:tree" in card:
                line["card_tree"] = {"tokens": card["dflash:tree"][0], "acc": card["dflash:tree"][1],
                                     "passes": card["dflash:tree"][2]}
            rows.append(line)
            short = {k: (v if not isinstance(v, dict) or "hist" not in v else round(v["accepted"] / v["passes"], 3))
                     for k, v in line.items() if k != "card_tree"}
            print(json.dumps(short), flush=True)
            (out / "rows.json").write_text(json.dumps(rows, indent=1))
    meta = {"mode": args.mode, "gsm_range": args.gsm_range, "house": args.with_house, "gate": gated,
            "arms": [{"name": a.name, "kind": a.kind, "k": a.k, "warm": a.warm, "shape": a.shape,
                      "p": list(a.p) if a.kind != "look" else None, "q": a.q} for a in arms],
            "stats": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in stats.items()},
            "table_bytes": {a.name: (a.table.nbytes if a.table is not None else 0) for a in arms}}
    (out / "rows.json").write_text(json.dumps(rows, indent=1))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    (out / "accept-rounds.json.gz").write_bytes(gzip.compress(json.dumps(raw).encode(), mtime=0))
    print(f"stats {meta['stats']}")
    print(f"done in {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
