# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx-lm==0.28.3", "mlx==0.32.4", "numpy==2.5.4", "safetensors==0.8.0", "tokenizers==0.23.3",
#                 "pyarrow==26.0.1", "transformers==5.22.1", "huggingface_hub", "llvq-tetra"]
# [tool.uv.sources]
# llvq-tetra = { path = "../llvq-tetra" }
# ///
"""Chat acceptance of the speculative loop on the sealed Tetra 4B, on the Mac, at $0 (2026-11-11).

    uv run ops/spec_chat.py dflash <packed> <mtbench.jsonl> <house.jsonl> <arbre32_brut> <out_dir>
    uv run ops/spec_chat.py tables <packed> <mtbench.jsonl> <house.jsonl> <arbre32_brut> <out_dir>

Prompts: the 80 MT-Bench first turns (ops/data/spec-chat-mtbench-v1.jsonl) and the house set's 10 chat prompts.
Text: the MLX door's own greedy decode, 256 tokens after `first`, stopped at <|im_end|> included (dflash_arbre32's
convention; there is no card text for these prompts). The house chat text is arbre32's mac-greedy.json, unchanged.

dflash: per pending position m, the drafter's top 4 at depths 1..15 (dflash_arbre32's forward), for two drafters:
"ref" (the checkpoint widened to f32) and "int4" (int4 g128 on the 35 linear weights and fc, dflash_brouillon_int4's
fake quantizer). Written per position: the accepted depth of the tree of N nodes for N in 1, 3, 7, 15
(draft_tree_scored's construction), of the depth-15 chain (a chain of depth c accepts min(that, c)), and the drafter's
own confidence known before the verify pass: depth-1 top-1 and top-2 log probabilities, the expected accepted count
of each tree (sum of the drafter's path probabilities over its nodes) and of the chain. Also the target's top-1
probability at row m-1 (the row that produced the root, known from the previous pass at no cost).

tables: the weightless drafters of spec_sans_poids.py (exact verify rows) on the same text, warm tables in file order
(MT-Bench and house separate), with a gate on the lookup match length n known before the pass: below t no pass is
run and the served 4E emits one token (the table's root row is then that 4E row's top k). Arms record each pass's
rows (0 = a 4E token), accepted count and n.
"""
from __future__ import annotations

import gzip
import heapq
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "mlx_tetra"))
import dflash_arbre32 as A  # noqa: E402

TREES = (1, 3, 7, 15)


def jobs(mtb: Path, house: Path):
    out = []
    for line in mtb.read_text().splitlines():
        o = json.loads(line)
        if "id" in o:
            out.append(("mtbench", o["id"], o["category"], [("user", o["prompt"])]))
    for r in A.load_prompts(house):
        if r["kind"] == "chat":
            out.append(("house", r["id"], "house-chat", r["messages"]))
    return out


def texts(model, tok, js, a32: Path, out: Path):
    """Greedy text per prompt: arbre32's for the house chat, the MLX door's own here for MT-Bench (cached)."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    old = json.loads((a32 / "mac-greedy.json").read_text())
    cache_p = out / "mac-greedy.json"
    gen = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    for set_name, pid, _, msgs in js:
        if pid in gen:
            continue
        if set_name == "house":
            gen[pid] = old[pid]
        else:
            ids = A.prompt_ids(tok, msgs)
            g = []
            for (t, _), _ in zip(generate_step(mx.array(ids), model, max_tokens=A.HOUSE_BUDGET + 1),
                                 range(A.HOUSE_BUDGET + 1)):
                g.append(int(t))
                if int(t) == A.IM_END and len(g) > 1:
                    break
            gen[pid] = g
            print(f"text {pid}: {len(g)} tokens", flush=True)
        cache_p.write_text(json.dumps(gen))
    return gen


def capture_p1(model, x):
    """dflash_arbre32.capture plus the target's top-1 probability a row."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner = model.model
    h = inner.embed_tokens(mx.array(x)[None])
    mask = create_attention_mask(h, None)
    feats = []
    for idx, layer in enumerate(inner.layers):
        if idx in A.CTX_LAYERS:
            feats.append(h[0].astype(mx.float32))
        h = layer(h, mask, None)
    z = inner.embed_tokens.as_linear(inner.norm(h))[0].astype(mx.float32)
    am = mx.argmax(z, axis=-1)
    p1 = mx.exp(mx.max(z, axis=-1) - mx.logsumexp(z, axis=-1))
    f = mx.concatenate(feats, axis=-1)
    mx.eval(f, am, p1)
    return f, am.tolist(), np.array(p1)


def tree_stats(x, p, ti, tl):
    """Per position: accepted depth for each N in TREES, expected count under the drafter for each N."""
    n = len(x)
    big = max(TREES)
    acc = {nb: [] for nb in TREES}
    exp_n = {nb: [] for nb in TREES}
    for r, m in enumerate(range(p, n - 1)):
        cand, lp = ti[r], tl[r]
        heap = [(-float(lp[0, k]), (k,)) for k in range(A.TOPK)]
        heapq.heapify(heap)
        rank_of, scores = {}, []
        while heap and len(rank_of) < big:
            s, path = heapq.heappop(heap)
            rank_of[path] = len(rank_of)
            scores.append(-s)
            if len(path) < A.BLOCK - 1:
                d = len(path)
                for k in range(A.TOPK):
                    heapq.heappush(heap, (s - float(lp[d, k]), path + (k,)))
        idx, path, a = [], (), 0
        while a < A.BLOCK - 1 and m + 1 + a < n:
            hit = np.nonzero(cand[a] == x[m + 1 + a])[0]
            if not len(hit) or path + (int(hit[0]),) not in rank_of:
                break
            path = path + (int(hit[0]),)
            idx.append(rank_of[path])
            a += 1
        cum = np.cumsum(np.exp(np.asarray(scores)))
        for nb in TREES:
            depth = 0
            while depth < len(idx) and idx[depth] < nb:
                depth += 1
            acc[nb].append(depth)
            exp_n[nb].append(round(float(cum[nb - 1]), 4))
    return acc, exp_n


def run_dflash(argv):
    import mlx.core as mx
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    import dflash_brouillon_int4 as Q
    import llvq_mlx as L

    packed, mtb, house, a32, out = argv[0], Path(argv[1]), Path(argv[2]), Path(argv[3]), Path(argv[4])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    model, _, _ = L.build(packed)
    qe = model.model.embed_tokens
    table = mx.dequantize(qe.weight, qe.scales, qe.biases, group_size=qe.group_size, bits=qe.bits).astype(mx.float32)
    mx.eval(table)
    tok = Tokenizer.from_file(str(Path(packed) / "tokenizer.json"))
    js = jobs(mtb, house)
    gen = texts(model, tok, js, a32, out)
    print(f"texts ready at {time.perf_counter() - t0:.0f} s", flush=True)
    dpath = hf_hub_download(A.DRAFT_REPO, "model.safetensors", revision=A.DRAFT_REV)
    arms, dtypes, err = Q.build_arms(dpath, ["ref", "int4g128+fc"])
    arms = {"ref": arms["ref"], "int4": arms["int4g128+fc"]}
    print(f"drafter {A.DRAFT_REPO}@{A.DRAFT_REV[:8]} {dtypes}; weight error int4 "
          f"{json.dumps(err['int4g128+fc'])}; loaded at {time.perf_counter() - t0:.0f} s", flush=True)
    rows, raw = [], {}
    for set_name, pid, cat, msgs in js:
        ids = A.prompt_ids(tok, msgs)
        g = gen[pid]
        x, p = ids + g, len(ids)
        feats, am, p1 = capture_p1(model, x)
        agree = sum(1 for i in range(p - 1, len(x) - 1) if am[i] == x[i + 1]) / (len(x) - p)
        ms = list(range(p, len(x) - 1))
        rec = {"tgt_p1": [round(float(p1[m - 1]), 4) for m in ms]}
        line = {"set": set_name, "id": pid, "category": cat, "prompt": p, "tokens": len(g) - 1,
                "tf_agree": round(agree, 4)}
        for name, d in arms.items():
            ks, vs = d.context(feats)
            parts = [d.drafts(table, table, x, ms[i:i + A.BATCH], ks, vs) for i in range(0, len(ms), A.BATCH)]
            ti = np.concatenate([q[0] for q in parts])
            tl = np.concatenate([q[1] for q in parts])
            order = np.argsort(-tl, axis=-1, kind="stable")
            ti = np.take_along_axis(ti, order, axis=-1)
            tl = np.take_along_axis(tl, order, axis=-1)
            acc, exp_n = tree_stats(x, p, ti, tl)
            chain = A.chain_depths(x, p, ti)
            ec = np.cumsum(np.cumprod(np.exp(tl[:, :, 0]), axis=1), axis=1)  # expected chain count by depth
            rec[name] = {"chain15": chain, **{f"tree{nb}": acc[nb] for nb in TREES},
                         **{f"exp_tree{nb}": exp_n[nb] for nb in TREES},
                         "exp_chain15": [round(float(v), 4) for v in ec[:, -1]],
                         "lp1": [round(float(v), 4) for v in tl[:, 0, 0]],
                         "lp2": [round(float(v), 4) for v in tl[:, 0, 1]]}
            rr = A.replay(x, p, acc[15])
            line[name + "_tree15"] = round(sum(rr) / len(rr), 3)
            del ks, vs
        raw[f"{set_name}/{pid}"] = rec
        rows.append(line)
        print(json.dumps(line), flush=True)
        del feats
    (out / "rows.json").write_text(json.dumps(rows, indent=1))
    (out / "positions.json.gz").write_bytes(gzip.compress(json.dumps(raw).encode(), mtime=0))
    print(f"done in {time.perf_counter() - t0:.0f} s")
    return 0


# ---------------------------------------------------------------- tables


def run_tables(argv):
    import mlx.core as mx
    from tokenizers import Tokenizer

    import llvq_mlx as L
    import spec_sans_poids as S

    packed, mtb, house, a32, out = argv[0], Path(argv[1]), Path(argv[2]), Path(argv[3]), Path(argv[4])
    out.mkdir(parents=True, exist_ok=True)
    gen = json.loads((out.parent / "dflash" / "mac-greedy.json").read_text())
    t0 = time.perf_counter()
    model, _, _ = L.build(packed)
    S.patch_dense(L, mx)
    tg = S.Target(model, mx)
    tok = Tokenizer.from_file(str(Path(packed) / "tokenizer.json"))

    # (name, kind, gate t): kind comb = lookup spine + warm recycling table k 4 (`mid`, q 0.9); look = lookup chain only
    specs = [("comb4w", "comb", 0)] + [(f"comb4w-g{t}", "comb", t) for t in (1, 2, 3, 4, 6)] + \
            [("look", "look", 0)] + [(f"look-g{t}", "look", t) for t in (1, 2, 3, 4, 6)]
    arms = []
    for name, kind, t in specs:
        a = S.Arm(name, kind, 4, True, "mid", 0.9 if kind == "comb" else 0.0)
        a.gate = t
        arms.append(a)
    print(f"tables: arms {specs}; loaded in {time.perf_counter() - t0:.1f} s", flush=True)

    js = jobs(mtb, house)
    rows, raw, gated = [], {}, None
    stats = {"fwd_s": 0.0, "rows": 0, "fwds": 0}
    cur_set = None
    for set_name, pid, cat, msgs in js:
        if set_name != cur_set:
            for a in arms:
                if a.table is not None:
                    a.table.fill(-1)
            cur_set = set_name
        ids = A.prompt_ids(tok, msgs)
        g = gen[pid]
        x, p = ids + g, len(ids)
        n = len(x)
        arr = np.asarray(x, dtype=np.int64)
        tp = time.perf_counter()
        kv, tf = tg.capture(x)
        if gated is None:
            gated = S.gate(tg, model, mx, x, p, tf, kv)
            print("gate " + json.dumps(gated), flush=True)
        cache: dict = {}
        state = {a.name: {"m": p, "passes": []} for a in arms}
        r0 = stats["rows"]
        while True:
            live = [a for a in arms if state[a.name]["m"] < n - 1]
            if not live:
                break
            plan, need = {}, set()
            for a in live:
                m = state[a.name]["m"]
                _, ln = S.lookup(arr, m)
                if ln < a.gate:
                    plan[a.name] = (None, None, ln)
                    continue
                nodes = a.tree(arr, m)
                ks = S.node_keys(nodes, x, m) if a.kind != "look" else None
                plan[a.name] = (nodes, ks, ln)
                for kind, key in (ks or []):
                    if kind == "key" and key not in cache:
                        j, tail = key
                        for l in range(1, len(tail) + 1):
                            need.add((j, tail[:l]))
            if need:
                ta = time.perf_counter()
                cache.update(tg.keys(kv, sorted(need)))
                stats["fwd_s"] += time.perf_counter() - ta
                stats["rows"] += len(need)
                stats["fwds"] += 1
            for a in live:
                st = state[a.name]
                m = st["m"]
                nodes, ks, ln = plan[a.name]
                if nodes is None:  # gate says no: one 4E token; its row (the root's) feeds the table
                    st["passes"].append((0, 0, ln))
                    if a.table is not None:
                        a.table[int(arr[m])] = tf[m][:a.k]
                    st["m"] = m + 1
                    continue
                acc = S.accept(nodes, x, m)
                e = min(acc + 1, n - 1 - m)
                st["passes"].append((len(nodes), e - 1, ln))
                st["m"] = m + e
                if ks:
                    a.update(nodes, [tf[key] if kind == "tf" else cache[key] for kind, key in ks])
        del kv
        line = {"set": set_name, "id": pid, "category": cat, "prompt": p, "tokens": len(g) - 1,
                "rows": stats["rows"] - r0, "s": round(time.perf_counter() - tp, 1)}
        for a in arms:
            ps = state[a.name]["passes"]
            spec = [q for q in ps if q[0] > 0]
            line[a.name] = round(sum(q[1] for q in ps) / len(ps), 3)
            line[a.name + "_spec"] = len(spec)
        raw[f"{set_name}/{pid}"] = {a.name: state[a.name]["passes"] for a in arms}
        rows.append(line)
        print(json.dumps(line), flush=True)
        (out / "rows.json").write_text(json.dumps(rows, indent=1))
    meta = {"gate": gated, "arms": specs, "shape": S.SHAPES["mid"][:4], "q": 0.9,
            "stats": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in stats.items()}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    (out / "passes.json.gz").write_bytes(gzip.compress(json.dumps(raw).encode(), mtime=0))
    print(f"stats {meta['stats']}; done in {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    mode = sys.argv[1]
    sys.exit(run_dflash(sys.argv[2:]) if mode == "dflash" else run_tables(sys.argv[2:]))
