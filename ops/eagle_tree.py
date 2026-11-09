# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx-lm==0.28.3", "mlx==0.32.4", "numpy==2.5.4", "safetensors==0.8.0", "tokenizers==0.23.3",
#                 "pyarrow==26.0.1", "transformers==5.22.1", "huggingface_hub", "llvq-tetra"]
# [tool.uv.sources]
# llvq-tetra = { path = "../llvq-tetra" }
# ///
"""EAGLE-3 draft trees on the sealed Tetra 4B, on the Mac, at $0 (2026-11-09).

    uv run ops/eagle_tree.py <packed_dir> <out_dir> [n_gsm8k=50] [K=8] [D=10] [head=bf16|int4]

The sibling of `ops/eagle_alpha.py`, same target (the sealed 4B through the MLX door), same pinned head
(AngelSlim/Qwen3-4B_eagle3 at fd331e59), same five prompts and 50 GSM8K pilot problems. It measures the tokens a
verify pass emits when the head drafts a TREE of N nodes instead of a chain of k, for two tree policies:

- dynamic (EAGLE-2): from the root, every level expands the K candidates of highest cumulative draft log
  probability, each proposes its top-K children; after D levels the N candidates of highest cumulative score form
  the tree (a child never scores above its parent, so the set is closed under ancestors). D head steps a round.
- static: a fixed set of N rank paths (rank of the token among the head's candidates at each depth). For workload
  w the set is the N most frequent rank prefixes of the true tokens over the OTHER workloads (leave one workload
  out, each workload weighted equally), so no workload chooses its own tree. Head steps a round = the tree's depth.

Greedy decoding makes acceptance a function of the greedy text: a tree accepts depth d when the greedy tokens
x[m+1..m+d] form a path of the tree from its root. The head's cache over positions 0..m-1 is built from the target's
states (what EAGLE-3 holds after a verify), exactly as `eagle_alpha.py` replays a chain. Every position of every
completion gets its tree in one batched pass; the rounds are then replayed as the loop runs them, for every N.

`head=int4` rounds every matrix of the head (fc, q, k, v, o, gate, up, down and the 32k head) to int4 g64 by
round-to-nearest with an f16 scale and an f16 offset a group of 64 inputs, the form the card would hold (+0.24 b/param
at 4B, plan-speculation.md), then runs it in f32. The norms stay as published.

Gate: the chain k = 1, 2, 3 replayed from this script's teacher-forced ranks must equal `eagle_alpha.py`'s own replay
on the same completions (which reproduces eagle-alpha-2026-11-08.txt).

The completions are cached in <out_dir>/completions.json (written on the first run, read on the next); the
per-position ranks and dynamic accepts go to <out_dir>/per-position.json.gz.
What this does not measure: the card, any head or verify cost, a quantized head, a head retrained on Tetra features.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eagle_alpha as EA  # noqa: E402  (also puts ops/mlx_tetra on the path)

DMAX_FORCED = 10   # depth of the teacher-forced rank paths (static trees, the chain gate)
RANK_CAP = 32      # ranks at or past this count as "not drafted" for the static trees
N_LIST = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 16, 20, 24, 31, 32)
WORKLOADS = ("chat", "code", "summary", "edit", "rag", "gsm8k")


def int4_g64(w):
    """Round-to-nearest int4 with an f16 scale and offset a group of 64 inputs, dequantized to f32."""
    import mlx.core as mx

    rows, cols = w.shape
    g = w.reshape(rows, cols // 64, 64)
    lo, hi = g.min(axis=-1, keepdims=True), g.max(axis=-1, keepdims=True)
    s = ((hi - lo) / 15.0).astype(mx.float16).astype(mx.float32)
    b = lo.astype(mx.float16).astype(mx.float32)
    q = mx.clip(mx.round((g - b) / mx.where(s > 0, s, 1.0)), 0, 15)
    return (s * q + b).reshape(rows, cols)


def quantize_head(head):
    import mlx.core as mx

    for name in ("fc", "lm", "q", "k", "v", "o", "gate", "up", "down"):
        w = int4_g64(getattr(head, name))
        mx.eval(w)  # materialized once: a lazy graph would be rebuilt and held at every use
        setattr(head, name, w)


def log_softmax(z):
    import mlx.core as mx

    return z - mx.logsumexp(z, axis=-1, keepdims=True)


def head_logits(head, o):
    import mlx.core as mx

    return mx.fast.rms_norm(o, head.norm, head.eps) @ head.lm.T


def ranks_of(z, dy):
    """Rank of draft id dy[r] in row r of z under argmax's tie rule (first index wins); RANK_CAP when dy < 0."""
    import mlx.core as mx

    ok = dy >= 0
    d = mx.where(ok, dy, 0)
    L = mx.take_along_axis(z, d[:, None], axis=1)
    idx = mx.arange(z.shape[1])[None, :]
    r = (z > L).sum(axis=1) + ((z == L) & (idx < d[:, None])).sum(axis=1)
    return mx.where(ok, r, RANK_CAP)


def forced_ranks(head, model, x, aux, t2d, depth=DMAX_FORCED):
    """ranks[j][i]: rank of the true token x[i+1+j] (j = 1..depth) in the head's draft at base row i, every earlier
    level fed the TRUE tokens: the chain of eagle_alpha.drafts with d_j replaced by x[i+1+j]. Also returns o1, k1, v1
    (the base pass), which the dynamic expansion reuses."""
    import mlx.core as mx

    n = len(x)
    t = n - 1
    emb = model.model.embed_tokens(mx.array(x[1:])).astype(mx.float32)
    fused = mx.concatenate([a[:t] for a in aux], axis=-1) @ head.fc.T
    r1, q1, k1, v1 = head.qkv(emb, fused, 0)
    o1 = head.finish(r1, head.attend(q1, k1, v1, [], []))
    xs = np.asarray(x, dtype=np.int64)
    out = []
    o, ek, ev = o1, [], []
    for j in range(1, depth + 1):
        pos = np.arange(t) + 1 + j
        y = np.where(pos < n, xs[np.minimum(pos, n - 1)], 0)
        dy = np.where(pos < n, t2d[y], -1)
        z = head_logits(head, o)
        rk = ranks_of(z, mx.array(dy.astype(np.int32)))
        mx.eval(rk)
        out.append(np.asarray(rk))
        if j == depth:
            break
        e = model.model.embed_tokens(mx.array(y.astype(np.int32))).astype(mx.float32)
        r, q, k, v = head.qkv(e, o, j)
        ek.append(k)
        ev.append(v)
        o = head.finish(r, head.attend(q, k1, v1, ek, ev))
        mx.eval(o)
    return np.stack(out, axis=0), (o1, k1, v1)


def qkv_rows(head, emb, hid, offset: int):
    """Head.qkv for K slots of P rows, emb and hid [K, P, H]; row r of every slot sits at position offset + r."""
    import mlx.core as mx

    rms = lambda x, g: mx.fast.rms_norm(x, g, head.eps)
    x = mx.concatenate([rms(emb, head.inl), rms(hid, head.hn)], axis=-1)
    K, P = x.shape[0], x.shape[1]
    q = (x @ head.q.T).reshape(K, P, head.hq, head.dh).transpose(0, 2, 1, 3)
    k = (x @ head.k.T).reshape(K, P, head.hk, head.dh).transpose(0, 2, 1, 3)
    v = (x @ head.v.T).reshape(K, P, head.hk, head.dh).transpose(0, 2, 1, 3)
    return hid, head.rope(q, offset), head.rope(k, offset), v


def attend_slot(head, q, kb, vb, i0, extra_k, extra_v):
    """q [Hq, P, D] for rows at positions i0..i0+P-1: row r attends base keys 0..i0+r and its extra keys."""
    import mlx.core as mx

    rep = head.hq // head.hk
    kb, vb = mx.repeat(kb, rep, axis=0), mx.repeat(vb, rep, axis=0)
    P, t = q.shape[1], kb.shape[1]
    scale = head.dh ** -0.5
    s = (q @ kb.transpose(0, 2, 1)) * scale
    allowed = mx.arange(t)[None, :] <= (i0 + mx.arange(P))[:, None]
    cols = [mx.where(allowed, s, -mx.inf)]
    ek = [mx.repeat(k, rep, axis=0) for k in extra_k]
    ev = [mx.repeat(v, rep, axis=0) for v in extra_v]
    cols += [(q * k).sum(-1, keepdims=True) * scale for k in ek]
    p = mx.softmax(mx.concatenate(cols, axis=-1), axis=-1, precise=True)
    out = p[..., :t] @ vb
    for j, v in enumerate(ev):
        out = out + p[..., t + j : t + j + 1] * v
    return out


def expand(head, model, x, p, base, K, D):
    """EAGLE-2's candidate tree at every round position. Returns per level l = 1..D: tok [P, M_l] (target ids),
    sc [P, M_l] (cumulative log prob), par [P, M_l] (index into level l-1, -1 at level 1), for base rows
    i = p-1 .. n-3 (the rounds m = i + 1 = p .. n-2)."""
    import mlx.core as mx

    o1, k1, v1 = base
    n = len(x)
    i0, i1 = p - 1, n - 2
    P = i1 - i0
    if P <= 0:
        return None
    d2t = head.d2t
    z1 = log_softmax(head_logits(head, o1[i0:i1]))
    ids = mx.argpartition(-z1, kth=K - 1, axis=-1)[:, :K]
    s1 = mx.take_along_axis(z1, ids, axis=1)
    mx.eval(ids, s1)
    levels = [dict(tok=np.asarray(ids + d2t[ids]), sc=np.asarray(s1), par=np.full((P, K), -1, dtype=np.int64))]
    # The expanded set E_l: K slots a position.
    e_tok = (ids + d2t[ids]).T                                  # [K, P]
    e_sc = s1.T                                                 # [K, P]
    e_h = mx.broadcast_to(o1[i0:i1][None], (K, P, o1.shape[1]))  # parent output, [K, P, H]
    e_cand = np.tile(np.arange(K)[:, None], (1, P))            # candidate index at level l
    anc_k: list = []                                            # [K, Hk, P, Dh] per ancestor level
    anc_v: list = []
    H = o1.shape[1]
    for lvl in range(1, D):
        # The K slots of a level in one call, so the head's weights are read once a level.
        emb = model.model.embed_tokens(e_tok.reshape(-1)).astype(mx.float32).reshape(K, P, H)
        hid, q, kk, vv = qkv_rows(head, emb, e_h, i0 + lvl)   # q [K, Hq, P, D], kk vv [K, Hk, P, D]
        ctx = mx.concatenate([attend_slot(head, q[k], k1, v1, i0, [a[k] for a in anc_k] + [kk[k]],
                                          [a[k] for a in anc_v] + [vv[k]]) for k in range(K)], axis=1)
        o = head.finish(hid.reshape(K * P, H), ctx)             # [K*P, H]
        z = log_softmax(head_logits(head, o))
        cid = mx.argpartition(-z, kth=K - 1, axis=-1)[:, :K]
        cz = mx.take_along_axis(z, cid, axis=1)
        mx.eval(o, cid, cz, kk, vv)
        o_all = o.reshape(K, P, H)
        cid = cid.reshape(K, P, K)
        cz = cz.reshape(K, P, K)
        cs = e_sc[:, :, None] + cz                              # [K, P, K]
        tok = np.asarray(cid + d2t[cid]).transpose(1, 0, 2).reshape(P, K * K)
        sc = np.asarray(cs).transpose(1, 0, 2).reshape(P, K * K)
        par = np.repeat(e_cand.T, K, axis=1)                    # [P, K*K]: slot s's candidate, K times
        levels.append(dict(tok=tok, sc=sc, par=par))
        if lvl + 1 == D:
            break
        sel = np.argsort(-sc, axis=1, kind="stable")[:, :K]     # [P, K] into K*K
        pslot = (sel // K).T                                    # [K, P]
        e_cand = sel.T
        e_tok = mx.array(np.take_along_axis(tok, sel, axis=1).T.astype(np.int32))
        e_sc = mx.array(np.take_along_axis(sc, sel, axis=1).T)
        ps = mx.array(pslot.astype(np.int32))
        e_h = mx.take_along_axis(o_all, ps[:, :, None], axis=0)
        ak = anc_k + [kk]
        av = anc_v + [vv]
        anc_k = [mx.take_along_axis(a, ps[:, None, :, None], axis=0) for a in ak]
        anc_v = [mx.take_along_axis(a, ps[:, None, :, None], axis=0) for a in av]
        mx.eval(e_h, *anc_k, *anc_v)
    return levels


def dynamic_accept(levels, x, p, d_list, n_list):
    """acc[d][N] [P]: greedy tokens a tree of the N best candidates of levels <= d accepts, at every round position."""
    n = len(x)
    i0 = p - 1
    P = levels[0]["tok"].shape[0]
    D = len(levels)
    acc = {d: {N: np.zeros(P, dtype=np.int64) for N in n_list} for d in d_list}
    for r in range(P):
        i = i0 + r
        # the true path through the candidates
        path = []  # (level index, candidate index)
        prev = -1
        for lv in range(D):
            j = i + 2 + lv
            if j > n - 1:
                break
            tk, pr = levels[lv]["tok"][r], levels[lv]["par"][r]
            hit = np.nonzero((tk == x[j]) & (pr == prev))[0]
            if hit.size == 0:
                break
            prev = int(hit[0])
            path.append((lv, prev))
        if not path:
            continue
        for d in d_list:
            pool = np.concatenate([levels[lv]["sc"][r] for lv in range(min(d, D))])
            lvl_of = np.concatenate([np.full(levels[lv]["sc"].shape[1], lv) for lv in range(min(d, D))])
            off = np.cumsum([0] + [levels[lv]["sc"].shape[1] for lv in range(min(d, D))])
            ranks = []
            for lv, c in path:
                if lv >= d:
                    break
                s = levels[lv]["sc"][r][c]
                g = off[lv] + c
                ranks.append(int((pool > s).sum() + ((pool == s) & ((lvl_of < lv) | ((lvl_of == lv) & (np.arange(pool.size) < g)))).sum()))
            for N in n_list:
                a = 0
                while a < len(ranks) and ranks[a] < N:
                    a += 1
                acc[d][N][r] = a
    return acc


def rounds_of(accept, x, p):
    """Replay a round loop over x[p:], `accept[m - p]` the drafts kept at round m: (rounds, tokens)."""
    n = len(x)
    m, rounds = p, 0
    while m < n - 1:
        a = int(accept[m - p])
        m += min(a + 1, n - 1 - m)
        rounds += 1
    return rounds, n - 1 - p


def static_tree(freqs: list[dict], N: int) -> set:
    """The N rank paths of highest mean frequency over the given workloads (ancestor-closed: a prefix is never
    rarer than its extension; ties go to the shorter path, then lexicographic)."""
    keys = set().union(*[f.keys() for f in freqs])
    mean = {k: sum(f.get(k, 0.0) for f in freqs) / len(freqs) for k in keys}
    order = sorted(mean, key=lambda k: (-mean[k], len(k), k))
    return set(order[:N])


def prefix_freqs(rank_rows: list[np.ndarray]) -> dict:
    """Frequency of every rank prefix (ranks < RANK_CAP) over the rows of a workload, each row one round position."""
    cnt: dict = {}
    tot = 0
    for rr in rank_rows:
        for col in range(rr.shape[1]):
            tot += 1
            pre = ()
            for j in range(rr.shape[0]):
                v = int(rr[j, col])
                if v >= RANK_CAP:
                    break
                pre = pre + (v,)
                cnt[pre] = cnt.get(pre, 0) + 1
    return {k: v / tot for k, v in cnt.items()}


def static_accept(rr: np.ndarray, tree: set) -> np.ndarray:
    """Drafts kept at every round position of one completion, rr [DMAX, P] the true ranks."""
    P = rr.shape[1]
    a = np.zeros(P, dtype=np.int64)
    for col in range(P):
        pre = ()
        for j in range(rr.shape[0]):
            pre = pre + (int(rr[j, col]),)
            if pre not in tree:
                break
            a[col] += 1
    return a


def main(argv: list[str]) -> int:
    import mlx.core as mx
    from huggingface_hub import hf_hub_download
    from mlx_lm.generate import generate_step

    packed, out_dir = argv[1], Path(argv[2])
    n_gsm = int(argv[3]) if len(argv) > 3 else 50
    K = int(argv[4]) if len(argv) > 4 else 8
    D = int(argv[5]) if len(argv) > 5 else 10
    head_form = argv[6] if len(argv) > 6 else "bf16"
    if head_form not in ("bf16", "int4"):
        raise SystemExit(f"head form {head_form!r}: bf16 or int4")
    out_dir.mkdir(parents=True, exist_ok=True)
    import llvq_mlx as L

    t0 = time.perf_counter()
    model, _, _ = L.build(packed)
    fmt = L.ChatFormat(packed)
    hp = hf_hub_download(EA.HEAD_REPO, "model.safetensors", revision=EA.HEAD_REV)
    head = EA.Head(hp)
    if head_form == "int4":
        quantize_head(head)
    d2t = np.asarray(head.d2t).astype(np.int64)
    vocab = int(model.model.embed_tokens.weight.shape[0])
    t2d = np.full(vocab, -1, dtype=np.int64)
    t2d[np.arange(d2t.size) + d2t] = np.arange(d2t.size)
    print(f"target: the sealed 4B through the MLX door ({packed}); head {EA.HEAD_REPO}@{EA.HEAD_REV[:8]}; "
          f"aux layers {EA.AUX}; head {head_form}; draft vocabulary {d2t.size} of {vocab}; K = {K}, D = {D}; "
          f"loaded in {time.perf_counter() - t0:.1f} s", flush=True)

    work: list[tuple[str, str, int]] = [(n, m, 256) for n, m in EA.specrun_prompts().items()]
    if n_gsm > 0:
        import pyarrow.parquet as pq

        gp = hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset",
                             revision=EA.GSM8K_REV)
        qs = pq.read_table(gp).column("question").to_pylist()
        for j in EA.splitmix_select(len(qs), n_gsm):
            work.append((f"gsm8k/{j}", qs[j].strip() + "\n" + EA.INSTRUCTION, 1024))

    cache = out_dir / "completions.json"
    comp = json.loads(cache.read_text()) if cache.exists() else {}
    stops = (fmt.im_end, fmt.eot)
    seqs = []
    for name, msg, max_new in work:
        ids = fmt.turn(msg)
        if name in comp and comp[name]["prompt"] == ids:
            gen = comp[name]["gen"]
        else:
            gen = []
            for (tid, _), _ in zip(generate_step(mx.array(ids), model, max_tokens=max_new), range(max_new)):
                tid = int(tid)
                if tid in stops:
                    break
                gen.append(tid)
            comp[name] = {"prompt": ids, "gen": gen}
            cache.write_text(json.dumps(comp))
        seqs.append((name, ids, gen))
    print(f"completions ready ({len(seqs)}), {time.perf_counter() - t0:.0f} s", flush=True)

    rank_rows: dict[str, list] = {w: [] for w in WORKLOADS}
    dyn: dict[str, list] = {w: [] for w in WORKLOADS}
    chain_gate = {w: {k: [0, 0, 0, 0] for k in (1, 2, 3)} for w in WORKLOADS}
    d_list = tuple(d for d in (2, 3, 4, 5, 6, 8, 10) if d <= D)
    n_tok = 0
    for si, (name, ids, gen) in enumerate(seqs):
        w = "gsm8k" if name.startswith("gsm8k") else name
        x = ids + gen
        p = len(ids)
        if len(x) - 1 - p < 1:
            continue
        aux, am = EA.target_states(model, x)
        tf = am.tolist()
        agree = sum(1 for i in range(p - 1, len(x) - 1) if tf[i] == x[i + 1]) / max(1, len(x) - p)
        ranks, base = forced_ranks(head, model, x, aux, t2d)
        rr = ranks[:, p - 1 : len(x) - 2]                     # round positions m = p .. n-2
        rank_rows[w].append(rr)
        # chain gate: eagle_alpha's own drafts and replay against this script's ranks
        d = EA.drafts(head, model, x, aux)
        for k in (1, 2, 3):
            r_ea, e_ea = EA.replay(x, p, d, k)
            lead = np.zeros(rr.shape[1], dtype=np.int64)
            for col in range(rr.shape[1]):
                a = 0
                while a < k and rr[a, col] == 0:
                    a += 1
                lead[col] = a
            r_me, e_me = rounds_of(lead, x, p)
            g = chain_gate[w][k]
            g[0] += r_ea
            g[1] += e_ea
            g[2] += r_me
            g[3] += e_me
        levels = expand(head, model, x, p, base, K, D)
        dyn[w].append((x, p, dynamic_accept(levels, x, p, d_list, N_LIST)))
        n_tok += len(gen)
        line = {"name": name, "prompt": p, "tokens": len(gen), "teacher_forced_agree": round(agree, 4),
                "elapsed_s": round(time.perf_counter() - t0)}
        print(json.dumps(line), flush=True)

    # Raw per-position data: ranks and the dynamic accepts, for re-analysis without the Mac's GPU.
    raw = {}
    for w in WORKLOADS:
        raw[w] = {"ranks": [r.tolist() for r in rank_rows[w]],
                  "dynamic": [{"p": p, "n": len(x), "x_tail": x[p - 1:],
                               "acc": {str(dd): {str(N): a[dd][N].tolist() for N in N_LIST} for dd in d_list}}
                              for x, p, a in dyn[w]]}
    import gzip

    (out_dir / "per-position.json.gz").write_bytes(gzip.compress(json.dumps(raw).encode(), mtime=0))

    print(f"\nGate: the chain from this script's ranks against eagle_alpha.py's replay, rounds / tokens, k = 1 / 2 / 3")
    ok = True
    for w in WORKLOADS:
        cells = []
        for k in (1, 2, 3):
            r_ea, e_ea, r_me, e_me = chain_gate[w][k]
            same = (r_ea, e_ea) == (r_me, e_me)
            ok &= same
            cells.append(f"{e_ea / max(1, r_ea):.2f} ({r_ea}/{e_ea}) {'=' if same else '!='} ({r_me}/{e_me})")
        print(f"  {w:8s} " + "   ".join(cells))
    print(f"  chain gate: {'HOLDS' if ok else 'FAILS'}")

    def table(title, fn):
        print(f"\n{title}")
        print("  " + " " * 8 + "".join(f"{N:>7d}" for N in N_LIST))
        for w in WORKLOADS:
            print(f"  {w:8s}" + "".join(f"{fn(w, N):7.2f}" for N in N_LIST))

    def dyn_tpv(w, N, dd):
        r = e = 0
        for x, p, a in dyn[w]:
            rr_, ee = rounds_of(a[dd][N], x, p)
            r += rr_
            e += ee
        return e / max(1, r)

    print(f"\nTokens a verify pass (accepted drafts + 1), by workload and tree size N (drafted nodes; the verify "
          f"pass has N + 1 rows). GSM8K over {n_gsm} problems. {n_tok} completion tokens in all.")
    for dd in d_list:
        table(f"dynamic (EAGLE-2), K = {K}, depth {dd} = {dd} head steps a round", lambda w, N, dd=dd: dyn_tpv(w, N, dd))

    freqs = {w: prefix_freqs(rank_rows[w]) for w in WORKLOADS}
    depth_of = {}

    def stat_tpv(w, N):
        tree = static_tree([freqs[v] for v in WORKLOADS if v != w], N)
        depth_of[(w, N)] = max(len(k) for k in tree)
        r = e = 0
        for (x, p, _), rr in zip(dyn[w], rank_rows[w]):
            rr_, ee = rounds_of(static_accept(rr, tree), x, p)
            r += rr_
            e += ee
        return e / max(1, r)

    table("static, leave one workload out (the N most frequent rank paths of the other five workloads)", stat_tpv)
    print("  depth of each static tree (head steps a round):")
    print("  " + " " * 8 + "".join(f"{N:>7d}" for N in N_LIST))
    for w in WORKLOADS:
        print(f"  {w:8s}" + "".join(f"{depth_of[(w, N)]:7d}" for N in N_LIST))

    def chain_tpv(w, k):
        r = e = 0
        for (x, p, _), rr in zip(dyn[w], rank_rows[w]):
            lead = np.zeros(rr.shape[1], dtype=np.int64)
            for col in range(rr.shape[1]):
                a = 0
                while a < k and a < rr.shape[0] and rr[a, col] == 0:
                    a += 1
                lead[col] = a
            rr_, ee = rounds_of(lead, x, p)
            r += rr_
            e += ee
        return e / max(1, r)

    print("\nchain (top-1 path), k drafts = k head steps a round, k = 1..8")
    for w in WORKLOADS:
        print(f"  {w:8s}" + "".join(f"{chain_tpv(w, k):7.2f}" for k in range(1, 9)))
    print(f"done in {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
