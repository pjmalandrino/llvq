# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Projection of a tree verify on 4E from the measured tokens a pass (2026-11-09), stdlib only.

    uv run ops/tree_projection.py docs/mesures/eagle-tree-2026-11-09-brut/per-position.json.gz

Reads the per-position output of `ops/eagle_tree.py` (ranks along the true path, and the dynamic EAGLE-2 accepts),
replays the rounds for each tree, and projects decode tok/s = tau / (c_R + d * D) x greedy decode, where tau is the
measured tokens a pass, c_R the verify pass of R rows in greedy 4E tokens (*estimated*, docs/plan-verify-tree.md
section 2), d the head steps a round and D one head step in greedy tokens (*estimated*). Greedy decode tok/s are
spec-final-2026-11-08.txt's (*measured*). The 8B and 14B reuse the 4B's tau: no head was run at those sizes.
"""
from __future__ import annotations

import gzip
import json
import sys

WORKLOADS = ("chat", "code", "summary", "edit", "rag", "gsm8k")
RANK_CAP = 32

# Greedy decode tok/s, spec-final-2026-11-08.txt section 2 (*measured*).
GREEDY = {
    "4B": dict(chat=221.0, code=220.5, summary=212.3, edit=218.9, rag=210.9, gsm8k=216.9),
    "8B": dict(chat=141.3, code=140.8, summary=136.2, edit=137.6, rag=135.5, gsm8k=138.0),
    "14B": dict(chat=89.1, code=88.2, summary=85.3, edit=88.1, rag=85.3, gsm8k=86.3),
}
# c_R, the verify pass of R rows in greedy 4E tokens: low, central, high (*estimated*, plan-verify-tree.md section 2).
C_R = {4: (0.97, 1.13, 1.32), 8: (0.99, 1.18, 1.41), 16: (1.03, 1.27, 1.60), 32: (1.14, 1.56, 2.03)}
# One head step in greedy tokens: int4 g64 head bytes at 550-700 GB/s plus a step's fixed cost (*estimated*).
D_STEP = (0.04, 0.05, 0.07)


def rounds_of(accept, n, p, per_round=None):
    m, rounds = p, 0
    while m < n - 1:
        e = min(int(accept[m - p]) + 1, n - 1 - m)
        if per_round is not None:
            per_round.append(e)
        m += e
        rounds += 1
    return rounds, n - 1 - p


def lead_zeros(rr, k):
    out = []
    for col in range(len(rr[0])):
        a = 0
        while a < k and a < len(rr) and rr[a][col] == 0:
            a += 1
        out.append(a)
    return out


def prefix_freqs(seqs):
    cnt, tot = {}, 0
    for rr in seqs:
        for col in range(len(rr[0])):
            tot += 1
            pre = ()
            for j in range(len(rr)):
                v = rr[j][col]
                if v >= RANK_CAP:
                    break
                pre = pre + (v,)
                cnt[pre] = cnt.get(pre, 0) + 1
    return {k: v / tot for k, v in cnt.items()}


def static_tree(freqs, N):
    keys = set().union(*[f.keys() for f in freqs])
    mean = {k: sum(f.get(k, 0.0) for f in freqs) / len(freqs) for k in keys}
    return set(sorted(mean, key=lambda k: (-mean[k], len(k), k))[:N])


def static_accept(rr, tree):
    out = []
    for col in range(len(rr[0])):
        pre, a = (), 0
        for j in range(len(rr)):
            pre = pre + (rr[j][col],)
            if pre not in tree:
                break
            a += 1
        out.append(a)
    return out


def main(argv):
    raw = json.loads(gzip.decompress(open(argv[1], "rb").read()))
    tau = {}  # (workload, config) -> (tau, d, R)

    se = {}

    def put(w, cfg, accepts, d, R):
        r = e = 0
        per = []
        for dyn, acc in zip(raw[w]["dynamic"], accepts):
            rr_, ee = rounds_of(acc, dyn["n"], dyn["p"], per)
            r += rr_
            e += ee
        tau[(w, cfg)] = (e / max(1, r), d, R)
        if len(per) < 2:
            se[(w, cfg)] = float("nan")
            return
        mu = sum(per) / len(per)
        se[(w, cfg)] = (sum((v - mu) ** 2 for v in per) / (len(per) - 1)) ** 0.5 / len(per) ** 0.5

    freqs = {w: prefix_freqs(raw[w]["ranks"]) for w in WORKLOADS}
    for w in WORKLOADS:
        ranks = raw[w]["ranks"]
        for k in (1, 3):
            put(w, f"chain k={k}", [lead_zeros(rr, k) for rr in ranks], k, k + 1)
        for N, R in ((7, 8), (15, 16), (31, 32)):
            for d in (3, 4, 5, 6):
                put(w, f"dynamic N={N} d={d}", [dyn["acc"][str(d)][str(N)] for dyn in raw[w]["dynamic"]], d, R)
            tree = static_tree([freqs[v] for v in WORKLOADS if v != w], N)
            put(w, f"static N={N}", [static_accept(rr, tree) for rr in ranks], max(len(t) for t in tree), R)

    cfgs = ["chain k=1", "chain k=3"] + [f"dynamic N={N} d={d}" for N in (7, 15, 31) for d in (3, 4, 5, 6)] + \
           [f"static N={N}" for N in (7, 15, 31)]
    print("Positions (rounds of greedy decoding) per workload: " + ", ".join(
        f"{w} {sum(d['n'] - 1 - d['p'] for d in raw[w]['dynamic'])}" for w in WORKLOADS))
    print("Tokens a pass tau (head steps a round d, verify rows R), measured on the sealed 4B through the MLX door;")
    print("standard error of tau over rounds in the second block (computed, rounds treated as independent)")
    print(f"  {'config':22s} {'R':>3s} " + "".join(f"{w:>14s}" for w in WORKLOADS))
    for c in cfgs:
        cells = "".join(f"{tau[(w, c)][0]:9.2f} (d{tau[(w, c)][1]})" for w in WORKLOADS)
        print(f"  {c:22s} {tau[('chat', c)][2]:3d} {cells}")
    print(f"  {'standard error':22s}")
    for c in cfgs:
        print(f"  {c:22s}     " + "".join(f"{se[(w, c)]:14.3f}" for w in WORKLOADS))

    def speed(w, c, ci, di):
        t, d, R = tau[(w, c)]
        return t / (C_R[R][ci] + d * D_STEP[di])

    print("\nDecode speed over greedy, x, central c_R and D = 0.05 [low c_R and D = 0.04 ; high c_R and D = 0.07]")
    print(f"  {'config':22s} " + "".join(f"{w:>22s}" for w in WORKLOADS))
    for c in cfgs[1:]:
        cells = "".join(f"  x{speed(w, c, 1, 1):4.2f} [{speed(w, c, 0, 0):4.2f};{speed(w, c, 2, 2):4.2f}]" for w in WORKLOADS)
        print(f"  {c:22s} {cells}")

    # Two trees fixed before reading the tok/s, the same at every workload: no per-workload choice on these data.
    for c in ("dynamic N=7 d=3", "dynamic N=15 d=4"):
        print(f"\nDecode tok/s, {c} ({tau[('chat', c)][2]} rows), central [high c_R ; low c_R], 4B tau reused at 8B "
              f"and 14B")
        for size, g in GREEDY.items():
            cells = [f"{w} {g[w] * speed(w, c, 1, 1):5.1f} [{g[w] * speed(w, c, 2, 2):5.1f}-{g[w] * speed(w, c, 0, 0):5.1f}]"
                     for w in WORKLOADS]
            print(f"  {size:4s} " + " | ".join(cells))

    print("\nKill rule (task): 4B chat over greedy with c_16 = 1.25 and D = 0.04, at every 16-row tree")
    for c in cfgs[1:]:
        t, d, R = tau[("chat", c)]
        if R == 16:
            print(f"  {c:22s} tau {t:.2f}, d {d}: x{t / (1.25 + d * 0.04):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
