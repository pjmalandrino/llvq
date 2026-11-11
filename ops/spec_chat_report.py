"""Tables, gates and projected tok/s for ops/spec_chat.py's chat runs (stdlib only, 2026-11-11).

    python3 ops/spec_chat_report.py <brut_dir> <m3_house_a12ip_log>

Reads <brut>/dflash/{rows.json,positions.json.gz} and <brut>/tables/passes.json.gz. Writes nothing: prints the report.

Costs (card, L40S, measured 2026-11-11, spec-mem-m3-2026-11-11-brut): 4E greedy 4.46 ms a token; DFlash drafter
round 4.04 ms; table drafter 0.75 ms; verify a12ip 9.56 ms at 1 row and 10.9 ms at 16 rows, a12 8.58 ms at 16 rows.
A pass of R rows (root + drafted) is interpolated linearly between the 1-row and 16-row times (an interpolation, not a
measurement). a12 at 1 row was never measured: it is a12ip's 1-row time scaled by 8.58 / 10.9 (estimated).

Halves: MT-Bench in file order, even index = train (questions 81, 83, ...), odd = test; 5 a category in each. Every
threshold is chosen on train for the highest tok/s and frozen; the house chat 10 are a second held-out set.
"""
from __future__ import annotations

import gzip
import itertools
import json
import math
import re
import statistics
import sys
from pathlib import Path

T4E, DR_DF, DR_TAB = 4.46, 4.04, 0.75
V = {"a12ip": (9.56, 10.9), "a12": (9.56 * 8.58 / 10.9, 8.58)}
BPP = {"ref": 4.87, "int4": 3.30, "tab": 2.73}
CATS = ["writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities", "house-chat"]


def verify(arm, rows):
    v1, v16 = V[arm]
    return v1 + (rows - 1) * (v16 - v1) / 15


def replay(acc, choose, nospec):
    """acc: {N: list per position}; choose(r) -> N (0 = one 4E token). Returns passes [(N, accepted)], and time per
    verify arm is computed by the caller. nospec is unused here (cost only)."""
    n = len(acc[15])
    m, out = 0, []
    while m < n:
        N = choose(m)
        if N == 0:
            out.append((0, 0))
            m += 1
            continue
        e = min(acc[N][m] + 1, n - m)
        out.append((N, e - 1))
        m += e
    return out


def cost(passes, arm, dr, nospec_cost):
    t = 0.0
    for N, _ in passes:
        t += nospec_cost if N == 0 else dr + verify(arm, N + 1)
    return t


def tokps(group, arm, dr, nospec_cost):
    tok = sum(len(p) + sum(a for _, a in p) for p in group)
    return 1000.0 * tok / sum(cost(p, arm, dr, nospec_cost) for p in group)


def acc_rate(group):
    sp = [a for p in group for N, a in p if N > 0]
    return sum(sp) / len(sp) if sp else float("nan")


def hist(group):
    h = [0] * 16
    for p in group:
        for N, a in p:
            if N > 0:
                h[a] += 1
    return h


def med_range(group):
    v = [sum(a for N, a in p if N > 0) / max(1, sum(1 for N, _ in p if N > 0)) for p in group]
    return f"{statistics.median(v):.3f} [{min(v):.3f}-{max(v):.3f}]"


def quant(vals, qs):
    s = sorted(vals)
    return sorted({s[min(len(s) - 1, int(q * (len(s) - 1)))] for q in qs})


def main(argv):
    brut = Path(argv[1])
    rows = json.loads((brut / "dflash" / "rows.json").read_text())
    pos = json.loads(gzip.decompress((brut / "dflash" / "positions.json.gz").read_bytes()))
    meta = {f"{r['set']}/{r['id']}": r for r in rows}
    keys = list(pos)
    mtb = [k for k in keys if k.startswith("mtbench/")]
    train, test = mtb[0::2], mtb[1::2]
    house = [k for k in keys if k.startswith("house/")]
    cat = {k: meta[k]["category"] for k in keys}

    # drafter wall split on the card, chat, a12ip: what a depth-1-only pre-check would still pay
    wall = [0.0] * 4
    for m in re.finditer(r"== chat-\d+.*?drafter wall, ms: context extension ([\d.]+), block ([\d.]+), head ([\d.]+), "
                         r"readout ([\d.]+)", Path(argv[2]).read_text(), re.S):
        wall = [w + float(g) for w, g in zip(wall, m.groups())]
    f_pre = (wall[0] + wall[1] + (wall[2] + wall[3]) / 15) / sum(wall)
    print(f"card drafter wall, chat 10, a12ip (measured): context {wall[0]:.1f}, block {wall[1]:.1f}, head "
          f"{wall[2]:.1f}, readout {wall[3]:.1f} ms; a depth-1-only pre-check keeps context + block + 1/15 of head and "
          f"readout = {f_pre:.3f} of the round (computed), {f_pre * DR_DF:.2f} ms of 4.04")
    print(f"texts: MT-Bench {len(mtb)} prompts, {sum(meta[k]['tokens'] for k in mtb)} tokens after `first`; "
          f"teacher-forced agreement min {min(meta[k]['tf_agree'] for k in keys):.4f}; house chat {len(house)}")

    # ------------------------------------------------ fixed arms
    def fixed(drafter, name):
        if name.startswith("tree"):
            N = int(name[4:])
            a = {N: None}
            return {k: replay({N: pos[k][drafter][name], 15: pos[k][drafter][name]}, lambda m, N=N: N, None)
                    for k in keys}
        c = int(name[5:])
        return {k: replay({c: [min(d, c) for d in pos[k][drafter]["chain15"]],
                           15: pos[k][drafter]["chain15"]}, lambda m, c=c: c, None) for k in keys}

    print("\n== FIXED ARMS: accepted drafted tokens a pass (measured), tok/s (estimated: a12 / a12ip)")
    print(f"greedy 4E: {1000 / T4E:.1f} tok/s (card, chat, measured 224.0)")
    arms = [("ref", "tree15"), ("int4", "tree15"), ("ref", "tree7"), ("ref", "tree3"), ("ref", "tree1"),
            ("ref", "chain1"), ("ref", "chain2"), ("ref", "chain3"), ("ref", "chain7"), ("ref", "chain15")]
    fx = {}
    for d, nm in arms:
        P = fixed(d, nm)
        fx[(d, nm)] = P
        allm = [P[k] for k in mtb]
        hs = [P[k] for k in house]
        print(f"{d:4s} {nm:7s} MT80 {acc_rate(allm):.3f} med {med_range(allm)}  {tokps(allm, 'a12', DR_DF, T4E):6.1f} / "
              f"{tokps(allm, 'a12ip', DR_DF, T4E):6.1f}   house10 {acc_rate(hs):.3f} {tokps(hs, 'a12', DR_DF, T4E):6.1f} / "
              f"{tokps(hs, 'a12ip', DR_DF, T4E):6.1f}  b/param {BPP[d]}")
    for d, nm in [("ref", "tree15"), ("int4", "tree15")]:
        P = fx[(d, nm)]
        print(f"\n{d} {nm} by category: acc a pass, per-prompt median [range], tok/s a12 / a12ip")
        for c in CATS:
            g = [P[k] for k in keys if cat[k] == c]
            print(f"  {c:11s} {acc_rate(g):.3f}  {med_range(g)}  {tokps(g, 'a12', DR_DF, T4E):6.1f} / "
                  f"{tokps(g, 'a12ip', DR_DF, T4E):6.1f}")
        print(f"  histogram 0..15, MT80: {hist([P[k] for k in mtb])}")
    print("\nchains and small trees by category (ref), acc a pass:")
    print("  " + " ".join(f"{nm:>7s}" for _, nm in arms[2:]))
    for c in CATS:
        print(f"  {c:11s} " + " ".join(f"{acc_rate([fx[a][k] for k in keys if cat[k] == c]):7.3f}" for a in arms[2:]))
    print("  histograms MT80 (ref): " + "; ".join(f"{nm} {hist([fx[('ref', nm)][k] for k in mtb])[:8]}"
                                                  for nm in ("chain1", "chain3", "chain7", "tree7")))

    # ------------------------------------------------ gates
    feats = {"lp1": +1, "margin": +1, "exp_tree15": +1, "exp_chain15": +1, "tgt_p1": +1}

    def fv(k, d, f):
        p = pos[k]
        if f == "tgt_p1":
            return p["tgt_p1"]
        if f == "margin":
            return [a - b for a, b in zip(p[d]["lp1"], p[d]["lp2"])]
        return p[d][f]

    variants = {"round": T4E + DR_DF, "precheck": T4E + f_pre * DR_DF, "free": T4E}
    feat_var = {"lp1": ["round", "precheck"], "margin": ["round", "precheck"], "exp_tree15": ["round"],
                "exp_chain15": ["round"], "tgt_p1": ["free"]}

    def run_policy(ks, d, f, thr, Ns=(15,)):
        """thr: ascending thresholds; option i chosen when feature in [thr[i-1], thr[i]); options (0,) + Ns."""
        out = []
        for k in ks:
            v = fv(k, d, f)
            acc = {N: pos[k][d][f"tree{N}"] for N in (1, 3, 7, 15)}
            opts = (0,) + tuple(Ns)

            def ch(m):
                i = 0
                while i < len(thr) and v[m] >= thr[i]:
                    i += 1
                return opts[i]

            out.append(replay(acc, ch, None))
        return out

    print("\n== GATE (no speculation below a threshold: one 4E token). Threshold chosen on train (MT-Bench even), "
          "frozen, reported on test (odd) and house 10. tok/s estimated; no-spec cost per variant: "
          + ", ".join(f"{a} {b:.2f} ms" for a, b in variants.items()))
    best = {}
    for d in ("ref", "int4"):
        for va in ("a12", "a12ip"):
            for f, vlist in feat_var.items():
                vals = [x for k in train for x in fv(k, d, f)]
                grid = [-math.inf] + quant(vals, [i / 40 for i in range(41)])
                for vn in vlist:
                    nc = variants[vn]
                    sc = []
                    for t in grid:
                        P = run_policy(train, d, f, [t])
                        sc.append((tokps(P, va, DR_DF, nc), t))
                    s_tr, t = max(sc)
                    Pt = run_policy(test, d, f, [t])
                    Ph = run_policy(house, d, f, [t])
                    Pall = run_policy(mtb, d, f, [t])
                    base = tokps(run_policy(test, d, f, [-math.inf]), va, DR_DF, nc)
                    frac = sum(1 for p in Pt for N, _ in p if N == 0) / sum(len(p) for p in Pt)
                    r = (tokps(Pt, va, DR_DF, nc), s_tr, t, tokps(Ph, va, DR_DF, nc), base, frac,
                         tokps(Pall, va, DR_DF, nc))
                    best[(d, va, f, vn)] = r
                    print(f"  {d:4s} {va:5s} {f:11s} {vn:8s} thr {t:8.4f}: train {s_tr:6.1f}  TEST {r[0]:6.1f} "
                          f"(ungated {base:6.1f}, {100 * frac:4.1f} % 4E tokens)  house10 {r[3]:6.1f}  MT80 {r[6]:6.1f}")

    # ------------------------------------------------ per prompt and per category, frozen policies
    def ema_policy(ks, d, theta, probe, alpha=0.5):
        """Speculate (tree 15) while an EMA of accepted counts stays >= theta; else `probe` 4E tokens, then one pass."""
        out = []
        for k in ks:
            acc = pos[k][d]["tree15"]
            n, m, e_, wait, ps = len(acc), 0, 99.0, 0, []
            while m < n:
                if e_ < theta and wait < probe:
                    ps.append((0, 0))
                    m += 1
                    wait += 1
                    continue
                e = min(acc[m] + 1, n - m)
                ps.append((15, e - 1))
                e_ = alpha * (e - 1) + (1 - alpha) * min(e_, 15.0)
                wait = 0
                m += e
            out.append(ps)
        return out

    print("\n== EMA PROBE POLICY (tree 15 while the EMA (alpha 0.5) of accepted counts >= theta, else `probe` 4E "
          "tokens then one probe pass; no drafter round on 4E tokens; theta and probe chosen on train)")
    ema_best = {}
    for d in ("ref", "int4"):
        for va in ("a12", "a12ip"):
            sc = []
            for theta in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
                for probe in (2, 4, 8, 16, 32):
                    sc.append((tokps(ema_policy(train, d, theta, probe), va, DR_DF, T4E), theta, probe))
            s, theta, probe = max(sc)
            ema_best[(d, va)] = (theta, probe)
            Pt, Ph = ema_policy(test, d, theta, probe), ema_policy(house, d, theta, probe)
            fr = sum(1 for p in Pt for N, _ in p if N == 0) / sum(len(p) for p in Pt)
            print(f"  {d:4s} {va:5s} theta {theta} probe {probe}: train {s:6.1f}  TEST {tokps(Pt, va, DR_DF, T4E):6.1f} "
                  f"({100 * fr:4.1f} % 4E passes)  house10 {tokps(Ph, va, DR_DF, T4E):6.1f}")

    g = 1000 / T4E
    for va in ("a12", "a12ip"):
        print(f"\nbreak-even accepted a pass, tree 15, {va}: DFlash {(DR_DF + verify(va, 16)) / T4E - 1:.2f}, tables "
              f"{(DR_TAB + verify(va, 16)) / T4E - 1:.2f} (computed)")
    print(f"\n== PER CATEGORY AND PER PROMPT, frozen policies on all 90 prompts (tok/s estimated; x = against greedy "
          f"{g:.1f}); 'slower' = prompts under greedy")
    pols = []
    for d in ("ref", "int4"):
        for va in ("a12", "a12ip"):
            pols.append((f"{d} {va} ungated", d, va, lambda ks, d=d: [fx[(d, 'tree15')][k] for k in ks], T4E + DR_DF))
            t = best[(d, va, "tgt_p1", "free")][2]
            pols.append((f"{d} {va} tgt_p1>={t:.3f} free", d, va,
                         lambda ks, d=d, t=t: run_policy(ks, d, "tgt_p1", [t]), T4E))
            t = best[(d, va, "lp1", "precheck")][2]
            pols.append((f"{d} {va} lp1>={t:.3f} precheck", d, va,
                         lambda ks, d=d, t=t: run_policy(ks, d, "lp1", [t]), variants["precheck"]))
            be = (DR_DF + verify(va, 16)) / T4E - 1

            def oracle(ks, d=d, be=be):
                return [replay({15: pos[k][d]["tree15"]}, lambda m, a=pos[k][d]["tree15"]: 15 if a[m] >= be else 0,
                               None) for k in ks]

            pols.append((f"{d} {va} ORACLE gate (bound)", d, va, oracle, T4E))
            th, pr = ema_best[(d, va)]
            pols.append((f"{d} {va} ema {th}/{pr}", d, va, lambda ks, d=d, th=th, pr=pr: ema_policy(ks, d, th, pr), T4E))
    print("  policy                              " + " ".join(f"{c[:6]:>6s}" for c in CATS) + "  MT80  slower  min x")
    for name, d, va, fn, nc in pols:
        cells = [tokps(fn([k for k in keys if cat[k] == c]), va, DR_DF, nc) / g for c in CATS]
        per = [tokps(fn([k]), va, DR_DF, nc) / g for k in keys]
        print(f"  {name:36s}" + " ".join(f"{x:6.3f}" for x in cells) +
              f" {tokps(fn(mtb), va, DR_DF, nc) / g:5.3f}  {sum(1 for x in per if x < 1):2d}/90  {min(per):.3f}")

    print("\n== ADAPTIVE TREE (0 = 4E token, else 1, 3, 7 or 15 drafted rows by the same feature; thresholds on train)")
    for d in ("ref", "int4"):
        for va in ("a12", "a12ip"):
            for f, vn in (("exp_tree15", "round"), ("tgt_p1", "free"), ("lp1", "precheck")):
                nc = variants[vn]
                vals = [x for k in train for x in fv(k, d, f)]
                grid = [-math.inf] + quant(vals, [i / 8 for i in range(9)]) + [math.inf]
                bestc = None
                for thr in itertools.combinations_with_replacement(grid, 4):
                    P = run_policy(train, d, f, list(thr), Ns=(1, 3, 7, 15))
                    s = tokps(P, va, DR_DF, nc)
                    if bestc is None or s > bestc[0]:
                        bestc = (s, thr)
                s, thr = bestc
                Pt = run_policy(test, d, f, list(thr), Ns=(1, 3, 7, 15))
                Ph = run_policy(house, d, f, list(thr), Ns=(1, 3, 7, 15))
                mix = [0] * 5
                for p in Pt:
                    for N, _ in p:
                        mix[(0, 1, 3, 7, 15).index(N)] += 1
                print(f"  {d:4s} {va:5s} {f:10s} {vn:8s} thr {[round(t, 3) for t in thr]}: train {s:6.1f}  TEST "
                      f"{tokps(Pt, va, DR_DF, nc):6.1f}  house10 {tokps(Ph, va, DR_DF, nc):6.1f}  test passes by "
                      f"N 0/1/3/7/15 {mix}")

    # ------------------------------------------------ tables
    tp = brut / "tables" / "passes.json.gz"
    if tp.exists():
        T = json.loads(gzip.decompress(tp.read_bytes()))
        names = list(next(iter(T.values())))
        print("\n== TABLE DRAFTERS (exact verify rows; drafter 0.75 ms a pass, gated token 4.46 ms). b/param 2.73")

        def tps(ks, nm, va):
            tok = t = 0.0
            for k in ks:
                for rws, a, _ in T[k][nm]:
                    tok += a + 1
                    t += T4E if rws == 0 else DR_TAB + verify(va, rws)
            return 1000 * tok / t

        def tacc(ks, nm):
            sp = [a for k in ks for rws, a, _ in T[k][nm] if rws > 0]
            return sum(sp) / len(sp) if sp else float("nan")

        for nm in names:
            fr = sum(1 for k in mtb for q in T[k][nm] if q[0] == 0) / sum(len(T[k][nm]) for k in mtb)
            rws = [q[0] for k in mtb for q in T[k][nm] if q[0] > 0]
            pp = [sum(q[1] for q in T[k][nm] if q[0]) / max(1, sum(1 for q in T[k][nm] if q[0])) for k in mtb]
            print(f"  {nm:10s} MT80 acc/spec-pass {tacc(mtb, nm):.3f} med {statistics.median(pp):.3f} "
                  f"[{min(pp):.3f}-{max(pp):.3f}] mean rows {sum(rws) / max(1, len(rws)):5.2f}  4E share "
                  f"{100 * fr:4.1f} %  tok/s a12 {tps(mtb, nm, 'a12'):6.1f} a12ip {tps(mtb, nm, 'a12ip'):6.1f}  "
                  f"house10 acc {tacc(house, nm):.3f} {tps(house, nm, 'a12'):6.1f} / {tps(house, nm, 'a12ip'):6.1f}")
        h = [0] * 16
        for k in mtb:
            for rws, a, _ in T[k]["comb4w"]:
                h[a] += 1
        print(f"  comb4w histogram MT80: {h}")
        byn = {}
        for k in mtb:
            for rws, a, n in T[k]["comb4w"]:
                byn.setdefault(min(n, 6), []).append(a)
        print("  comb4w acc a pass by lookup match length n (MT80): "
              + ", ".join(f"n={n}{'+' if n == 6 else ''} {sum(v) / len(v):.3f} ({len(v)})" for n, v in sorted(byn.items())))
        print("  comb4w by category: " + ", ".join(
            f"{c} {tacc([k for k in keys if cat[k] == c], 'comb4w'):.3f}" for c in CATS))
        for va in ("a12", "a12ip"):
            sc = max((tps(train, nm, va), nm) for nm in names)
            print(f"  gate chosen on train, {va}: {sc[1]} (train {sc[0]:.1f}) -> TEST {tps(test, sc[1], va):.1f}, "
                  f"house10 {tps(house, sc[1], va):.1f}; ungated comb4w TEST {tps(test, 'comb4w', va):.1f}")
            for nm in ("comb4w", sc[1]):
                cells = [tps([k for k in keys if cat[k] == c], nm, va) / g for c in CATS]
                per = [tps([k], nm, va) / g for k in keys]
                print(f"    {nm:10s} x by category " + " ".join(f"{c[:6]} {x:.3f}" for c, x in zip(CATS, cells)) +
                      f"; slower {sum(1 for x in per if x < 1)}/90, min x {min(per):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
