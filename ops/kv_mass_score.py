# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0"]
# ///
"""Score one KV capture of `LLVQ_KV_CAPTURE` (llvq-llm/src/kvcapture.rs): how many KV bytes a decode row
must read so that the softmax mass it skips stays under a budget, head by head.

    uv run ops/kv_mass_score.py <capture_dir> [--out PREFIX] [--sinks 64] [--recent 256]
    uv run ops/kv_mass_score.py --selftest
    uv run ops/kv_mass_score.py --rule a.json b.json c.json     # the prereg's reading over three texts

What a row attends (read from `Block::attention_ctx`, written in meta.json): captured token t sits at position
ctx + t and attends positions 0..=ctx+t, its own key included, so L(t) = ctx + t + 1. The score of key j for query
head h is c <q_h, k_j>, c = 1/sqrt(128), natural softmax. Query head h reads KV head h // groups.

Layout of one row's positions:
  dense   positions [0, S) (sinks), and [S + n P, L) where n = (L - R - S) // P: the recent R positions plus the
          fewer than P left over between the last full page and them;
  pages   n pages of P positions, page i = [S + i P, S + (i + 1) P), aligned to absolute positions.
A page is skipped for a KV head only if it is skipped for all its query heads.

Selectors (pages read, for a mass budget delta and a page size P):
  true-mass  the oracle of oracles: pages in increasing order of their true mass (max over the heads); the longest
             prefix whose summed mass stays <= delta in every head is skipped.
  page-max   the best any upper bound can do: bound on the mass of a page for head h is
             P exp(s_max_page - m) / Z_lo, s_max_page the page's exact max score, Z_lo the partition sum over what
             is read, m its shift. Pages are read in decreasing max_h P exp(s_max - m) / Z_dense until the bound on
             the rest is <= delta in every head.
  box        Quest's bound: per page and channel the min and max of K; U = c sum_i max(q_i min_i, q_i max_i);
             same rule as page-max with U for s_max. Charged 2 x 128 f16 a page a KV head.
  ball       centroid mu and radius rho of the page's keys (computed in f64; a stored f16 summary would round
             them outward); U = c (<q, mu> + |q| rho); charged 129 f16 a page a KV head.
Bytes read = 512 B a position a KV head (K and V, binary16, head dim 128) over the dense positions and the read
pages, plus the summaries of every page; over 512 B x L(t). Pooled over layers and KV heads by bytes, then median,
p10 and p90 over tokens.

The assertion that guards the bounds: for every row, head, selector, P and delta, the actual skipped mass
(sum over skipped pages of exp(s - m) / Z_full, all in f64) is <= the bound the selector used, times (1 + 1e-9)
plus 1e-300, and the bound is <= delta. Scores and bounds are computed in float64 from the binary16 values, so the
rounding of a score is about 1e-13 of its size; the 1e-9 margin covers the sum of P equal exponentials against
P times one exponential in the uniform case.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

SELECTORS = ("true-mass", "page-max", "box", "ball")
PAGES = (16, 32, 64)
DELTAS = (1e-2, 1e-3, 1e-4)
POS_BYTES = 2 * 128 * 2  # K and V rows of one position, one KV head, binary16
SUMMARY_BYTES = {"true-mass": 0, "page-max": 0, "box": 2 * 128 * 2, "ball": 129 * 2}
MARGIN_REL, MARGIN_ABS = 1e-9, 1e-300


class BoundViolation(AssertionError):
    pass


def load_capture(d: Path):
    meta = json.loads((d / "meta.json").read_text())
    nl, nh, hd, T = meta["n_layers"], meta["n_heads"], meta["head_dim"], meta["tokens"]
    q = np.fromfile(d / "q.bin", dtype="<f2")
    assert q.size == T * nl * nh * hd, f"q.bin holds {q.size} values, expected {T}x{nl}x{nh}x{hd}"
    return meta, q.reshape(T, nl, nh, hd)


def load_layer(d: Path, meta, layer: int):
    nkv, hd, L = meta["n_kv"], meta["head_dim"], meta["positions"]
    k = np.fromfile(d / f"k-{layer}.bin", dtype="<f2")
    assert k.size == nkv * L * hd, f"k-{layer}.bin holds {k.size} values"
    vn = np.fromfile(d / f"vnorm-{layer}.bin", dtype="<f4")
    assert vn.size == nkv * L, f"vnorm-{layer}.bin holds {vn.size} values"
    return k.reshape(nkv, L, hd), vn.reshape(nkv, L)


def self_check(d: Path, meta, q):
    """The engine's attention output of the last token at the check layer, recomputed from the dump."""
    nkv, nh, hd, L = meta["n_kv"], meta["n_heads"], meta["head_dim"], meta["positions"]
    lay, t = meta["check"]["layer"], meta["check"]["token"]
    groups = nh // nkv
    k, _ = load_layer(d, meta, lay)
    v = np.fromfile(d / "v-check.bin", dtype="<f2").reshape(nkv, L, hd).astype(np.float64)
    out = np.fromfile(d / "out-check.bin", dtype="<f2").reshape(nh, hd).astype(np.float64)
    c = meta["c"]
    n = meta["ctx"] + t + 1
    res = {}
    for name, length in (("0..=p", n), ("0..p", n - 1)):
        o = np.zeros((nh, hd))
        for h in range(nh):
            g = h // groups
            s = c * (k[g, :length].astype(np.float64) @ q[t, lay, h].astype(np.float64))
            w = np.exp(s - s.max())
            o[h] = (w / w.sum()) @ v[g, :length]
        res[name] = float(np.abs(o - out).max() / np.abs(out).max())
    # the previous token's query, as a check that the query matters
    o = np.zeros((nh, hd))
    for h in range(nh):
        g = h // groups
        s = c * (k[g, :n].astype(np.float64) @ q[t - 1, lay, h].astype(np.float64))
        w = np.exp(s - s.max())
        o[h] = (w / w.sum()) @ v[g, :n]
    res["prev_q"] = float(np.abs(o - out).max() / np.abs(out).max())
    return res


def mm(a, b):
    """[T, H, d] x [n, d] -> [T, H, n], by BLAS."""
    T, H, d = a.shape
    return (a.reshape(T * H, d) @ b.T).reshape(T, H, b.shape[0])


def page_geometry(Ls, S, R, P):
    """Pages per row and the dense count per row, for row lengths `Ls`."""
    n = np.maximum(0, (Ls - R - S) // P)
    dense = Ls - n * P
    return n, dense


def score_group(s, Ls, S, R, kg, qg, c, mutant=None):
    """One layer and KV head. s: [T, H, Lmax] f64 scores (invalid positions are ignored by masks);
    kg: [Lmax, hd] f64 keys; qg: [T, H, hd] f64 queries. Returns {(sel, P, delta): (pages_read[T], dense[T],
    n_pages[T])} and the skipped masses for the output-error bound."""
    T, H, Lmax = s.shape
    pos = np.arange(Lmax)
    valid = pos[None, :] < Ls[:, None]  # [T, Lmax]
    sm = np.where(valid[:, None, :], s, -np.inf)
    smax = sm.max(axis=2, keepdims=True)  # [T, H, 1]
    e = np.exp(sm - smax)  # 0 at invalid positions
    out = {}
    for P in PAGES:
        npg, dense_n = page_geometry(Ls, S, R, P)
        npmax = int(npg.max())
        if npmax == 0:
            for sel in SELECTORS:
                for dl in DELTAS:
                    out[(sel, P, dl)] = (np.zeros(T, int), dense_n, npg, np.zeros((T, H)))
            continue
        lo, hi = S, S + npmax * P
        pvalid = np.arange(npmax)[None, :] < npg[:, None]  # [T, np]
        in_page = (pos[None, :] >= S) & (pos[None, :] < S + (npg * P)[:, None])  # [T, Lmax]
        dense_mask = valid & ~in_page
        Zd = (e * dense_mask[:, None, :]).sum(axis=2)  # [T, H]
        E = e[:, :, lo:hi].reshape(T, H, npmax, P).sum(axis=3) * pvalid[:, None, :]  # [T, H, np]
        Zfull = Zd + E.sum(axis=2)
        # bounds, in the same shift as e: T_term = n_p exp(U - smax)
        smax_page = sm[:, :, lo:hi].reshape(T, H, npmax, P).max(axis=3)
        kp = kg[lo:hi].reshape(npmax, P, -1)
        U = {"page-max": smax_page}
        kmin, kmax = kp.min(axis=1), kp.max(axis=1)  # [np, hd]
        qpos, qneg = np.maximum(qg, 0.0), np.minimum(qg, 0.0)
        if mutant == "box_min":
            # min where max is due: a lower bound, not an upper one
            U["box"] = c * (mm(qpos, kmin) + mm(qneg, kmax))
        else:
            U["box"] = c * (mm(qpos, kmax) + mm(qneg, kmin))
        mu = kp.mean(axis=1)  # [np, hd]
        rho = np.sqrt(((kp - mu[:, None, :]) ** 2).sum(axis=2)).max(axis=1)  # [np]
        qn = np.sqrt((qg**2).sum(axis=2))  # [T, H]
        U["ball"] = c * (mm(qg, mu) + qn[:, :, None] * rho[None, None, :])
        # true-mass
        w = E / Zfull[:, :, None]
        M = np.where(pvalid, w.max(axis=1), np.inf)  # [T, np]
        order = np.argsort(M, axis=1, kind="stable")
        wo = np.take_along_axis(w, order[:, None, :], axis=2)
        wo = np.where(np.take_along_axis(pvalid, order, axis=1)[:, None, :], wo, np.inf)
        C = np.concatenate([np.zeros((T, H, 1)), np.cumsum(wo, axis=2)], axis=2)  # [T, H, np+1]
        Cmax = C.max(axis=1)  # [T, np+1]
        for dl in DELTAS:
            ks = (Cmax <= dl).sum(axis=1) - 1  # pages skipped, a prefix
            ks = np.minimum(ks, npg)
            skipped = np.take_along_axis(C, ks[:, None, None].repeat(H, axis=1), axis=2)[:, :, 0]
            if not np.all(skipped <= dl * (1 + MARGIN_REL) + MARGIN_ABS):
                raise BoundViolation(f"true-mass: skipped {skipped.max():.3e} > delta {dl}")
            out[("true-mass", P, dl)] = (npg - ks, dense_n, npg, skipped)
        # bound selectors
        for sel in ("page-max", "box", "ball"):
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                factor = 1.0 if mutant == "no_P" else float(P)
                Tt = np.where(pvalid[:, None, :], factor * np.exp(U[sel] - smax), 0.0)  # [T, H, np]
                key = np.where(pvalid, (Tt / Zd[:, :, None]).max(axis=1), -np.inf)
            order = np.argsort(-key, axis=1, kind="stable")
            Eo = np.take_along_axis(E, order[:, None, :], axis=2)
            To = np.take_along_axis(Tt, order[:, None, :], axis=2)
            zeros = np.zeros((T, H, 1))
            Zlo = Zd[:, :, None] + np.concatenate([zeros, np.cumsum(Eo, axis=2)], axis=2)  # read k pages
            rest_T = np.concatenate([np.cumsum(To[:, :, ::-1], axis=2)[:, :, ::-1], zeros], axis=2)
            rest_E = np.concatenate([np.cumsum(Eo[:, :, ::-1], axis=2)[:, :, ::-1], zeros], axis=2)
            with np.errstate(invalid="ignore", divide="ignore"):
                bound = rest_T / Zlo  # [T, H, np+1]
            bmax = bound.max(axis=1)
            for dl in DELTAS:
                ok = bmax <= dl
                ok[np.arange(T), npg] = True  # reading every page leaves nothing to bound
                k = ok.argmax(axis=1)
                kk = k[:, None, None].repeat(H, axis=1)
                b = np.take_along_axis(bound, kk, axis=2)[:, :, 0]
                actual = np.take_along_axis(rest_E, kk, axis=2)[:, :, 0] / Zfull
                b = np.where(k[:, None] == npg[:, None], 0.0, b)
                if not np.all(actual <= b * (1 + MARGIN_REL) + MARGIN_ABS):
                    i = np.unravel_index(np.argmax(actual - b), actual.shape)
                    raise BoundViolation(
                        f"{sel} P={P} delta={dl}: actual skipped mass {actual[i]:.6e} > bound {b[i]:.6e} "
                        f"(row {i[0]}, head {i[1]})"
                    )
                if not np.all(b <= dl * (1 + MARGIN_REL)):
                    raise BoundViolation(f"{sel} P={P} delta={dl}: bound {b.max():.3e} above delta")
                out[(sel, P, dl)] = (k, dense_n, npg, actual)
    return out


def score_capture(d: Path, S=64, R=256, mutant=None, quiet=False):
    meta, q = load_capture(d)
    nl, nh, nkv = meta["n_layers"], meta["n_heads"], meta["n_kv"]
    T, ctx, c = meta["tokens"], meta["ctx"], meta["c"]
    groups = nh // nkv
    Ls = ctx + np.arange(T) + 1
    keys = [(sel, P, dl) for sel in SELECTORS for P in PAGES for dl in DELTAS]
    read = {kk: np.zeros((T, nl, nkv)) for kk in keys}
    pages_read = {kk: np.zeros((T, nl, nkv), dtype=np.int64) for kk in keys}
    total = (POS_BYTES * Ls)[:, None, None] * np.ones((1, nl, nkv))
    errb = []  # 2 x skipped x max |v|, box P=32 delta=1e-3
    for lay in range(nl):
        k, vn = load_layer(d, meta, lay)
        for g in range(nkv):
            kg = k[g].astype(np.float64)
            qg = q[:, lay, g * groups : (g + 1) * groups].astype(np.float64)  # [T, H, hd]
            s = c * mm(qg, kg)
            res = score_group(s, Ls, S, R, kg, qg, c, mutant)
            for (sel, P, dl), (kread, dense_n, npg, skipped) in res.items():
                read[(sel, P, dl)][:, lay, g] = POS_BYTES * (dense_n + P * kread) + SUMMARY_BYTES[sel] * npg
                pages_read[(sel, P, dl)][:, lay, g] = kread
                if (sel, P, dl) == ("box", 32, 1e-3):
                    vmax = np.array([vn[g, :n].max() for n in Ls], dtype=np.float64)
                    errb.append(2.0 * skipped * vmax[:, None])
        if not quiet:
            print(f"  layer {lay + 1}/{nl} scored", file=sys.stderr, flush=True)
    return meta, Ls, read, pages_read, total, errb


def stats(x):
    return float(np.median(x)), float(np.percentile(x, 10)), float(np.percentile(x, 90))


def summarize(meta, Ls, read, pages_read, total, errb, S, R):
    nl = meta["n_layers"]
    rows = {}
    for (sel, P, dl), r in read.items():
        frac = r.sum(axis=(1, 2)) / total.sum(axis=(1, 2))
        rows[f"{sel}|{P}|{dl:g}"] = {"selector": sel, "P": P, "delta": dl, "median": stats(frac)[0],
                                     "p10": stats(frac)[1], "p90": stats(frac)[2]}
    per_layer = {}
    for P in (16, 32):
        for dl in DELTAS:
            r = read[("box", P, dl)]
            fr = r.sum(axis=2) / total.sum(axis=2)  # [T, nl]
            per_layer[f"box|{P}|{dl:g}"] = [dict(zip(("median", "p10", "p90"), stats(fr[:, i]))) for i in range(nl)]
    best = {}
    for sel in SELECTORS:
        for dl in DELTAS:
            cand = [(rows[f"{sel}|{P}|{dl:g}"]["median"], P) for P in PAGES]
            m, P = min(cand)
            best[f"{sel}|{dl:g}"] = {"P": P, "median": m}
    # the orderings, in pages read, row by row
    order = {}
    for P in PAGES:
        for dl in DELTAS:
            pr = {sel: pages_read[(sel, P, dl)] for sel in SELECTORS}
            n = pr["true-mass"].size
            order[f"{P}|{dl:g}"] = {
                "true-mass > page-max": int((pr["true-mass"] > pr["page-max"]).sum()),
                "page-max > box": int((pr["page-max"] > pr["box"]).sum()),
                "page-max > ball": int((pr["page-max"] > pr["ball"]).sum()),
                "rows": n,
            }
    eb = np.concatenate([x.reshape(-1) for x in errb]) if errb else np.zeros(1)
    return {
        "capture": {kk: meta[kk] for kk in ("corpus", "window", "ctx", "tokens", "sealed_sha256", "attention")},
        "sinks": S, "recent": R, "pages": PAGES, "deltas": DELTAS,
        "L": [int(Ls[0]), int(Ls[-1])],
        "fraction_read": rows, "best_P": best, "per_layer": per_layer, "order_violations": order,
        "output_error_bound_box32_1e-3": {"what": "2 x actual skipped mass x max_j |v_j|, per (token, layer, head)",
                                           "median": float(np.median(eb)), "p90": float(np.percentile(eb, 90)),
                                           "max": float(eb.max())},
        "score_delta_1e-3": {sel: best[f"{sel}|0.001"] for sel in SELECTORS},
    }


def table(summary, check=None):
    out = []
    cap = summary["capture"]
    out.append(f"KV mass score: {cap['corpus']} window {cap['window']}, ctx {cap['ctx']}, {cap['tokens']} tokens, "
               f"L {summary['L'][0]}..{summary['L'][1]}, sinks {summary['sinks']}, recent {summary['recent']}")
    out.append(f"  sealed {cap['sealed_sha256']}; attention {cap['attention']}")
    if check:
        out.append("  self-check (last token, check layer), max |recomputed - engine| / max |engine|: "
                   f"{check['0..=p']:.3e} over positions 0..=p, {check['0..p']:.3e} over 0..p, "
                   f"{check['prev_q']:.3e} with the previous token's query")
    out.append("")
    out.append("fraction of KV bytes read, median [p10, p90] over tokens, pooled over layers and KV heads by bytes")
    out.append(f"  {'selector':<10} {'P':>3}  " + "  ".join(f"{'delta ' + format(d, 'g'):<26}" for d in DELTAS))
    for sel in SELECTORS:
        for P in PAGES:
            cells = []
            for dl in DELTAS:
                r = summary["fraction_read"][f"{sel}|{P}|{dl:g}"]
                cells.append(f"{r['median']:.4f} [{r['p10']:.4f}, {r['p90']:.4f}]".ljust(26))
            out.append(f"  {sel:<10} {P:>3}  " + "  ".join(cells))
    out.append("")
    out.append("SCORE (delta 1e-3, best P, median):")
    for sel in SELECTORS:
        b = summary["score_delta_1e-3"][sel]
        out.append(f"  SCORE {sel:<10} P={b['P']:<3} {b['median']:.4f}")
    out.append("")
    for kk, rows in summary["per_layer"].items():
        sel, P, dl = kk.split("|")
        if dl != "0.001":
            continue
        out.append(f"box, P = {P}, delta 1e-3, by layer: median [p10, p90]")
        for i, r in enumerate(rows):
            out.append(f"  layer {i:>2}  {r['median']:.4f} [{r['p10']:.4f}, {r['p90']:.4f}]")
    out.append("")
    out.append("order violations in pages read (rows = tokens x layers x KV heads):")
    for kk, v in summary["order_violations"].items():
        out.append(f"  P|delta {kk:<9} " + ", ".join(f"{a}: {b}" for a, b in v.items()))
    eb = summary["output_error_bound_box32_1e-3"]
    out.append(f"output-error bound, box P=32 delta 1e-3 ({eb['what']}): median {eb['median']:.3e}, "
               f"p90 {eb['p90']:.3e}, max {eb['max']:.3e}")
    return "\n".join(out)


def rule(paths):
    """The prereg's reading over three texts."""
    sums = [json.loads(Path(p).read_text()) for p in paths]
    assert len(sums) == 3, "three texts"
    pm = [s["score_delta_1e-3"]["page-max"]["median"] for s in sums]
    bx = [s["score_delta_1e-3"]["box"]["median"] for s in sums]
    names = [s["capture"]["corpus"] for s in sums]
    for n, a, b in zip(names, pm, bx):
        print(f"  {n:<16} page-max {a:.4f}  box {b:.4f}")
    if sum(x > 0.50 for x in pm) >= 2:
        print("READING: KILL (page-max at its best P reads more than 0.50 on two texts of three)")
    elif sum(x < 0.35 for x in bx) >= 2:
        print("READING: ALIVE (box at its best P reads less than 0.35 on two texts of three)")
    else:
        print("READING: BETWEEN (the summary is the problem)")


# ---------------------------------------------------------------------------------------------------------------
# selftest: synthetic captures in the capture's own file layout


def write_synthetic(d: Path, q, k, v, ctx, corpus):
    T, nl, nh, hd = q.shape
    nkv, L = k.shape[1], k.shape[2]
    d.mkdir(parents=True)
    q.astype("<f2").tofile(d / "q.bin")
    for lay in range(nl):
        k[lay].astype("<f2").tofile(d / f"k-{lay}.bin")
        np.sqrt((v[lay].astype(np.float64) ** 2).sum(axis=2)).astype("<f4").tofile(d / f"vnorm-{lay}.bin")
    lay = nl // 2
    v[lay].astype("<f2").tofile(d / "v-check.bin")
    # the check output, recomputed in f64 and rounded to f16, standing in for the engine's
    groups = nh // nkv
    c = 1 / np.sqrt(hd)
    n = ctx + T
    out = np.zeros((nh, hd))
    for h in range(nh):
        g = h // groups
        s = c * (k[lay, g, :n].astype(np.float16).astype(np.float64) @ q[T - 1, lay, h].astype(np.float16).astype(np.float64))
        w = np.exp(s - s.max())
        out[h] = (w / w.sum()) @ v[lay, g, :n].astype(np.float16).astype(np.float64)
    out.astype("<f2").tofile(d / "out-check.bin")
    meta = {"n_layers": nl, "n_heads": nh, "n_kv": nkv, "head_dim": hd, "groups": groups, "ctx": ctx, "tokens": T,
            "positions": L, "corpus": corpus, "window": 0, "sealed_sha256": "synthetic", "attention": "synthetic",
            "c": float(c), "check": {"token": T - 1, "layer": lay}}
    (d / "meta.json").write_text(json.dumps(meta))


def synth(kind, rng, nl=2, nh=8, nkv=2, hd=128, ctx=1500, T=8):
    L = ctx + T
    groups = nh // nkv
    v = rng.normal(0, 1, (nl, nkv, L, hd))
    if kind == "onehot":
        # every query head of a group points at one sink key, 20 long; the others are small noise
        u = rng.normal(0, 1, (nl, nkv, hd))
        u /= np.linalg.norm(u, axis=2, keepdims=True)
        k = rng.normal(0, 0.1, (nl, nkv, L, hd))
        k[:, :, 10, :] = 20 * u
        q = np.repeat(20 * u[None, :, :, None, :], T, axis=0).repeat(groups, axis=3).reshape(T, nl, nh, hd)
        q = q + rng.normal(0, 0.01, q.shape)
    elif kind == "uniform":
        # one key repeated at every position: every score equal, a page holds P / L of the mass
        k = np.repeat(rng.normal(0, 1, (nl, nkv, 1, hd)), L, axis=2)
        q = rng.normal(0, 1, (T, nl, nh, hd))
    elif kind == "mixed":
        # the query heads of a group share a direction u, 12 long, plus noise of their own; background keys are
        # N(0, 1); six runs of 5 keys at random places in the middle carry 12 u. Attention is concentrated on the
        # runs (score about 12.7 against a background sd of about 1.1), the page maxima of the background stay
        # near 2, and the box and ball bounds of a background page sit above the runs' score: the three bounds
        # read different numbers of pages.
        u = rng.normal(0, 1, (nl, nkv, hd))
        u /= np.linalg.norm(u, axis=2, keepdims=True)
        k = rng.normal(0, 1, (nl, nkv, L, hd))
        q = np.repeat(12 * u[None, :, :, None, :], T, axis=0).repeat(groups, axis=3).reshape(T, nl, nh, hd)
        q = q + rng.normal(0, 0.3, q.shape)
        for lay in range(nl):
            for g in range(nkv):
                for a in rng.choice(np.arange(100, ctx - 300), 6, replace=False):
                    k[lay, g, a:a + 5] = 12 * u[lay, g]
    else:
        raise ValueError(kind)
    return q, k, v, ctx


def selftest():
    rng = np.random.default_rng(20261114)
    tmp = Path(tempfile.mkdtemp(prefix="kvmass-"))
    S, R = 64, 256
    lines = []
    ok = True
    try:
        res = {}
        for kind in ("onehot", "uniform", "mixed"):
            q, k, v, ctx = synth(kind, rng)
            d = tmp / kind
            write_synthetic(d, q.astype(np.float16), k.astype(np.float16), v.astype(np.float16), ctx, kind)
            meta, Ls, read, pages_read, total, errb = score_capture(d, S, R, quiet=True)
            summ = summarize(meta, Ls, read, pages_read, total, errb, S, R)
            res[kind] = (summ, pages_read, Ls, read, total)
            chk = self_check(d, meta, load_capture(d)[1])
            lines.append(f"{kind}: self-check rel err {chk['0..=p']:.2e} over 0..=p, {chk['0..p']:.2e} over 0..p, "
                         f"{chk['prev_q']:.2e} with the previous token's query")
        # one-hot: every page skippable at every delta and P, for every selector
        summ, pr, Ls, read, total = res["onehot"]
        for (sel, P, dl), x in pr.items():
            if x.max() != 0:
                ok = False
                lines.append(f"FAIL onehot: {sel} P={P} delta={dl} reads {x.max()} pages")
        dense = {P: page_geometry(Ls, S, R, P)[1] for P in PAGES}
        for P in PAGES:
            want = np.median(dense[P] / Ls)
            got = summ["fraction_read"][f"true-mass|{P}|0.001"]["median"]
            lines.append(f"onehot P={P}: true-mass reads {got:.4f}, the dense share {want:.4f}; box "
                         f"{summ['fraction_read'][f'box|{P}|0.001']['median']:.4f}, ball "
                         f"{summ['fraction_read'][f'ball|{P}|0.001']['median']:.4f}")
            ok &= abs(got - want) < 1e-12
        # uniform: nothing skippable at 1e-3 (a page holds P / L > 1e-3 of the mass)
        summ, pr, Ls, read, total = res["uniform"]
        for sel in SELECTORS:
            for P in PAGES:
                npg = page_geometry(Ls, S, R, P)[0]
                x = pr[(sel, P, 1e-3)]
                if not np.all(x == npg[:, None, None]):
                    ok = False
                    lines.append(f"FAIL uniform: {sel} P={P} delta=1e-3 skips a page")
        lines.append("uniform delta 1e-3: " + ", ".join(
            f"{sel} {summ['fraction_read'][f'{sel}|16|0.001']['median']:.4f}" for sel in SELECTORS) + " (P = 16)")
        # the orderings, in pages read, on every row of every case
        for kind, (summ, pr, *_rest) in res.items():
            for kk, v in summ["order_violations"].items():
                bad = {a: b for a, b in v.items() if a != "rows" and b}
                if bad:
                    ok = False
                    lines.append(f"FAIL order {kind} P|delta {kk}: {bad}")
            m = summ["fraction_read"]
            lines.append(f"{kind} delta 1e-3 P=16 fractions: " + ", ".join(
                f"{sel} {m[f'{sel}|16|0.001']['median']:.4f}" for sel in SELECTORS))
        lines.append("order true-mass <= page-max <= box, page-max <= ball in pages read: "
                     + ("holds on every row of the three cases" if ok else "SEE FAILURES"))
        # mutants: each must trip the bound assertion on some case
        for mutant in ("no_P", "box_min"):
            caught = []
            for kind in ("onehot", "uniform", "mixed"):
                try:
                    score_capture(tmp / kind, S, R, mutant=mutant, quiet=True)
                except BoundViolation as e:
                    caught.append(f"{kind} ({str(e)[:90]})")
            verdict = "KILLED" if caught else "SURVIVED"
            ok &= bool(caught)
            lines.append(f"mutant {mutant}: {verdict} on " + ("; ".join(caught) if caught else "no case"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(lines))
    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("capture", nargs="?", type=Path)
    ap.add_argument("--out", type=Path, help="output prefix; default <capture>/kv-mass-score")
    ap.add_argument("--sinks", type=int, default=64)
    ap.add_argument("--recent", type=int, default=256)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--rule", nargs=3, metavar="JSON")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.rule:
        rule(a.rule)
        return 0
    if a.capture is None:
        ap.error("a capture directory, --selftest or --rule")
    meta, q = load_capture(a.capture)
    check = self_check(a.capture, meta, q)
    if check["0..=p"] > 1e-2:
        raise SystemExit(f"self-check failed: the dump does not give the engine's output ({check})")
    meta, Ls, read, pages_read, total, errb = score_capture(a.capture, a.sinks, a.recent)
    summ = summarize(meta, Ls, read, pages_read, total, errb, a.sinks, a.recent)
    summ["self_check"] = check
    txt = table(summ, check)
    prefix = a.out or (a.capture / "kv-mass-score")
    Path(str(prefix) + ".json").write_text(json.dumps(summ, indent=1))
    Path(str(prefix) + ".txt").write_text(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
