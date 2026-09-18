# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Stage 1 of the E8 arbitration: E8 cubed against Tetra on the same blocks.

Protocol: proofs/preregistration-e8-etape1-2026-09-18.md, stamped
7b75b465cf8be967be747f45983d480ccce0d470be74b69109d8d14be9114c54.

The codebook is read from the Rust module that stage 0 verified, and re-checked
here: a codebook that crossed a file boundary is a codebook that can have been
truncated.

Both arms share the gain. `picked` depends only on the block norm and the cell's
centroids, so the level is the same for both, and the whole difference between
the arms is the direction. That makes the comparison purely angular:

    |x - u*picked|^2 = |x|^2 - 2*picked*<x,u> + picked^2

Arm B maximizes <x,u> exactly over the product codebook. |u| couples the three
sub-blocks only through their shells, so the maximum is taken by scanning each
sub-block once per shell and then comparing the 125 shell triples.
"""

import json
import math
import os
import sys

import numpy as np

ROOT = os.path.expanduser("~/tetra-diag-4b-2026-09-18")
BOOK = os.environ.get("E8_BOOK", f"{ROOT}/e8-codebook-norm10.txt")
DIM, SUB = 24, 8


def load_codebook():
    y = np.loadtxt(BOOK, dtype=np.int64)
    assert y.ndim == 2 and y.shape[1] == SUB, f"codebook shape {y.shape}"
    q = (y * y).sum(1)
    parity = np.abs(y) % 2
    assert (parity.min(1) == parity.max(1)).all(), "a codebook point has mixed parity"
    assert (y.sum(1) % 4 == 0).all(), "a codebook point fails the sum condition"
    assert (q == 8).sum() == 240, "kissing number"
    shells = sorted(set(q.tolist()))
    for s in shells:
        n = s // 8
        s3 = sum(d**3 for d in range(1, n + 1) if n % d == 0)
        assert (q == s).sum() == 240 * s3, f"theta at norm {2*n}"
    idx = math.ceil(math.log2(len(y)))
    print(f"codebook re-checked: {len(y)} points, norms {[s//4 for s in shells]}, "
          f"kissing 240, theta exact")
    print(f"rate: {idx} bits an index, {3*idx} + 1 gain = {(3*idx+1)/24:.4f} b/weight "
          f"against Tetra's measured 1.9907")
    return y.astype(np.float64), q


def best_per_shell(book, q, shells, xs):
    """For each sub-block column of `xs`, the best dot per shell and its index."""
    dots = book @ xs                      # (56880, k)
    out = np.empty((len(shells), xs.shape[1]))
    arg = np.empty((len(shells), xs.shape[1]), dtype=np.int64)
    for i, s in enumerate(shells):
        m = np.where(q == s)[0]
        sub = dots[m]
        j = sub.argmax(0)
        out[i] = sub[j, np.arange(xs.shape[1])]
        arg[i] = m[j]
    return out, arg


def main():
    book, q = load_codebook()
    shells = np.array(sorted(set(q.tolist())))
    snorm = shells.astype(np.float64)     # |y|^2 of a point of that shell
    triples = [(a, b, c) for a in range(len(shells)) for b in range(len(shells))
               for c in range(len(shells))]
    tri_norm = np.array([snorm[a] + snorm[b] + snorm[c] for a, b, c in triples])
    tri_idx = np.array(triples)

    # ---- control 1 and 2, before any cell is read ----
    zero = np.zeros(DIM)
    bp, _ = best_per_shell(book, q, shells, zero.reshape(3, SUB).T)
    assert np.allclose(bp, 0.0), "control 1: a zero block must give a zero dot"
    pt = np.concatenate([book[0], book[1], book[2]])
    u = pt / math.sqrt((pt * pt).sum())
    picked = 0.05
    x = u * picked
    bp, _ = best_per_shell(book, q, shells, x.reshape(3, SUB).T)
    score = (bp[tri_idx[:, 0], 0] + bp[tri_idx[:, 1], 1] + bp[tri_idx[:, 2], 2])
    best = (score / np.sqrt(tri_norm)).max()
    err = picked**2 - 2 * picked * best + picked**2
    # Relative to the block energy: the identity is exact in real arithmetic and
    # the residue is machine epsilon on quantities of order picked^2.
    assert abs(err) < 1e-12 * picked**2, (
        f"control 2: a codebook block must reconstruct exactly, got {err:e}"
    )
    print("controls 1 and 2 pass: zero block zero, codebook block exact")

    rows_out = []
    c3_worst = 0.0
    c3b_worst = 0.0
    c5_worst = 0.0
    for arm in ("replay-capture-a-v64", "replay-capture-b-v64"):
        for cell in sorted(os.listdir(f"{ROOT}/{arm}")):
            d = f"{ROOT}/{arm}/{cell}"
            if not os.path.isdir(d):
                continue
            b = json.load(open(f"{d}/bundle.json"))
            n, cent = b["width"], np.array(b["centroids"])
            h = np.memmap(f"{ROOT}/{arm.replace('replay-', '')}/{b['hessian']['name']}",
                          dtype=np.float64, mode="r").reshape(n, n)
            nb = n // DIM
            for rf in sorted(f for f in os.listdir(d) if f.startswith("row-")):
                g = json.load(open(f"{d}/{rf}"))["diagnostic"]
                x = np.array(g["original"][: nb * DIM])
                w = np.array(g["witness"][: nb * DIM])
                rs = g["row_scale"]
                xb = x.reshape(nb, DIM)
                wb = w.reshape(nb, DIM)
                nx = np.linalg.norm(xb, axis=1)
                # The gain is taken from the object itself: |witness| IS the
                # amplitude the encoder picked. Deviation E1 explains why the
                # level is read here rather than recomputed.
                picked = np.linalg.norm(wb, axis=1)
                lvl = np.abs((cent * rs)[None, :] - picked[:, None]).argmin(1)
                c3_worst = max(c3_worst, float(np.abs(picked - (cent * rs)[lvl]).max()))
                ea = ((xb - wb) ** 2).sum(1)
                # Block 0 carries no compensation, so there the replay's
                # `euclidean` at the chosen level IS the served error.
                c3b_worst = max(
                    c3b_worst, abs(float(ea[0]) - g["shadow"][0]["euclidean"][lvl[0]])
                )
                # arm B, exact joint angular maximum
                cols = xb.reshape(nb * 3, SUB).T
                bp, ba = best_per_shell(book, q, shells, cols)
                bp = bp.reshape(len(shells), nb, 3)
                sc = (bp[tri_idx[:, 0], :, 0] + bp[tri_idx[:, 1], :, 1]
                      + bp[tri_idx[:, 2], :, 2]) / np.sqrt(tri_norm)[:, None]
                dotb = sc.max(0)
                tb = sc.argmax(0)
                eb = nx**2 - 2 * picked * dotb + picked**2
                # the row reconstruction of arm B, for J_local
                ba = ba.reshape(len(shells), nb, 3)
                recb = np.empty((nb, DIM))
                for k in range(nb):
                    a, bb, c = tri_idx[tb[k]]
                    p = np.concatenate([book[ba[a, k, 0]], book[ba[bb, k, 1]], book[ba[c, k, 2]]])
                    recb[k] = p / math.sqrt((p * p).sum()) * picked[k]
                dota = (xb * wb).sum(1)
                # control 5: the analytic arm B error must equal the error of
                # the row this loop assembled. If the blocks were misordered or
                # the shell triple misread, J_local below would be nonsense
                # while the Euclidean number stayed plausible.
                c5_worst = max(
                    c5_worst,
                    float(np.abs(eb - ((xb - recb) ** 2).sum(1)).max() / max(eb.max(), 1e-300)),
                )
                fam = cell.split("-", 3)[3]
                # J_local at row level, as the prereg defines it
                ra = (x - w)
                rb = (x - recb.reshape(-1))
                hv = np.asarray(h[: nb * DIM, : nb * DIM])
                den = float(x @ (hv @ x))
                ja = float(ra @ (hv @ ra)) / den
                jb = float(rb @ (hv @ rb)) / den
                # Control 6, the isotropic reference. For a residual of the
                # same energy pointing in a random direction,
                # E[e'He] = |e|^2 tr(H)/n, exactly. An arm whose J sits far
                # from its own isotropic value has a residual ALIGNED with the
                # curvature, and that is a claim about the code, not about the
                # amount of error it makes.
                trn = float(np.trace(hv)) / hv.shape[0]
                ja_iso = float(ra @ ra) * trn / den
                jb_iso = float(rb @ rb) * trn / den
                rows_out.append(dict(
                    family=fam, cell=cell, row=rf,
                    blocks=int(nb), ea=float(ea.sum()), eb=float(eb.sum()),
                    x2=float((nx**2).sum()),
                    cosa=float((dota / (nx * picked)).mean()),
                    cosb=float((dotb / nx).mean()),
                    ja=ja, jb=jb, ja_iso=ja_iso, jb_iso=jb_iso))
            del h
    print(f"control 3a: |witness| against centroid*row_scale, worst gap {c3_worst:.3e}")
    assert c3_worst < 1e-12, "control 3a failed: the witness is not a coded amplitude"
    print(f"control 3b: block 0 against the replay's euclidean, worst gap {c3b_worst:.3e}")
    assert c3b_worst < 1e-12, "control 3b failed"
    print(f"control 5: analytic arm B error against the assembled row, worst relative {c5_worst:.3e}")
    assert c5_worst < 1e-10, "control 5 failed: the arm B row assembly disagrees"
    print("control 4: one pass, same blocks, same order, paired by block\n")

    fams = sorted({r["family"] for r in rows_out})
    print(f"{'family':22s} {'rows':>5s} {'blocks':>7s} {'MSE A':>12s} {'MSE B':>12s} "
          f"{'ratio':>7s} {'cos A':>8s} {'cos B':>8s} {'J_A':>10s} {'J_B':>10s} {'J ratio':>8s}"
          f"  {'A/iso':>7s} {'B/iso':>7s}")
    for f in fams + ["POOLED"]:
        rs_ = rows_out if f == "POOLED" else [r for r in rows_out if r["family"] == f]
        blocks = sum(r["blocks"] for r in rs_)
        ea, eb = sum(r["ea"] for r in rs_), sum(r["eb"] for r in rs_)
        x2 = sum(r["x2"] for r in rs_)
        ja = float(np.mean([r["ja"] for r in rs_]))
        jb = float(np.mean([r["jb"] for r in rs_]))
        jai = float(np.mean([r["ja_iso"] for r in rs_]))
        jbi = float(np.mean([r["jb_iso"] for r in rs_]))
        print(f"{f:22s} {len(rs_):5d} {blocks:7d} {ea/x2:12.6e} {eb/x2:12.6e} {eb/ea:7.4f} "
              f"{np.mean([r['cosa'] for r in rs_]):8.5f} {np.mean([r['cosb'] for r in rs_]):8.5f} "
              f"{ja:10.4e} {jb:10.4e} {jb/ja:8.4f}  {ja/jai:7.3f} {jb/jbi:7.3f}")

    pooled = sum(r["eb"] for r in rows_out) / sum(r["ea"] for r in rows_out)
    print(f"\nPRIMARY: MSE ratio B/A pooled = {pooled:.4f}")
    print(f"prereg point 1.09, interval [1.00, 1.20] -> "
          f"{'INSIDE' if 1.00 <= pooled <= 1.20 else 'OUTSIDE'}")
    ca = float(np.mean([r["cosa"] for r in rows_out]))
    cb = float(np.mean([r["cosb"] for r in rows_out]))
    print(f"cos A - cos B = {ca - cb:+.5f}  (prereg +0.005, interval [+0.002, +0.010])")
    print(f"arm B rate stated above; Tetra measured 1.9907 b/weight of stream")
    json.dump(rows_out, open(sys.argv[1], "w") if len(sys.argv) > 1 else sys.stdout, indent=1)


if __name__ == "__main__":
    main()
