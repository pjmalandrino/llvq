# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Driver for E8 stage 1 bis: both arms through the same GPTQ loop.

Protocol: proofs/preregistration-e8-etape1bis-2026-09-18.md, stamped
8faa76613b95dd446f015d96636e5fbfaca81a788d3a0e09805c8b4af6551ec6.

The arithmetic is entirely in `llvq-bench --bin e8gptq`, which runs
`quantize_layer` twice per row. This script only reads each cell's
`bundle.json` for the width, the centroids and the damping, calls the binary,
and aggregates. No number is computed here that the binary did not print.
"""

import json
import os
import subprocess
import sys
import time

ROOT = os.path.expanduser("~/tetra-diag-4b-2026-09-18")
BIN = "./target/release/e8gptq"
# The cap is a knob because the bit budget is the confound: at 8 arm B runs
# on 3.7 % FEWER bits than Tetra, at 10 on 2.6 % more. A verdict needs both.
CAP = int(os.environ.get("E8_CAP", "8"))


def main():
    cells = []
    for arm in ("capture-a-v64", "capture-b-v64"):
        for cell in sorted(os.listdir(f"{ROOT}/replay-{arm}")):
            if os.path.isdir(f"{ROOT}/replay-{arm}/{cell}"):
                cells.append((arm, cell))
    print(f"{len(cells)} cells, cap {CAP}\n")

    rows = []
    for arm, cell in cells:
        b = json.load(open(f"{ROOT}/replay-{arm}/{cell}/bundle.json"))
        n, cent = b["width"], b["centroids"]
        damping = b["plan"]["damping"]
        nrows = len(b["row_ids"])
        h = f"{ROOT}/{arm}/{b['hessian']['name']}"
        rw = f"{ROOT}/{arm}/{b['original_rows']['name']}"
        t0 = time.time()
        out = subprocess.run(
            [BIN, h, rw, str(n), str(nrows), str(CAP), str(damping),
             repr(cent[0]), repr(cent[1])],
            capture_output=True, text=True, check=True).stdout
        el = time.time() - t0
        fam = cell.split("-", 3)[3]
        for line in out.splitlines():
            if line.startswith("#"):
                continue
            r, a, err, jl, ji, cs = line.split("\t")
            rows.append(dict(family=fam, cell=cell, row=int(r), arm=a,
                             err=float(err), jloc=float(jl), jiso=float(ji),
                             cos=float(cs)))
        print(f"  {cell:34s} n {n:5d} rows {nrows} : {el:6.1f} s")

    def agg(sel):
        a = [r for r in sel if r["arm"] == "tetra"]
        b = [r for r in sel if r["arm"] == "e8cubed"]
        assert len(a) == len(b) and a, "arms must pair"
        sa, sb = sum(r["jloc"] for r in a), sum(r["jloc"] for r in b)
        ea, eb = sum(r["err"] for r in a), sum(r["err"] for r in b)
        ia = sum(r["jloc"] for r in a) / sum(r["jiso"] for r in a)
        ib = sum(r["jloc"] for r in b) / sum(r["jiso"] for r in b)
        ca = sum(r["cos"] for r in a) / len(a)
        cb = sum(r["cos"] for r in b) / len(b)
        return len(a), sa, sb, sb / sa, ea, eb, eb / ea, ia, ib, ca, cb

    fams = sorted({r["family"] for r in rows})
    print(f"\n{'family':16s} {'rows':>5s} {'J_A':>11s} {'J_B':>11s} {'J B/A':>7s} "
          f"{'err A':>11s} {'err B':>11s} {'e B/A':>7s} {'A/iso':>7s} {'B/iso':>7s} "
          f"{'cos A':>8s} {'cos B':>8s}")
    for f in fams + ["POOLED"]:
        sel = rows if f == "POOLED" else [r for r in rows if r["family"] == f]
        k, sa, sb, jr, ea, eb, er, ia, ib, ca, cb = agg(sel)
        print(f"{f:16s} {k:5d} {sa:11.4e} {sb:11.4e} {jr:7.4f} {ea:11.4e} {eb:11.4e} "
              f"{er:7.4f} {ia:7.4f} {ib:7.4f} {ca:8.5f} {cb:8.5f}")

    k, sa, sb, jr, ea, eb, er, ia, ib, ca, cb = agg(rows)
    print(f"\nPRIMARY  J_B/J_A = {jr:.4f}   prereg 1.09, interval [0.95, 1.30] -> "
          f"{'INSIDE' if 0.95 <= jr <= 1.30 else 'OUTSIDE'}")
    print(f"second   unweighted B/A = {er:.4f}   interval [0.95, 1.30] -> "
          f"{'INSIDE' if 0.95 <= er <= 1.30 else 'OUTSIDE'}")
    print(f"gate     |B/iso - A/iso| = {abs(ib - ia):.4f}   must be below 0.05 -> "
          f"{'PASSES' if abs(ib - ia) < 0.05 else 'FAILS, the run is void'}")
    print(f"fourth   cos A - cos B = {ca - cb:+.5f}   interval [0.000, +0.020] -> "
          f"{'INSIDE' if 0.0 <= ca - cb <= 0.020 else 'OUTSIDE'}")
    if len(sys.argv) > 1:
        json.dump(rows, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
