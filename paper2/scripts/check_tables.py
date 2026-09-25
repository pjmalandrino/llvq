#!/usr/bin/env python3
"""Refuse a build whose hand-typed tables have drifted from their CSVs.

Paper 1 carries the same guard for the same reason: a table is typed once and
lives for months, and the CSV under it gets re-measured. The check is a
containment test: every value a CSV holds for a cell must appear, verbatim, in
the section that owns the table. The CSVs are written in the paper's own
formatting, so the comparison is a string comparison and not a rounding
convention. It catches a number that moved in the data and not in the paper;
it does not catch a number that moved in both.

A cell whose measurement has not come back is typed `\\pend` in the paper and
left empty in the CSV. The check accepts that pair and says so; with
RELEASE=1 in the environment it refuses the build while any `\\pend` is left.

Run by `make check`, which `make` does not skip.
"""

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "data"
SEC = ROOT / "paper2" / "sections"

FAILURES: list[str] = []
NOTES: list[str] = []


def read_csv(name: str) -> list[dict]:
    with open(DATA / name, newline="") as f:
        return list(csv.DictReader(f))


def need(section: str, needles: list[str], why: str) -> None:
    text = (SEC / section).read_text()
    for n in needles:
        if n and n not in text:
            FAILURES.append(f"{section}: {n!r} not found ({why})")


def pvalue(p: str) -> str:
    """1.8e-25 -> 1.8\\cdot10^{-25}, the paper's spelling."""
    mant, exp = p.split("e")
    return f"{mant}\\cdot10^{{{int(exp)}}}"


def check_main() -> None:
    """tab:main against paper2-results.csv."""
    rows = read_csv("paper2-results.csv")
    body = [r[c] for r in rows for c in ("bparam", "mmlu", "toks", "gb")]
    need("experiments.tex", body, "cell of tab:main")
    pending = [f"{r['model']} {r['arm']}" for r in rows if not r["toks"] or not r["gb"]]
    if pending:
        NOTES.append(f"tab:main: {len(pending)} rows pending ({', '.join(pending)})")


def check_chain() -> None:
    """tab:chain and tab:files against paper2-chain.csv."""
    body = []
    for r in read_csv("paper2-chain.csv"):
        body += [r["bparam"], r["mmlu"]]
        if r["delta_pp"]:
            body += [f"$+{r['delta_pp']}$",
                     f"$[+{r['ci_lo_pp']}, +{r['ci_hi_pp']}]$",
                     pvalue(r["mcnemar_p"])]
    need("experiments.tex", body, "cell of tab:chain")
    sealed = [r["bparam"] for r in read_csv("paper2-chain.csv")
              if r["stage"] == "sealed"]
    need("model.tex", sealed, "bits per parameter in tab:files")


def check_gaps() -> None:
    """The paired gaps quoted in the text of the main results."""
    body = []
    for r in read_csv("paper2-gaps.csv"):
        if r["pair"] == "f16_minus_awq4":
            body.append(r["delta_pp"])
        else:
            body.append(f"{r['delta_pp']}")
            body.append(f"[{r['ci_lo_pp']}, {r['ci_hi_pp']}]")
    need("experiments.tex", body, "paired gap in the main results")


def check_bench() -> None:
    """tab:bench and the appendix table against echelle-formats.csv."""
    rows = {r["layout"]: r for r in read_csv("echelle-formats.csv")}
    cols = ("med_ms", "gb_read", "bpw_kernel", "gbps")
    body = [rows[k][c] for k in ("FP16", "AWQ", "Planes14", "Tetra48", "nullk")
            for c in cols]
    need("kernel.tex", body, "cell of tab:bench")
    app = [r[c] for r in rows.values() for c in ("med_ms", "bpw_kernel")]
    need("appendix.tex", app, "row of the ten-arm table")


def check_tile() -> None:
    """The tile table next to fig:tile, and the prose, against tuile-l40s.csv."""
    rows = read_csv("tuile-l40s.csv")
    body = [r[c] for r in rows
            for c in ("tile", "nullk_ms", "planes14_ms", "tetra48_ms")]
    for col in ("nullk_ms", "planes14_ms", "tetra48_ms"):
        ys = [float(r[col]) for r in rows]
        body.append(f"{100 * (max(ys) / min(ys) - 1):.1f}")
    need("kernel.tex", body, "cell or range of the tile table")


def check_shape() -> None:
    """The CSVs must be rectangular: an unescaped comma in a free-text field
    throws the rest of the row into the None key, and nothing fails on its
    own. Paper 1 enforces the same rule."""
    for name in ("tuile-l40s.csv", "paper2-results.csv", "paper2-gaps.csv",
                 "paper2-chain.csv"):
        with open(DATA / name, newline="") as f:
            reader = csv.DictReader(f)
            width = len(reader.fieldnames or [])
            for i, row in enumerate(reader, start=2):
                if None in row:
                    FAILURES.append(f"{name}:{i}: more fields than the header "
                                    f"({width}); an unescaped comma?")
                if any(v is None for v in row.values()):
                    FAILURES.append(f"{name}:{i}: fewer fields than the header")


def check_pending() -> None:
    """Count the `\\pend` cells; a release build refuses any."""
    n = sum((SEC / f).read_text().count("\\pend") for f in os.listdir(SEC)
            if f.endswith(".tex"))
    if n:
        msg = f"{n} \\pend cells left in the sections"
        if os.environ.get("RELEASE") == "1":
            FAILURES.append(msg + " (RELEASE=1)")
        else:
            NOTES.append(msg)


def main() -> None:
    check_shape()
    check_main()
    check_chain()
    check_gaps()
    check_bench()
    check_tile()
    check_pending()
    for n in NOTES:
        print("check_tables: note: " + n)
    if FAILURES:
        print("check_tables: the paper and its data disagree\n")
        for f in FAILURES:
            print("  " + f)
        sys.exit(1)
    print("check_tables: every table agrees with its CSV")


if __name__ == "__main__":
    main()
