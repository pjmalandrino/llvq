#!/usr/bin/env python3
"""Refuse a build whose hand-typed tables have drifted from their CSVs.

Paper 1 carries the same guard (`paper/scripts/check_tables.py`) for the same
reason: a table is typed once and lives for months, and the CSV under it gets
re-measured. The tables of paper 2 are small enough that the check is a
containment test — every value a CSV holds for a cell must appear, verbatim,
in the section that owns the table. The CSVs are written in the paper's own
formatting so that the comparison is a string comparison and not a rounding
convention. It catches a number that moved in the data and not in the paper;
it does not catch a number that moved in both.

Run by `make check`, which `make` does not skip.
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "data"
SEC = ROOT / "paper2" / "sections"

FAILURES: list[str] = []


def read_csv(name: str) -> list[dict]:
    with open(DATA / name, newline="") as f:
        return list(csv.DictReader(f))


def need(section: str, needles: list[str], why: str) -> None:
    text = (SEC / section).read_text()
    for n in needles:
        if n and n not in text:
            FAILURES.append(f"{section}: {n!r} not found — {why}")


def check_bench() -> None:
    """Table 1 (tab:bench) and the appendix table against echelle-formats.csv."""
    rows = {r["layout"]: r for r in read_csv("echelle-formats.csv")}
    cols = ("med_ms", "gb_read", "bpw_kernel", "gbps")
    body = [rows[k][c] for k in ("FP16", "AWQ", "Planes14", "Tetra48", "nullk")
            for c in cols]
    need("results.tex", body, "cell of Table 1 (tab:bench)")
    app = [r[c] for r in rows.values() for c in ("med_ms", "bpw_kernel")]
    need("appendix.tex", app, "row of the ten-arm table")


def check_tile() -> None:
    """Table 2 (tab:tile), its figure and the prose against tuile-l40s.csv."""
    rows = read_csv("tuile-l40s.csv")
    body = [r[c] for r in rows
            for c in ("tile", "nullk_ms", "planes14_ms", "tetra48_ms")]
    # The three amplitudes the prose and the caption both quote; the figure
    # computes them from the same column.
    for col in ("nullk_ms", "planes14_ms", "tetra48_ms"):
        ys = [float(r[col]) for r in rows]
        body.append(f"{100 * (max(ys) / min(ys) - 1):.1f}")
    need("kernel.tex", body, "cell or amplitude of Table 2 (tab:tile)")


def check_formats() -> None:
    """Table 4 (tab:formats) against tetra-formats.csv."""
    body = [r[c] for r in read_csv("tetra-formats.csv")
            for c in ("disk_gb", "bpp_whole_model", "bpw_kernel", "ppl", "mmlu")]
    need("results.tex", body, "cell of Table 4 (tab:formats)")


def check_rowscales() -> None:
    """Table 5 (tab:rowscales) and the prose against tetra-rowscales.csv."""
    body = [r[c] for r in read_csv("tetra-rowscales.csv")
            for c in ("mmlu_before", "mmlu_after", "delta_pp",
                      "ppl_before", "ppl_after")]
    need("results.tex", body, "cell of Table 5 (tab:rowscales)")


def check_shape() -> None:
    """The CSVs must be rectangular: an unescaped comma in a free-text field
    throws the rest of the row into the None key, and nothing fails on its
    own. Paper 1 enforces the same rule."""
    for name in ("tuile-l40s.csv", "tetra-rowscales.csv", "tetra-formats.csv"):
        with open(DATA / name, newline="") as f:
            reader = csv.DictReader(f)
            width = len(reader.fieldnames or [])
            for i, row in enumerate(reader, start=2):
                if None in row:
                    FAILURES.append(f"{name}:{i}: more fields than the header "
                                    f"({width}); an unescaped comma?")
                if any(v is None for v in row.values()):
                    FAILURES.append(f"{name}:{i}: fewer fields than the header")


def main() -> None:
    check_shape()
    check_bench()
    check_tile()
    check_formats()
    check_rowscales()
    if FAILURES:
        print("check_tables: the paper and its data disagree\n")
        for f in FAILURES:
            print("  " + f)
        sys.exit(1)
    print("check_tables: every table agrees with its CSV")


if __name__ == "__main__":
    main()
