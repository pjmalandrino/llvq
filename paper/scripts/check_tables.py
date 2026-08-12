#!/usr/bin/env python3
"""Confront the paper's hand-typed tables with the CSVs they claim to come from.

The paper promises, in four places, that every number regenerates from
committed CSV files. That was true of the *figures* — `make_figures.py` reads
the CSVs — and false of the *tables*, which are LaTeX typed by hand from the
same sources. A promise checked only by the author's care is the failure mode
this repository documents everywhere else, so it is checked here instead.

Exit code 1 on any mismatch, with the cell named. Wired into `make`, so a
table that drifts from its CSV fails the build rather than the review.

Scope, stated so nobody mistakes it for more: this checks Table 1 of
`sections/layouts.tex` against `docs/data/echelle-formats.csv`, cell by cell,
including the derived `of bound` column. The other tables have no CSV of the
same shape yet; adding them is mechanical and they are listed at the bottom.
"""

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "data"
SEC = ROOT / "paper" / "sections"

# The bench computes GB/s from each arm's fastest kept round; the paper says
# so in its methodology paragraph. Reproduced here rather than recomputed from
# the median, because the check must confront what the CSV records.
FP16_REF_GBPS = 661


def parse_layout_table(tex: str) -> dict[str, list[str]]:
    """The rows of the first tabular in layouts.tex, keyed by kernel name."""
    body = tex.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
    rows = {}
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("%") or line.startswith(r"\midrule"):
            continue
        cells = [c.strip() for c in line.rstrip("\\").split("&")]
        if len(cells) < 7:
            continue
        name = re.sub(r"\\textsc\{([^}]*)\}", r"\1", cells[0])
        name = name.replace(r"\,", "").strip()
        rows[name] = cells[1:]
    return rows


# CSV layout name -> the name as the table spells it.
TABLE_NAME = {
    "FP16": "FP16 (control)",
    "Slot32": "Slot32",
    "Planes14": "Planes14",
    "Planes12x": "Planes12x",
    "Golay70v1": "Golay70",
    "Golay70v2": "Golay70, hoisted",
    "AWQ": "AWQ w4g128",
}


def num(cell: str) -> str:
    """The first number in a LaTeX cell, bold and ranges stripped."""
    cell = re.sub(r"\\textbf\{([^}]*)\}", r"\1", cell)
    m = re.search(r"-?\d+\.?\d*", cell)
    return m.group(0) if m else ""


def check_layouts() -> list[str]:
    rows = {r["layout"]: r for r in csv.DictReader(open(DATA / "echelle-formats.csv"))}
    tex = (SEC / "layouts.tex").read_text()
    table = parse_layout_table(tex)
    bad = []

    for layout, r in rows.items():
        name = TABLE_NAME.get(layout)
        if name is None:
            bad.append(f"CSV row '{layout}' has no counterpart in TABLE_NAME")
            continue
        if name not in table:
            bad.append(f"CSV row '{layout}' is missing from Table 1 (as '{name}')")
            continue
        cells = table[name]
        want = [
            ("b/weight", f"{float(r['bpw_kernel']):.3f}", num(cells[0])),
            ("GB", f"{float(r['gb_read']):.2f}", num(cells[1])),
            ("med ms", f"{float(r['med_ms']):.3f}", num(cells[2])),
            ("GB/s", str(int(r["gbps"])), num(cells[3])),
        ]
        # `of bound` is derived, and the control's cell is a dash by design.
        if layout != "FP16":
            want.append(("of bound", r["pct_byte_bound"], num(cells[4])))
            want.append(("vs FP16", f"{float(r['ratio_vs_fp16']):.2f}", num(cells[5])))
            lo, hi = float(r["ratio_lo"]), float(r["ratio_hi"])
            got_range = re.findall(r"(\d+\.\d+)--(\d+\.\d+)", cells[5])
            if not got_range:
                bad.append(f"{name}: vs-FP16 cell carries no [lo--hi] range")
            elif (got_range[0][0], got_range[0][1]) != (f"{lo:.2f}", f"{hi:.2f}"):
                bad.append(
                    f"{name}: range {got_range[0]} in the table, "
                    f"({lo:.2f}, {hi:.2f}) in the CSV"
                )
        for field, expect, got in want:
            if expect != got:
                bad.append(f"{name} / {field}: CSV says {expect}, Table 1 says {got}")

        # The derived column must also follow from the CSV's own GB/s, or the
        # CSV is internally inconsistent and the table would inherit it.
        if layout != "FP16":
            derived = round(int(r["gbps"]) / FP16_REF_GBPS * 100)
            if str(derived) != r["pct_byte_bound"]:
                bad.append(
                    f"{layout} / pct_byte_bound: {r['pct_byte_bound']} recorded, "
                    f"{derived} implied by {r['gbps']} GB/s over {FP16_REF_GBPS}"
                )

    for name in table:
        if name not in TABLE_NAME.values():
            bad.append(f"Table 1 row '{name}' has no CSV row")
    return bad


def main() -> int:
    bad = check_layouts()
    if bad:
        print("table/CSV mismatch:", file=sys.stderr)
        for b in bad:
            print(f"  {b}", file=sys.stderr)
        return 1
    print("tables agree with docs/data/*.csv (layouts: 7 rows)")
    # Not yet covered, and named rather than left implicit: tab:campaign and
    # tab:campaign8b (campagne-finale.csv, tableau-8b.csv, different shape),
    # tab:lit (no CSV — transcribed from the original paper), tab:attribution
    # and tab:phases (phases.csv covers the figure, not the table).
    return 0


if __name__ == "__main__":
    sys.exit(main())
