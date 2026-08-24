#!/usr/bin/env python3
"""Confront the paper's hand-typed tables with the CSVs they claim to come from.

The paper promises, in four places, that every number regenerates from
committed CSV files. That was true of the *figures* — `make_figures.py` reads
the CSVs — and false of the *tables*, which are LaTeX typed by hand from the
same sources. A promise checked only by the author's care is the failure mode
this repository documents everywhere else, so it is checked here instead.

Exit code 1 on any mismatch, with the cell named. Wired into `make`, so a
table that drifts from its CSV fails the build rather than the review.

Scope, stated so nobody mistakes it for more. Seven tables are checked cell
by cell against the CSV (or journal) their own caption names: `tab:layouts`
(`echelle-formats.csv`, including the derived `of bound` column),
`tab:campaign` (`campagne-finale.csv`), `tab:campaign8b` (`tableau-8b.csv`,
four arms since the TACO v5 rewrite: the sealed f16-heads artifact is
tabulated next to the served int8-heads arm), `tab:scale` (`echelle-4b-8b.csv`
for perplexity and whole-model b/param, including the derived excess, its
fall and the memory margin, plus `mmlu-appariee.csv` for the paired MMLU
gap; the table lives in Appendix A since the rewrite), `tab:phases`
(`phases.csv`), `tab:seeds` (`knee-seeds.csv`, Appendix A) and `tab:e2e`
(the three-size end-to-end table of Section 4, against `campagne-finale.csv`,
`tableau-8b.csv` and the B2 journal that carries the 14B row). Tables that
are *not* checked are named at the bottom of this file, with a reason each,
rather than left implicit. `tab:progression` (`progression.csv`) was checked
until the rewrite deleted the table; the CSV stays.

Two CSVs are checked with no table behind them yet, added 2026-08-17:
`ppl-appariee.csv` and `ppl-genou.csv` carry the paired perplexity intervals
and the knee test. No tabular consumes them today -- the paper states those
numbers in prose -- so they are pinned to their journals and, where the
arithmetic allows, tied back to `echelle-4b-8b.csv`, which several checked
cells of `tab:scale` do come from. A guard written before the table exists is
the point: the numbers are in the repository now, and a table built on them
later inherits the guard instead of needing one.
"""

import csv
import math
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


def check_csv_shape() -> list[str]:
    """Every CSV in docs/data is rectangular, and none is silently truncated.

    Added 2026-08-17 after finding three rows that were NOT: an unquoted
    comma inside a free-text `notes` or `what` field splits that field, so
    `csv.DictReader` drops everything after it into the None key and the
    reader sees a note that stops mid-sentence. Nothing failed, because no
    checked table reads those columns -- which is exactly why the defect had
    survived in `campagne-finale.csv` (two rows) and `jobs.csv` (one) since
    they were written. The repository's convention is to separate clauses in
    those fields with ';' rather than quote them; this enforces it.
    """
    bad = []
    for path in sorted(DATA.glob("*.csv")):
        with open(path, newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if header is None:
                bad.append(f"{path.name}: empty file")
                continue
            for lineno, row in enumerate(reader, start=2):
                if row and len(row) != len(header):
                    bad.append(
                        f"{path.name} line {lineno}: {len(row)} fields, "
                        f"{len(header)} in the header -- an unquoted comma in a "
                        "free-text field truncates it on read"
                    )
    return bad


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
    "nullk": "No-weights control",
    "FP16": "FP16 (control)",
    "cuBLASf16": "FP16 via cuBLAS",
    "Slot32": "Slot32",
    "Planes14": "Planes14",
    "Planes12x": "Planes12x",
    "Golay70v1": "Golay70",
    "Golay70v2": "Golay70, hoisted",
    "AWQ": "AWQ w4g128",
    "QTIP": "QTIP 2-bit",
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


def table_body(tex: str, label: str) -> list[list[str]]:
    """The rows of the tabular carrying \\label{tab:<label>}, as split cells.

    Same reading as `parse_layout_table` -- everything between the first
    \\midrule and the \\bottomrule, comments and inner rules dropped -- but
    selected by label, because `evaluation.tex` holds three tabulars and a
    file-wide split would check the first one three times without saying so.
    Returns [] when no such tabular exists, which the callers report.
    """
    for chunk in re.split(r"\\begin\{table\}", tex)[1:]:
        if f"\\label{{tab:{label}}}" not in chunk or r"\midrule" not in chunk:
            continue
        body = chunk.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
        rows = []
        for line in body.splitlines():
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if line.startswith(r"\midrule") or line.startswith(r"\cmidrule"):
                continue
            rows.append([c.strip() for c in line.rstrip("\\").split("&")])
        return rows
    return []


def nums(cell: str) -> list[str]:
    """Every number a LaTeX cell prints, in print order, markup stripped.

    Unsigned on purpose: none of these cells carries a negative value, and a
    leading hyphen in LaTeX is a dash (`---` for "not measured"), not a sign.
    A cell that prints nothing numeric yields [], which is itself checked --
    a `---` that quietly becomes a number is a drift like any other.
    """
    cell = re.sub(r"\\textbf\{([^}]*)\}", r"\1", cell)
    cell = re.sub(r"\\[a-zA-Z]+", " ", cell)  # \times, \pm, \dagger, \texttt
    return re.findall(r"\d+\.?\d*", cell)


def show(values: list[str]) -> str:
    """A cell's numbers as the report prints them, absence included."""
    return ", ".join(values) if values else "(no number)"


def cell_says(where: str, expect: list[str], cell: str) -> list[str]:
    """Report the drift, with both sides spelled out, or nothing."""
    got = nums(cell)
    if got == expect:
        return []
    return [f"{where}: CSV says {show(expect)}, the table says {show(got)}"]


# The two campaign tables are transposed against the CSV: one row per metric,
# one column per arm. Each entry gives, per CSV arm, the fields that arm's
# cell prints and the decimals it prints them with -- "*" being the common
# case. An empty list is a cell that carries no number by design (AWQ has no
# throughput of ours; the FP16 arm prints no ratio against itself), and it is
# checked to carry none.
CAMPAIGN_SPEC = {
    "disk": {"*": [("disk_gb", 2)]},
    "VRAM": {
        "*": [("vram_gb", 2)],
        "awq": [("vram_bits_per_param", 2)],
        # Three decimals on our own cell, two on the AWQ's: the served figure
        # is the `rtbits` verdict on the exact bytes (5.162), and rounding it
        # here to 5.16 would let the table drift back toward the 5.15 that was
        # only ever the rounded card display. The AWQ's 5.30 is its own
        # engine's report, recorded at the precision it was reported.
        "llvq_fused": [("vram_gb", 2), ("vram_bits_per_param", 3)],
    },
    # Median then min--max, since 2026-08-19: `fusedrun` now times five
    # generations after a discarded one, and the cell prints the
    # distribution rather than the point it used to print.
    "throughput": {
        "*": [("speed_tokps", 1), ("speed_lo", 1), ("speed_hi", 1)],
        "awq": [],
    },
    "ppl (WikiText-2)": {
        "*": [("ppl", 4), ("ppl_ratio_vs_fp16", 3)],
        "fp16": [("ppl", 4)],
    },
    "MMLU micro (\\%)": {"*": [("mmlu_micro_pct", 2), ("mmlu_stderr_pp", 2)]},
}

CAMPAIGN8B_SPEC = {
    "disk": {"*": [("disk_gb", 2)]},
    "VRAM": {
        "*": [("vram_gb", 2)],
        "awq": [("vram_gb", 2), ("vram_bits_per_param", 2)],
        # The sealed f16-heads artifact is tabulated since the TACO v5
        # rewrite, so that the paired gaps of Appendix A (10.57, 7.49) are
        # reconstructible from a printed MMLU cell (65.52) rather than from
        # the served arm's 65.63. Same two-decimal convention as its
        # neighbour: the CSV records 6.461 and the cell prints 6.46.
        "llvq_f16emb": [("vram_gb", 2), ("vram_bits_per_param", 2)],
        "llvq_q8": [("vram_gb", 2), ("vram_bits_per_param", 2)],
    },
    # Median then min--max, since 2026-08-19: `fusedrun` now times five
    # generations after a discarded one, and the cell prints the
    # distribution rather than the point it used to print.
    "throughput": {
        "*": [("speed_tokps", 1), ("speed_lo", 1), ("speed_hi", 1)],
        "awq": [],
    },
    "ppl (WikiText-2)": {
        "*": [("ppl", 4), ("ppl_ratio_vs_f16", 3)],
        "fp16": [("ppl", 4)],
    },
    "MMLU micro (\\%)": {"*": [("mmlu_micro_pct", 2), ("mmlu_stderr_pp", 2)]},
}


def check_campaign_table(
    csv_name: str, label: str, arms: list[str], spec: dict, absent: dict[str, str]
) -> list[str]:
    """One campaign table, cell by cell, against the CSV its caption names.

    `arms` is the CSV key of each table column, in the table's left-to-right
    order. `absent` names the CSV arms the table deliberately does not carry,
    with the reason -- an unexplained one is reported like any other gap.
    """
    rows = {r["arm"]: r for r in csv.DictReader(open(DATA / csv_name))}
    table_rows = table_body((SEC / "evaluation.tex").read_text(), label)
    if not table_rows:
        return [f"tab:{label}: no tabular with that label in evaluation.tex"]
    table = {cells[0]: cells[1:] for cells in table_rows}
    bad = []

    for arm in arms:
        if arm not in rows:
            bad.append(f"tab:{label} column '{arm}' has no row in {csv_name}")
    for arm in rows:
        if arm not in arms and arm not in absent:
            bad.append(f"{csv_name} row '{arm}' is in no column of tab:{label}")

    for row_label, per_arm in spec.items():
        if row_label not in table:
            bad.append(f"tab:{label} has no row '{row_label}'")
            continue
        cells = table[row_label]
        if len(cells) != len(arms):
            bad.append(
                f"tab:{label} row '{row_label}': {len(cells)} value columns, "
                f"{len(arms)} arms expected"
            )
            continue
        for arm, cell in zip(arms, cells):
            if arm not in rows:
                continue
            fields = per_arm.get(arm, per_arm["*"])
            expect = [f"{float(rows[arm][f]):.{d}f}" for f, d in fields]
            bad += cell_says(f"tab:{label} / {row_label} / {arm}", expect, cell)

    for row_label in table:
        if row_label not in spec:
            bad.append(f"tab:{label} row '{row_label}' maps to no CSV field")
    return bad


# Table 4's columns are (invocation, arm, head) profiles of phases.csv. The
# head is carried here so that a CSV edit that re-labels a profile is caught
# too: the table's column headings say which head each arm ran.
PHASE_COLUMNS = [("f16", "dense", "f16"), ("f16", "fused", "f16"), ("q8", "fused", "q8")]

# Table row label -> the CSV `phase` it reports.
PHASE_ROWS = {
    "embedding": "embed",
    "transformer blocks": "blocks_norm",
    r"\texttt{lm\_head}": "lm_head",
    "argmax + misc": "argmax_misc",
}


def check_phases() -> list[str]:
    """Table 4 (tab:phases): sync-bounded medians, three of the four profiles."""
    med, heads = {}, {}
    for r in csv.DictReader(open(DATA / "phases.csv")):
        med[(r["invocation"], r["arm"], r["phase"])] = r["median_ms"]
        # A set, not the last row's value: the head is a property of the whole
        # profile, so a single re-labelled line has to show up as a set of two.
        heads.setdefault((r["invocation"], r["arm"]), set()).add(r["head"])
    table_rows = table_body((SEC / "integration.tex").read_text(), "phases")
    if not table_rows:
        return ["tab:phases: no tabular with that label in integration.tex"]
    table = {cells[0]: cells[1:] for cells in table_rows}
    bad = []

    for inv, arm, head in PHASE_COLUMNS:
        got = heads.get((inv, arm), set())
        if got != {head}:
            bad.append(
                f"phases.csv: profile ({inv}, {arm}) records head(s) "
                f"{sorted(got) or '(none)'}, tab:phases column says {head!r}"
            )

    for row_label, phase in PHASE_ROWS.items():
        if row_label not in table:
            bad.append(f"tab:phases has no row '{row_label}'")
            continue
        cells = table[row_label]
        if len(cells) != len(PHASE_COLUMNS):
            bad.append(
                f"tab:phases row '{row_label}': {len(cells)} value columns, "
                f"{len(PHASE_COLUMNS)} profiles expected"
            )
            continue
        for (inv, arm, _), cell in zip(PHASE_COLUMNS, cells):
            key = (inv, arm, phase)
            if key not in med:
                bad.append(f"phases.csv has no median for {key}")
                continue
            expect = [f"{float(med[key]):.3f}"]
            bad += cell_says(f"tab:phases / {phase} / {inv}-{arm}", expect, cell)

    for row_label in table:
        if row_label not in PHASE_ROWS:
            bad.append(f"tab:phases row '{row_label}' maps to no CSV phase")
    return bad


# Table 6's rows, in the table's top-to-bottom order, keyed by the `model`
# column of echelle-4b-8b.csv. A model in the CSV and not here (or the other
# way round) is reported: the scale story is the one place where a missing
# row *is* the defect.
SCALE_MODELS = ["Qwen3-4B", "Qwen3-8B", "Qwen3-14B"]

# The MMLU-gap column of tab:scale, pinned LITERALLY to its journals, and
# ALSO carried by docs/data/mmlu-appariee.csv. Both are checked against this
# dict, so the guard survives an edit to either one alone: a table edit fails
# against the CSV, a CSV edit fails against this pin, and moving a number for
# real means re-reading the journal and touching all three. That is the point.
#
# 2026-08-17 -- THIS PIN WAS INVERTED, and the inversion is the whole story of
# the day. It used to read `"Qwen3-14B": ["6.09"]` and to *forbid* an interval
# on that cell while *requiring* its \dagger, because the 14B run's
# per-question dumps were believed lost and 6.09 was a bare 78.21 - 72.12.
# The dumps were not lost: they were in the job's mounted output bucket, and
# the paired estimate was computed from them on 2026-08-17 for 0 USD
# (docs/mesures/mmlupair-14b-2026-08-17.txt). The point estimate did not move
# by a hundredth; it stopped being naked. All three cells are now
# subject-stratified paired bootstraps and all three must print an interval.
#
# The pin also survives the mutant that motivated it: rewriting the 14B cell
# as 6.85 -- the *paired f16-minus-2-bit* delta for the same model, from the
# same campaign -- passed every CSV-derived check silently. It now carries an
# interval of its own ([4.52, 9.12]), so the swap is still the most plausible
# drift this table has, and it is still caught here.
SCALE_GAP = {
    "Qwen3-4B": ["14.45", "11.60", "17.27"],
    "Qwen3-8B": ["7.49", "5.28", "9.70"],
    "Qwen3-14B": ["6.09", "3.62", "8.52"],
}

# The paired row of mmlu-appariee.csv each gap cell reports. Named rather
# than assumed: the same file holds `f16_minus_llvq2`, whose 14B value is the
# 6.85 above, and picking the wrong row is exactly the confusion to prevent.
SCALE_GAP_PAIR = "awq4_minus_llvq2"

# Which step of ppl-genou.csv supplies each tab:scale row's `fall` interval.
# The first row has no predecessor and so has no step; its cell is a dash, and
# that is checked too. Only the FP16 reference appears here -- `fall` is the
# relative drop of the excess over FP16, and the same reparameterisation
# against an AWQ reference would divide by an excess with no such meaning.
SCALE_FALL_STEP = {"Qwen3-8B": "4B->8B", "Qwen3-14B": "8B->14B"}
SCALE_FALL_REFERENCE = "f16"

# The two b/param columns of tab:scale, as (CSV arm, decimals). Whole-model
# accounting, embedding included -- the only one in which a memory number may
# be compared across methods, and the one the lot-A errata calls out by name.
# Three decimals on both: the served figures are `rtbits` verdicts on exact
# bytes (5.162 / 5.322 / 5.106), and two decimals would let the 4B cell drift
# back toward the 5.15 that was only ever the rounded card display.
SCALE_VRAM = [("llvq2", 3), ("awq4", 3)]


def check_scale_gap_csv() -> list[str]:
    """mmlu-appariee.csv against SCALE_GAP, before the table is read at all.

    The gap column has two sources on purpose (see SCALE_GAP). This is the
    half that confronts the CSV with the journals; check_scale confronts the
    table with the CSV. Neither alone would catch an edit to the other.
    """
    bad = []
    rows = {}
    for r in csv.DictReader(open(DATA / "mmlu-appariee.csv")):
        rows[(r["model"], r["pair"])] = r
    for model, want in SCALE_GAP.items():
        key = (model, SCALE_GAP_PAIR)
        if key not in rows:
            bad.append(f"mmlu-appariee.csv has no '{SCALE_GAP_PAIR}' row for {model}")
            continue
        r = rows[key]
        got = [r["delta_pp"], r["ci_lo_pp"], r["ci_hi_pp"]]
        if got != want:
            bad.append(
                f"mmlu-appariee.csv / {model} / {SCALE_GAP_PAIR}: the journals "
                f"say {show(want)}, the CSV says {show(got)}"
            )
        # A paired estimate with no interval, or with no journal behind it,
        # is the state this column spent a week in. It must not recur silently.
        if not r["ci_lo_pp"] or not r["ci_hi_pp"]:
            bad.append(f"mmlu-appariee.csv / {model} / {SCALE_GAP_PAIR}: no interval")
        if not r["source"]:
            bad.append(f"mmlu-appariee.csv / {model} / {SCALE_GAP_PAIR}: no source")
    return bad


def check_scale() -> list[str]:
    """Table 6 (tab:scale): FP16 ppl, 2-bit ratio, excess, fall, VRAM, gap.

    The two derived columns are recomputed here rather than trusted, because
    they carry the paper's actual claim: `excess` is the ratio minus one and
    `fall` is its relative drop from the row above, and it is the *fall* that
    the text calls a knee.

    Since 2026-08-17 the `fall` cell also prints a 95% interval, from
    `ppl-genou.csv`, and the check follows the same split as the gap column:
    the point estimate is *derived* from the two ratios in this file, the
    interval is *carried* from the paired journal, and a cell that drops its
    brackets fails -- a paired estimate printed bare is how the knee lost its
    metric in the first place.

    The two VRAM columns are whole-model b/param, embedding included. They
    are checked against the CSV, and the CSV is checked against itself: the
    recorded margin must follow from the two rates, or the CSV is internally
    inconsistent and the table would inherit it. The FP16 arm is pinned to
    16.000 exactly -- it is 2 bytes per parameter by construction, and
    `tableau-8b.csv` records 15.999 for the same quantity, which is a
    GB-to-bits rounding artefact and must not migrate here.

    The MMLU-gap column has no single source: it is pinned literally against
    the journals (SCALE_GAP, checked by check_scale_gap_csv) and carried by
    `docs/data/mmlu-appariee.csv`, which is what the table is compared to.
    The pin also enforces the shape the caption promises -- since 2026-08-17,
    an interval on all three cells and a dagger on none.
    """
    rows = {}
    for r in csv.DictReader(open(DATA / "echelle-4b-8b.csv")):
        rows[(r["model"], r["arm"])] = r
    paired = {}
    for r in csv.DictReader(open(DATA / "mmlu-appariee.csv")):
        paired[(r["model"], r["pair"])] = r
    knee = {}
    for r in csv.DictReader(open(DATA / "ppl-genou.csv")):
        knee[(r["step"], r["reference"])] = r
    # Since the TACO v5 rewrite the compact tab:scale lives in Appendix A
    # (appendix-scale.tex); evaluation.tex is kept as a fallback so the
    # check follows the table rather than the file.
    table = []
    for tex_file in ("appendix-scale.tex", "evaluation.tex"):
        table = table_body((SEC / tex_file).read_text(), "scale")
        if table:
            break
    if not table:
        return ["tab:scale: no tabular with that label in appendix-scale.tex or evaluation.tex"]
    if len(table) != len(SCALE_MODELS):
        return [
            f"tab:scale has {len(table)} rows, {len(SCALE_MODELS)} models expected"
        ]
    bad = []
    csv_models = {m for m, _ in rows}
    for model in csv_models - set(SCALE_MODELS):
        bad.append(f"echelle-4b-8b.csv has model '{model}' in no row of tab:scale")

    prev_excess = None
    for model, cells in zip(SCALE_MODELS, table):
        if cells[0] != model:
            bad.append(f"tab:scale: row expected '{model}', the table says {cells[0]!r}")
            continue
        if (model, "f16") not in rows or (model, "llvq2") not in rows:
            bad.append(f"echelle-4b-8b.csv lacks an f16 or llvq2 row for {model}")
            continue
        if len(cells) != 8:
            bad.append(f"tab:scale row '{model}': {len(cells)} columns, 8 expected")
            continue
        ratio = float(rows[(model, "llvq2")]["ppl_ratio_vs_f16"])
        excess = ratio - 1.0
        bad += cell_says(
            f"tab:scale / {model} / FP16 ppl",
            [f"{float(rows[(model, 'f16')]['ppl']):.4f}"],
            cells[1],
        )
        bad += cell_says(f"tab:scale / {model} / 2-bit ratio", [f"{ratio:.4f}"], cells[2])
        bad += cell_says(f"tab:scale / {model} / excess", [f"{excess:.4f}"], cells[3])
        # The first row has no predecessor, so its `fall` cell must be a dash.
        bad += check_scale_fall(model, excess, prev_excess, knee, cells[4])
        prev_excess = excess

        bad += check_scale_vram(model, rows, cells)

        # The gap cell: the CSV's paired triplet, then the shape the caption
        # promises. Both halves matter -- the values could be right while the
        # cell drops its brackets, which is how a paired estimate gets read
        # as a bare difference.
        pair = paired.get((model, SCALE_GAP_PAIR))
        if pair is None:
            bad.append(f"mmlu-appariee.csv has no '{SCALE_GAP_PAIR}' row for {model}")
            continue
        want_gap = [pair["delta_pp"], pair["ci_lo_pp"], pair["ci_hi_pp"]]
        got_gap = nums(cells[7])
        if got_gap != want_gap:
            bad.append(
                f"tab:scale / {model} / MMLU gap: mmlu-appariee.csv says "
                f"{show(want_gap)}, the table says {show(got_gap)}"
            )
            continue
        if "[" not in cells[7]:
            bad.append(
                f"tab:scale / {model} / MMLU gap: {want_gap[0]} is a paired "
                "bootstrap estimate and must print its interval"
            )
        # Since 2026-08-17 no cell in this column is an unpaired difference,
        # so the dagger that used to mark the 14B has nothing left to mark.
        # A dagger reappearing means the retired caveat came back with it.
        if "dagger" in cells[7]:
            bad.append(
                f"tab:scale / {model} / MMLU gap: all three cells are paired "
                "estimates since 2026-08-17; the \\dagger marker and its "
                "caption note are retired and must not return"
            )
    return bad


def check_scale_fall(
    model: str, excess: float, prev_excess, knee: dict, cell: str
) -> list[str]:
    """One tab:scale `fall` cell: derived point estimate, carried interval.

    The point estimate is recomputed from the two ratios of
    `echelle-4b-8b.csv`, so the cell cannot drift from the perplexities in its
    own table. The interval is not derivable from anything in that file --- it
    comes from pairing 12 windows across two campaigns --- so it is carried
    from `ppl-genou.csv`, which `check_ppl_knee` in turn pins to its journal
    and ties to its own log-ratio columns. Neither half alone would catch an
    edit to the other.
    """
    if prev_excess is None:
        # No predecessor: a dash, and nothing numeric. A number appearing here
        # would be a fall computed against a row that does not exist.
        return cell_says(f"tab:scale / {model} / fall", [], cell)
    derived = f"{(1 - excess / prev_excess) * 100:.1f}"
    key = (SCALE_FALL_STEP[model], SCALE_FALL_REFERENCE)
    r = knee.get(key)
    if r is None:
        return [f"ppl-genou.csv has no '{key[0]}' row for reference {key[1]}"]
    # The CSV records the fall as a signed percentage and the table prints the
    # sign as a LaTeX minus, which `nums` drops; compare magnitudes, and check
    # the CSV's own sign separately so a fall that turned into a rise fails.
    recorded = [r["excess_fall_pct"], r["excess_fall_lo_pct"], r["excess_fall_hi_pct"]]
    if not all(recorded):
        return [f"ppl-genou.csv / {key[0]} / {key[1]}: no excess-fall interval"]
    if f"-{derived.lstrip('-')}" != recorded[0]:
        return [
            f"tab:scale / {model} / fall: ppl-genou.csv records "
            f"{recorded[0]}, {derived} implied by the excesses in "
            "echelle-4b-8b.csv"
        ]
    if any(float(v) >= 0.0 for v in recorded):
        return [
            f"ppl-genou.csv / {key[0]} / {key[1]}: an excess-fall column is not "
            "negative, so the excess did not fall and `fall` is the wrong word"
        ]
    bad = cell_says(
        f"tab:scale / {model} / fall",
        [v.lstrip("-") for v in recorded],
        cell,
    )
    if not bad and "[" not in cell:
        bad.append(
            f"tab:scale / {model} / fall: {recorded[0]}\\% is a paired estimate "
            "over 12 windows and must print its interval"
        )
    return bad


def check_scale_vram(model: str, rows: dict, cells: list[str]) -> list[str]:
    """The two b/param columns of one tab:scale row, plus the CSV's own margin."""
    bad = []
    for (arm, dec), cell in zip(SCALE_VRAM, cells[5:7]):
        if (model, arm) not in rows:
            bad.append(f"echelle-4b-8b.csv lacks a {arm} row for {model}")
            continue
        r = rows[(model, arm)]
        if not r["vram_config"] or not r["vram_source"]:
            bad.append(
                f"echelle-4b-8b.csv / {model} / {arm}: b/param "
                f"{r['vram_bits_per_param']} carries no config or no source"
            )
        expect = [f"{float(r['vram_bits_per_param']):.{dec}f}"]
        bad += cell_says(f"tab:scale / {model} / b/param {arm}", expect, cell)

    # 16 bits per parameter is two bytes, exactly; nothing measures it.
    f16 = rows.get((model, "f16"), {}).get("vram_bits_per_param")
    if f16 != "16.000":
        bad.append(
            f"echelle-4b-8b.csv / {model} / f16: b/param recorded {f16!r}, "
            "16.000 exactly by construction (2 bytes per parameter)"
        )

    # The margin the prose quotes is derived; if it does not follow from the
    # two rates in this file, one of the three is wrong.
    ours = rows[(model, "llvq2")]["vram_bits_per_param"]
    theirs = rows[(model, "awq4")]["vram_bits_per_param"]
    derived = f"{(float(ours) / float(theirs) - 1) * 100:.1f}"
    recorded = rows[(model, "llvq2")]["vram_margin_vs_awq_pct"]
    if derived != recorded:
        bad.append(
            f"echelle-4b-8b.csv / {model} / vram_margin_vs_awq_pct: "
            f"{recorded} recorded, {derived} implied by {ours} over {theirs}"
        )
    return bad


# The nine paired perplexity intervals, pinned LITERALLY to their journals --
# same device as SCALE_GAP, and for the same reason: the CSV is the only copy
# outside a .txt, so an edit to it must fail against something.
#
# 2026-08-17 -- the three 4B rows are the ones this pin exists for. Until that
# morning the repository stated, in the 8B/14B journal's own words, that "none
# of the 4B's three pairs can be formed at all" and that restoring them meant
# re-running two arms of the campaign for about 0.25 USD. That was wrong about
# where the data was, not about what its absence cost: Hugging Face job logs
# are not purged, `hf jobs logs 6a746d8f...` returned the 36 per-window NLL
# lines of the 2026-08-06 campaign, and the raw output is now committed as
# docs/mesures/a4-campagne-4b-ppl-BRUT-2026-08-06.txt. The point estimates did
# not move -- 12.2369 / 13.5207 / 16.9422 replay from the NLLs to the
# ten-thousandth -- they stopped being bare.
PPL_PAIRED = {
    ("Qwen3-4B", "awq4_over_f16"): ["10.49", "8.55", "12.47"],
    ("Qwen3-4B", "llvq2_over_f16"): ["38.45", "33.62", "43.45"],
    ("Qwen3-4B", "llvq2_over_awq4"): ["25.31", "20.01", "30.84"],
    ("Qwen3-8B", "awq4_over_f16"): ["4.80", "4.24", "5.35"],
    ("Qwen3-8B", "llvq2_over_f16"): ["22.01", "19.37", "24.70"],
    ("Qwen3-8B", "llvq2_over_awq4"): ["16.42", "14.17", "18.72"],
    ("Qwen3-14B", "awq4_over_f16"): ["3.81", "3.27", "4.34"],
    ("Qwen3-14B", "llvq2_over_f16"): ["18.94", "17.22", "20.68"],
    ("Qwen3-14B", "llvq2_over_awq4"): ["14.58", "13.10", "16.08"],
}

# The pairs whose ratio is also recorded, per arm, in echelle-4b-8b.csv. Only
# the two FP16-referenced ones are: `llvq2_over_awq4` is a quotient of two
# already-rounded ratios there and differs in the fourth decimal at 14B
# (1.1457 derived, 1.1458 measured), so deriving it would enforce a rounding
# artefact rather than an agreement. It stays pinned and underived.
PPL_PAIR_TO_ARM = {"awq4_over_f16": "awq4", "llvq2_over_f16": "llvq2"}

# Slack for the two additivity identities below. They are exact in the
# journals -- the per-window residuals are literally 0.0 in double precision --
# but the CSV records the means rounded to nine decimals, so a sum of three
# such values can miss by 1.5e-9 for that reason alone (the observed misses
# are 1e-9). Set by the recorded precision, not by taste: the defect these
# identities exist to catch is a one-window offset between two arms, which
# moves a mean by ~1e-2 nat, seven orders of magnitude above this.
ADDITIVITY_TOL = 5e-9


def check_ppl_paired() -> list[str]:
    """ppl-appariee.csv: nine paired intervals, pinned and cross-tied.

    Two independent halves. The pin confronts the CSV with the journals. The
    cross-tie confronts it with `echelle-4b-8b.csv`, whose `ppl_ratio_vs_f16`
    is the same quantity reached by a different route -- aggregate ppl of two
    arms, versus exp of the mean per-window NLL difference. They are equal by
    an identity (every window scores the same token count), so any drift
    between the two files is a real defect and not a rounding gap.
    """
    bad = []
    rows = {}
    for r in csv.DictReader(open(DATA / "ppl-appariee.csv")):
        rows[(r["model"], r["pair"])] = r
    scale = {}
    for r in csv.DictReader(open(DATA / "echelle-4b-8b.csv")):
        scale[(r["model"], r["arm"])] = r

    for key, want in PPL_PAIRED.items():
        if key not in rows:
            bad.append(f"ppl-appariee.csv has no row for {key[0]} / {key[1]}")
            continue
        r = rows[key]
        got = [r["excess_pct"], r["ci_lo_pct"], r["ci_hi_pct"]]
        if got != want:
            bad.append(
                f"ppl-appariee.csv / {key[0]} / {key[1]}: the journals say "
                f"{show(want)}, the CSV says {show(got)}"
            )
        for field in ("fingerprint", "source", "t_stat"):
            if not r[field]:
                bad.append(f"ppl-appariee.csv / {key[0]} / {key[1]}: no {field}")
        # A paired interval that no longer excludes zero would be a different
        # claim; the prose in evaluation.tex says all nine do.
        if float(r["ci_lo_pct"]) <= 0.0:
            bad.append(
                f"ppl-appariee.csv / {key[0]} / {key[1]}: interval reaches zero, "
                "which contradicts the claim that all nine exclude it"
            )
        arm = PPL_PAIR_TO_ARM.get(key[1])
        if arm and (key[0], arm) in scale:
            derived = f"{float(scale[(key[0], arm)]['ppl_ratio_vs_f16']):.4f}"
            if derived != f"{float(r['ratio']):.4f}":
                bad.append(
                    f"ppl-appariee.csv / {key[0]} / {key[1]}: ratio {r['ratio']}, "
                    f"echelle-4b-8b.csv records {derived} for arm {arm}"
                )
        bad += ppl_row_is_self_consistent(key, r)

    # Additivity of the three paired means, which is exact: the per-window
    # differences telescope, so (f16 to AWQ) + (AWQ to 2-bit) is (f16 to
    # 2-bit) to floating point. Both journals run this as their second
    # reproduction control, because a one-window offset between two arms
    # breaks it and nothing else would show. Running it on the CSV catches a
    # row edited in isolation.
    for model in {m for m, _ in PPL_PAIRED}:
        try:
            a, b, c = (float(rows[(model, p)]["mean_nll_diff"]) for p in
                       ("awq4_over_f16", "llvq2_over_awq4", "llvq2_over_f16"))
        except KeyError:
            continue
        if abs(a + b - c) > ADDITIVITY_TOL:
            bad.append(
                f"ppl-appariee.csv / {model}: the three paired means are not "
                f"additive ({a} + {b} != {c}); one of the rows is not from the "
                "same 12 windows as the other two"
            )

    for key in rows:
        if key not in PPL_PAIRED:
            bad.append(f"ppl-appariee.csv row {key} is pinned to no journal value")
    return bad


def ppl_row_is_self_consistent(key: tuple, r: dict) -> list[str]:
    """One ppl-appariee.csv row against its own columns.

    The pin covers the three percentages; without this, `ratio`,
    `mean_nll_diff`, `se_nll_diff` and `t_stat` are carried but unguarded, and
    a mutation to any of them passes. Each is tied to the pinned columns by an
    exact relation rather than a tolerance: the ratio is the exponential of
    the mean difference (every window scores the same 4095 tokens, so the
    aggregation is a plain mean and the exponentiation is an identity), the
    excess is the ratio minus one, and t is the mean over its standard error.
    """
    bad = []
    where = f"ppl-appariee.csv / {key[0]} / {key[1]}"
    mean, se = float(r["mean_nll_diff"]), float(r["se_nll_diff"])
    if f"{math.exp(mean):.4f}" != r["ratio"]:
        bad.append(
            f"{where}: ratio {r['ratio']}, {math.exp(mean):.4f} implied by the "
            f"mean NLL difference {r['mean_nll_diff']}"
        )
    for pct, ratio, name in (
        (r["excess_pct"], r["ratio"], "excess_pct"),
        (r["ci_lo_pct"], r["ratio_lo"], "ci_lo_pct"),
        (r["ci_hi_pct"], r["ratio_hi"], "ci_hi_pct"),
    ):
        derived = f"{(float(ratio) - 1) * 100:.2f}"
        if derived != pct:
            bad.append(f"{where}: {name} {pct}, {derived} implied by {ratio}")
    if f"{mean / se:.2f}" != r["t_stat"]:
        bad.append(
            f"{where}: t {r['t_stat']}, {mean / se:.2f} implied by the mean "
            "over its standard error"
        )
    return bad


# The knee, pinned to docs/mesures/ppl-appariee-4b-2026-08-17.txt. Each row is
# a paired difference of differences over the same 12 windows -- the token
# fingerprint is common to all three campaigns, which is what makes pairing
# across model sizes legitimate at all.
#
# 2026-08-17 -- READ THE `resolved` COLUMN WITH ITS METRIC ATTACHED. On
# perplexity the slowdown is resolved: the two steps differ by -0.1010 in log
# ratio, interval [-0.1377, -0.0643], t = -6.06. On the MMLU gap to 4 bits it
# is NOT: 6.96 points then 1.40, p = 0.0001 then p = 0.40 (mmlu-appariee.csv,
# standard errors composed in quadrature). Both statements are true and this
# file only carries the first, so a sentence sourced from here must name
# perplexity or it is half wrong.
# 2026-08-17 -- AND READ `factor` IN THE RIGHT PARAMETERIZATION. It is the
# ratio of two perplexity RATIOS (1.2201/1.3845 = 0.8813), not of two
# EXCESSES (0.2201/0.3845 = 0.5724), which is the `excess_fall_pct` column and
# the table's `fall`. The source journal's inline heading calls the step a
# "facteur d'excès", which is the wrong label on a right number -- the same
# journal gives the excess reading separately and correctly as -42.8%. The two
# differ by a factor of 1.5 here, so a sentence that mixes them is not a
# rounding matter; `knee_factor_is_a_ratio_of_ratios` below makes the
# distinction fail loudly rather than rely on anyone reading this comment.
PPL_KNEE = {
    ("4B->8B", "f16"): ["0.881211", "0.856093", "0.907067"],
    ("8B->14B", "f16"): ["0.974855", "0.958819", "0.991159"],
    ("4B->8B", "awq4"): ["0.929104", "0.894695", "0.964836"],
    ("8B->14B", "awq4"): ["0.984154", "0.968597", "0.999962"],
    ("knee", "f16"): ["0.903941", "0.871386", "0.937711"],
    ("knee", "awq4"): ["0.944063", "0.903818", "0.986100"],
}

# Which pair of models each step spans, for the derived `excess_fall_pct`.
PPL_STEP_MODELS = {"4B->8B": ("Qwen3-4B", "Qwen3-8B"),
                   "8B->14B": ("Qwen3-8B", "Qwen3-14B")}


def check_ppl_knee() -> list[str]:
    """ppl-genou.csv: four steps and two knee tests, pinned and derived.

    `excess_fall_pct` is the column tab:scale prints as `fall` and the text
    calls a knee, so it is recomputed from `echelle-4b-8b.csv` rather than
    trusted -- exactly as check_scale recomputes the table's own cell. It is
    filled for the FP16 reference only: excess over FP16 is the quantity that
    must reach zero for the thesis to close, and the same reparameterisation
    against an AWQ reference would divide by an excess that has no such
    meaning. The empty cells are checked to stay empty.
    """
    bad = []
    rows = {}
    for r in csv.DictReader(open(DATA / "ppl-genou.csv")):
        rows[(r["step"], r["reference"])] = r
    scale = {}
    for r in csv.DictReader(open(DATA / "echelle-4b-8b.csv")):
        scale[(r["model"], r["arm"])] = r

    for key, want in PPL_KNEE.items():
        if key not in rows:
            bad.append(f"ppl-genou.csv has no row for {key[0]} / {key[1]}")
            continue
        r = rows[key]
        got = [r["factor"], r["factor_lo"], r["factor_hi"]]
        if got != want:
            bad.append(
                f"ppl-genou.csv / {key[0]} / {key[1]}: the journal says "
                f"{show(want)}, the CSV says {show(got)}"
            )
        if not r["source"]:
            bad.append(f"ppl-genou.csv / {key[0]} / {key[1]}: no source")
        mean, se = float(r["mean_logratio_diff"]), float(r["se_logratio_diff"])
        if f"{math.exp(mean):.6f}" != r["factor"]:
            bad.append(
                f"ppl-genou.csv / {key[0]} / {key[1]}: factor {r['factor']}, "
                f"{math.exp(mean):.6f} implied by {r['mean_logratio_diff']}"
            )
        if f"{mean / se:.2f}" != r["t_stat"]:
            bad.append(
                f"ppl-genou.csv / {key[0]} / {key[1]}: t {r['t_stat']}, "
                f"{mean / se:.2f} implied by the mean over its standard error"
            )
        fall = [r["excess_fall_pct"], r["excess_fall_lo_pct"], r["excess_fall_hi_pct"]]
        if key[0] == "knee" or key[1] != "f16":
            if any(fall):
                bad.append(
                    f"ppl-genou.csv / {key[0]} / {key[1]}: excess-fall columns "
                    "are for the FP16 reference of a single step only"
                )
            continue
        old, new = PPL_STEP_MODELS[key[0]]
        e_old = float(scale[(old, "llvq2")]["ppl_ratio_vs_f16"]) - 1.0
        e_new = float(scale[(new, "llvq2")]["ppl_ratio_vs_f16"]) - 1.0
        derived = f"{(e_new / e_old - 1) * 100:.1f}"
        if derived != r["excess_fall_pct"]:
            bad.append(
                f"ppl-genou.csv / {key[0]} / f16: excess_fall_pct "
                f"{r['excess_fall_pct']} recorded, {derived} implied by the "
                f"{old} and {new} ratios in echelle-4b-8b.csv"
            )
        if not (float(r["excess_fall_lo_pct"]) < float(r["excess_fall_pct"])
                < float(r["excess_fall_hi_pct"])):
            bad.append(
                f"ppl-genou.csv / {key[0]} / f16: the excess-fall point estimate "
                "is not inside its own interval"
            )
    bad += knee_factor_is_a_ratio_of_ratios(rows, scale)

    # The knee IS the difference of the two steps, so its mean must be their
    # difference exactly -- the paired means are linear in the per-window
    # terms. This is the one relation in the file that cannot hold by accident:
    # a knee row computed from anything but these two steps fails it.
    for ref in {r for _, r in PPL_KNEE}:
        try:
            s1 = float(rows[("4B->8B", ref)]["mean_logratio_diff"])
            s2 = float(rows[("8B->14B", ref)]["mean_logratio_diff"])
            k = float(rows[("knee", ref)]["mean_logratio_diff"])
        except KeyError:
            continue
        if abs((s1 - s2) - k) > ADDITIVITY_TOL:
            bad.append(
                f"ppl-genou.csv / knee / {ref}: {k} recorded, {s1 - s2} implied "
                "by the two steps it is the difference of"
            )
        # And the sign is the claim: a knee means the first step falls harder.
        if k >= 0.0:
            bad.append(
                f"ppl-genou.csv / knee / {ref}: {k} is not negative, so the "
                "first step does not fall harder and there is no knee to report"
            )

    for key in rows:
        if key not in PPL_KNEE:
            bad.append(f"ppl-genou.csv row {key} is pinned to no journal value")
    return bad


# A step's `factor` is a ratio of two four-decimal published ratios reached by
# a different route (paired means recorded to nine decimals), so the two agree
# to about 5e-5 and not exactly. This tolerance is 10x that and still four
# orders of magnitude below the defect it exists to catch: reading `factor` in
# the excess parameterization would put 0.5724 where 0.8813 belongs, a miss of
# 0.31.
KNEE_FACTOR_TOL = 5e-4


def knee_factor_is_a_ratio_of_ratios(rows: dict, scale: dict) -> list[str]:
    """The `factor` of each FP16 step, against the two ratios it spans.

    Guards a parameterization, not a value. `factor` is the ratio of two
    perplexity ratios; `excess_fall_pct` in the same row is the fall of the
    excess, and tab:scale prints the second. They differ by half again at
    these numbers, and the journal this file is pinned to labels the first
    "facteur d'excès" -- a wrong label on a right number, which is exactly the
    invitation to "fix" the number to match the label. Doing so fails here.
    """
    bad = []
    for step, (old, new) in PPL_STEP_MODELS.items():
        r = rows.get((step, "f16"))
        if r is None:
            continue
        try:
            r_old = float(scale[(old, "llvq2")]["ppl_ratio_vs_f16"])
            r_new = float(scale[(new, "llvq2")]["ppl_ratio_vs_f16"])
        except KeyError:
            bad.append(f"echelle-4b-8b.csv lacks an llvq2 ratio for {old} or {new}")
            continue
        implied = r_new / r_old
        if abs(float(r["factor"]) - implied) > KNEE_FACTOR_TOL:
            excess = (r_new - 1.0) / (r_old - 1.0)
            hint = (
                " -- that is the ratio of the two EXCESSES, and this column is "
                "the ratio of the two RATIOS"
                if abs(float(r["factor"]) - excess) <= KNEE_FACTOR_TOL
                else ""
            )
            bad.append(
                f"ppl-genou.csv / {step} / f16: factor {r['factor']}, "
                f"{implied:.6f} implied by the {old} and {new} perplexity "
                f"ratios in echelle-4b-8b.csv{hint}"
            )
    return bad


def signed_nums(cell: str) -> list[str]:
    """Every number a LaTeX cell prints, sign kept, as canonical floats.

    `nums` drops signs because no cell it reads carries one. The seed table
    does: the knee changes sign between the second and third re-draw, and a
    check that ignored the sign would pass a row whose claim is inverted.
    Math mode, \textbf and macros are stripped; a "+" is kept as a sign.
    """
    cell = re.sub(r"\\textbf\{([^}]*)\}", r"\1", cell)
    cell = re.sub(r"\\[a-zA-Z]+", " ", cell)
    cell = cell.replace("--", " ")  # a LaTeX en dash in a range is not a sign
    return [str(float(x)) for x in re.findall(r"[-+]?\d+\.?\d*", cell)]


def fmt(value: str, decimals: int) -> str:
    """A CSV value rounded as the table prints it, as a canonical float."""
    return str(float(f"{float(value):.{decimals}f}"))


# The seed table's rows, in CSV order, with the label each prints.
SEEDS_ROWS = {"published": "published", "seed1": "seed 1",
              "seed2": "seed 2", "seed3": "seed 3"}

# Tolerance for `knee = step1 - step2` on values the CSV records to four
# decimals: two roundings of 5e-5 each, so a miss of 1e-4 is rounding and a
# miss of 1e-3 is a knee that was not computed from its own two steps.
SEEDS_TOL = 1.5e-4


def check_seeds() -> list[str]:
    """Table tab:seeds (Appendix A): knee-seeds.csv, every cell, sign kept.

    The table replays the knee with each calibration re-draw of the 4B in
    place of the published artifact; the second step is the same in every
    row because the 8B and 14B are not re-drawn. Three things are checked:
    the cells against the CSV, the CSV against itself (the knee is the
    difference of its two steps; `knee_excludes_zero` follows from the
    interval), and the published row against ppl-genou.csv, which carries
    the same three numbers to nine decimals and is pinned to its journal.
    """
    rows = {r["arm_4b"]: r for r in csv.DictReader(open(DATA / "knee-seeds.csv"))}
    table = table_body((SEC / "appendix-scale.tex").read_text(), "seeds")
    if not table:
        return ["tab:seeds: no tabular with that label in appendix-scale.tex"]
    bad = []
    if len(table) != len(SEEDS_ROWS):
        bad.append(f"tab:seeds has {len(table)} rows, {len(SEEDS_ROWS)} expected")
    for arm in rows:
        if arm not in SEEDS_ROWS:
            bad.append(f"knee-seeds.csv row '{arm}' is in no row of tab:seeds")

    for (arm, label), cells in zip(SEEDS_ROWS.items(), table):
        if arm not in rows:
            bad.append(f"knee-seeds.csv has no row '{arm}'")
            continue
        r = rows[arm]
        if cells[0] != label:
            bad.append(f"tab:seeds: row expected {label!r}, the table says {cells[0]!r}")
            continue
        if len(cells) != 6:
            bad.append(f"tab:seeds row '{label}': {len(cells)} columns, 6 expected")
            continue
        want = [
            ("4B ppl", [fmt(r["ppl_4b"], 4)], cells[1]),
            ("step 4B->8B", [fmt(r["step1_logratio"], 4), fmt(r["step1_t"], 2)], cells[2]),
            ("step 8B->14B", [fmt(r["step2_logratio"], 4), fmt(r["step2_t"], 2)], cells[3]),
            ("knee", [fmt(r["knee_logratio"], 4), fmt(r["knee_lo"], 4),
                      fmt(r["knee_hi"], 4), fmt(r["knee_t"], 2)], cells[4]),
        ]
        for field, expect, cell in want:
            got = signed_nums(cell)
            if got != expect:
                bad.append(
                    f"tab:seeds / {label} / {field}: CSV says {show(expect)}, "
                    f"the table says {show(got)}"
                )
        if cells[5].strip() != r["knee_excludes_zero"]:
            bad.append(
                f"tab:seeds / {label} / excludes zero: CSV says "
                f"{r['knee_excludes_zero']!r}, the table says {cells[5]!r}"
            )
        if not r["source"]:
            bad.append(f"knee-seeds.csv / {arm}: no source")

        # The CSV against itself.
        s1, s2, k = (float(r[f]) for f in ("step1_logratio", "step2_logratio", "knee_logratio"))
        if abs((s1 - s2) - k) > SEEDS_TOL:
            bad.append(
                f"knee-seeds.csv / {arm}: knee {k} recorded, {s1 - s2:.4f} "
                "implied by the two steps it is the difference of"
            )
        lo, hi = float(r["knee_lo"]), float(r["knee_hi"])
        excludes = "yes" if (lo > 0.0 or hi < 0.0) else "no"
        if r["knee_excludes_zero"] != excludes:
            bad.append(
                f"knee-seeds.csv / {arm}: knee_excludes_zero {r['knee_excludes_zero']!r}, "
                f"{excludes!r} implied by the interval [{lo}, {hi}]"
            )
        if not (lo < k < hi):
            bad.append(f"knee-seeds.csv / {arm}: the knee is not inside its own interval")

    # The published row is the knee of ppl-genou.csv, to four decimals.
    genou = {}
    for r in csv.DictReader(open(DATA / "ppl-genou.csv")):
        genou[(r["step"], r["reference"])] = r
    pub = rows.get("published")
    if pub is not None:
        for field, key in (("step1_logratio", ("4B->8B", "f16")),
                           ("step2_logratio", ("8B->14B", "f16")),
                           ("knee_logratio", ("knee", "f16"))):
            g = genou.get(key)
            if g is None:
                bad.append(f"ppl-genou.csv has no row for {key[0]} / {key[1]}")
                continue
            if fmt(g["mean_logratio_diff"], 4) != fmt(pub[field], 4):
                bad.append(
                    f"knee-seeds.csv / published / {field}: {pub[field]}, "
                    f"ppl-genou.csv records {g['mean_logratio_diff']}"
                )
            t_field = field.replace("logratio", "t")
            if fmt(g["t_stat"], 2) != fmt(pub[t_field], 2):
                bad.append(
                    f"knee-seeds.csv / published / {t_field}: {pub[t_field]}, "
                    f"ppl-genou.csv records {g['t_stat']}"
                )
    return bad


# The B2 journal carries the only copy of the 14B end-to-end row and of the
# six published ratios (quotients of medians, formed within one invocation).
# Its table is parsed by shape: size, config, fused median [lo-hi], dense
# median [lo-hi], x ratio [envelope], GB fused / dense (division).
E2E_JOURNAL = ROOT / "docs" / "mesures" / "b2-fusedrun-plages-2026-08-18.txt"
E2E_LINE = re.compile(
    r"^(4B|8B|14B)\s+(q8|f16)\s.*?"
    r"([\d,]+) \[([\d,]+)[–-]([\d,]+)\]\s+"      # fused tok/s
    r"([\d,]+) \[([\d,]+)[–-]([\d,]+)\]\s+"      # dense tok/s
    r"×([\d,]+) \[[^\]]*\]\s+"                   # ratio of medians
    r"([\d,]+) / ([\d,]+)"                       # GB fused / dense
)


def read_e2e_journal() -> dict[tuple[str, str], dict]:
    """The six (size, config) rows of the B2 table, decimal commas converted."""
    rows = {}
    for line in E2E_JOURNAL.read_text().splitlines():
        m = E2E_LINE.match(line)
        if not m:
            continue
        f = [v.replace(",", ".") for v in m.groups()[2:]]
        rows[(m.group(1), m.group(2))] = {
            "fused": f[0], "fused_lo": f[1], "fused_hi": f[2],
            "dense": f[3], "dense_lo": f[4], "dense_hi": f[5],
            "ratio": f[6], "gb_fused": f[7], "gb_dense": f[8],
        }
    return rows


def check_e2e() -> list[str]:
    """Table tab:e2e (Section 4): three sizes, same-head and served, one L40S.

    Columns: dense tok/s [range], fused same-head [range], gain, fused served
    [range], gain, VRAM dense, VRAM served. The 4B and 8B cells are checked
    against the campaign CSVs their caption names; the 14B row and the six
    gains against the B2 journal, which is the only place they are recorded.
    The served VRAM of the 4B (2.60, card reading) and 8B (5.45) come from
    the CSVs, as the caption says; the journal's host-side counts for the
    same arms (2.56, 5.41) are not what the table prints, by declaration.
    Where a CSV and the journal both carry a value (fused medians and
    ranges) they are also checked against each other.
    """
    journal = read_e2e_journal()
    if len(journal) != 6:
        return [f"{E2E_JOURNAL.name}: {len(journal)} table rows parsed, 6 expected"]
    c4 = {r["arm"]: r for r in csv.DictReader(open(DATA / "campagne-finale.csv"))}
    c8 = {r["arm"]: r for r in csv.DictReader(open(DATA / "tableau-8b.csv"))}
    table = table_body((SEC / "integration.tex").read_text(), "e2e")
    if not table:
        return ["tab:e2e: no tabular with that label in integration.tex"]
    rows = {cells[0]: cells[1:] for cells in table}
    bad = []

    def speed(r: dict, k: str) -> list[str]:
        return [fmt(r[k], 1), fmt(r[k + "_lo"], 1), fmt(r[k + "_hi"], 1)]

    def csv_speed(r: dict) -> list[str]:
        return [fmt(r["speed_tokps"], 1), fmt(r["speed_lo"], 1), fmt(r["speed_hi"], 1)]

    # (size, dense source, same-head source, served source, dense GB, served GB)
    # -- each source is a dict with the speed triplet already formatted.
    spec = {
        "4B": dict(dense=csv_speed(c4["fp16"]),
                   head=speed(journal[("4B", "f16")], "fused"),
                   served=csv_speed(c4["llvq_fused"]),
                   gb_dense=fmt(c4["fp16"]["vram_gb"], 2),
                   gb_served=fmt(c4["llvq_fused"]["vram_gb"], 2)),
        "8B": dict(dense=csv_speed(c8["fp16"]),
                   head=csv_speed(c8["llvq_f16emb"]),
                   served=csv_speed(c8["llvq_q8"]),
                   gb_dense=fmt(c8["fp16"]["vram_gb"], 2),
                   gb_served=fmt(c8["llvq_q8"]["vram_gb"], 2)),
        "14B": dict(dense=speed(journal[("14B", "q8")], "dense"),
                    head=speed(journal[("14B", "f16")], "fused"),
                    served=speed(journal[("14B", "q8")], "fused"),
                    gb_dense=fmt(journal[("14B", "q8")]["gb_dense"], 2),
                    gb_served=fmt(journal[("14B", "q8")]["gb_fused"], 2)),
    }
    for size, want in spec.items():
        if size not in rows:
            bad.append(f"tab:e2e has no row '{size}'")
            continue
        cells = rows[size]
        if len(cells) != 7:
            bad.append(f"tab:e2e row '{size}': {len(cells)} value columns, 7 expected")
            continue
        checks = [
            ("dense tok/s", want["dense"], cells[0]),
            ("same-head tok/s", want["head"], cells[1]),
            ("same-head gain", [fmt(journal[(size, "f16")]["ratio"], 2)], cells[2]),
            ("served tok/s", want["served"], cells[3]),
            ("served gain", [fmt(journal[(size, "q8")]["ratio"], 2)], cells[4]),
            ("VRAM dense", [want["gb_dense"]], cells[5]),
            ("VRAM served", [want["gb_served"]], cells[6]),
        ]
        for field, expect, cell in checks:
            got = signed_nums(cell)
            if got != expect:
                bad.append(
                    f"tab:e2e / {size} / {field}: source says {show(expect)}, "
                    f"the table says {show(got)}"
                )
    for size in rows:
        if size not in spec:
            bad.append(f"tab:e2e row '{size}' maps to no source")

    # CSV against journal, where both carry the fused medians and ranges.
    for size, arm, key, src in (("4B", "llvq_fused", ("4B", "q8"), c4),
                                ("8B", "llvq_f16emb", ("8B", "f16"), c8),
                                ("8B", "llvq_q8", ("8B", "q8"), c8)):
        if csv_speed(src[arm]) != speed(journal[key], "fused"):
            bad.append(
                f"{size} / {arm}: CSV speed {show(csv_speed(src[arm]))}, "
                f"{E2E_JOURNAL.name} says {show(speed(journal[key], 'fused'))}"
            )
    return bad



def main() -> int:
    bad = check_csv_shape()
    bad += check_layouts()
    bad += check_campaign_table(
        "campagne-finale.csv",
        "campaign",
        ["fp16", "awq", "llvq_dense", "llvq_fused"],
        CAMPAIGN_SPEC,
        absent={},
    )
    bad += check_campaign_table(
        "tableau-8b.csv",
        "campaign8b",
        ["fp16", "awq", "llvq_f16emb", "llvq_q8"],
        CAMPAIGN8B_SPEC,
        absent={},
    )
    bad += check_scale_gap_csv()
    bad += check_scale()
    bad += check_ppl_paired()
    bad += check_ppl_knee()
    bad += check_phases()
    bad += check_seeds()
    bad += check_e2e()
    if bad:
        print("table/CSV mismatch:", file=sys.stderr)
        for b in bad:
            print(f"  {b}", file=sys.stderr)
        return 1
    print(
        "tables agree with docs/data/*.csv "
        "(layouts 10 arms, campaign 4 arms, campaign8b 4 arms, scale 3 models "
        "x {ppl, b/param, paired MMLU gap}, phases 3 profiles, seeds 4 arms, "
        "e2e 3 sizes); paired-perplexity CSVs pinned to their journals "
        "(9 intervals, 4 steps, 2 knee tests -- perplexity only, the MMLU "
        "verdict on the same slowdown differs); all CSVs rectangular"
    )
    # Still not covered, and named rather than left implicit:
    #   tab:lit          — in Section 5 since the TACO v5 rewrite. No CSV, and there
    #                      should not be one: rows 1 and 3–6 are transcribed
    #                      from the original paper's Table 6, not measured
    #                      here. Its rows 2 and 7 are our own, and they repeat
    #                      cells this script does check in tab:campaign and
    #                      tab:campaign8b.
    #   tab:envelope     — the rotation kernel's shared-memory table of Section 6:
    #                      intermediate sizes are architecture constants and
    #                      the byte counts are 4x them, read from the 14B
    #                      fusedrun journal; no CSV carries them.
    #   tab:intervals    — Appendix A; its 18 cells are the nine rows of
    #                      ppl-appariee.csv and the nine of mmlu-appariee.csv,
    #                      both pinned to their journals above, but the
    #                      tabular is two-block (a second header after an
    #                      inner \midrule) and table_body() does not read it.
    #   tab:validity    — Section 6; qualitative envelope rows restating facts
    #     carried (and checked where numeric) elsewhere in the paper.
    #   tab:fairness    — Appendix B; per-arm qualitative comparison conditions,
    #     sourced from the ten-arm run journal (thresholds, registers, grids).
    #   tab:prereg, tab:provenance — Appendix B; dates, criteria and file
    #                      names, no numeric CSV behind them.
    # Two former tables are figures since the TACO v5 rewrite, and their
    # guard moved to make_figures.py ("every CSV row plotted or fail"):
    #   fig:attribution  — the five stages of the Slot32 attribution, from
    #                      attribution-slot32.csv (replaces tab:attribution).
    #   fig:a100         — the two-card dumbbell, from echelle-formats.csv and
    #                      echelle-formats-a100.csv (replaces tab:a100, whose
    #                      cell-by-cell check lived here until then).
    # One CSV fact the covered tables leave on the floor, by design and not by
    # omission: phases.csv's (q8, dense) profile is measured but not tabulated.
    # progression.csv has no table since the rewrite deleted tab:progression.
    return 0


if __name__ == "__main__":
    sys.exit(main())
