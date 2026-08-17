#!/usr/bin/env python3
"""Confront the paper's hand-typed tables with the CSVs they claim to come from.

The paper promises, in four places, that every number regenerates from
committed CSV files. That was true of the *figures* — `make_figures.py` reads
the CSVs — and false of the *tables*, which are LaTeX typed by hand from the
same sources. A promise checked only by the author's care is the failure mode
this repository documents everywhere else, so it is checked here instead.

Exit code 1 on any mismatch, with the cell named. Wired into `make`, so a
table that drifts from its CSV fails the build rather than the review.

Scope, stated so nobody mistakes it for more. Six tables are checked cell by
cell against the CSV their own caption names: `tab:layouts`
(`echelle-formats.csv`, including the derived `of bound` column),
`tab:campaign` (`campagne-finale.csv`), `tab:campaign8b` (`tableau-8b.csv`),
`tab:scale` (`echelle-4b-8b.csv` for perplexity and whole-model b/param,
including the derived excess, its fall and the memory margin, plus
`mmlu-appariee.csv` for the paired MMLU gap), `tab:phases` (`phases.csv`) and
`tab:progression` (`progression.csv`). Two tables are *not* checked, for a
reason each, and they are named at the bottom of this file rather than left
implicit.
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
    "throughput": {"*": [("speed_tokps", 1)], "awq": []},
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
        "llvq_q8": [("vram_gb", 2), ("vram_bits_per_param", 2)],
    },
    "throughput": {"*": [("speed_tokps", 1)], "awq": []},
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
    table = table_body((SEC / "evaluation.tex").read_text(), "scale")
    if not table:
        return ["tab:scale: no tabular with that label in evaluation.tex"]
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
        want_fall = [] if prev_excess is None else [f"{(1 - excess / prev_excess) * 100:.1f}"]
        bad += cell_says(f"tab:scale / {model} / fall", want_fall, cells[4])
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


MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()

# The three measured columns of Table 5, in the table's order. A decimal
# count of None means "the table must print the CSV's own digits, verbatim",
# which is stricter than rounding, not looser: it forbids the script from
# silently re-rounding a value the CSV records more precisely. The b/param
# column needs it because its rows do not share a precision -- the served 4B
# figure is 5.162, the `rtbits` verdict on the exact bytes, and formatting it
# to two decimals here would print 5.16 and let the cell drift back toward the
# 5.15 that was only ever the rounded card display.
PROGRESSION_COLUMNS = [
    ("vram_gb", 2),
    ("tokps", 1),
    ("bits_per_param_modele_entier", None),
]


def check_progression() -> list[str]:
    """Table 5 (tab:progression): the four integration steps, CSV order.

    The step column is English prose against a French CSV label, so it is not
    compared; the date is, which is what pins a table row to its CSV row.
    """
    rows = list(csv.DictReader(open(DATA / "progression.csv")))
    table = table_body((SEC / "integration.tex").read_text(), "progression")
    if not table:
        return ["tab:progression: no tabular with that label in integration.tex"]
    if len(table) != len(rows):
        return [
            f"tab:progression has {len(table)} steps, progression.csv has {len(rows)}"
        ]
    bad = []
    for r, cells in zip(rows, table):
        if len(cells) != 2 + len(PROGRESSION_COLUMNS):
            bad.append(
                f"tab:progression row '{cells[0]}': {len(cells)} columns, "
                f"{2 + len(PROGRESSION_COLUMNS)} expected"
            )
            continue
        _, month, day = r["date"].split("-")
        want_date = f"{MONTHS[int(month) - 1]} {int(day)}"
        if cells[0] != want_date:
            bad.append(
                f"tab:progression: CSV date {r['date']} reads {want_date}, "
                f"the table says {cells[0]!r}"
            )
        for (field, dec), cell in zip(PROGRESSION_COLUMNS, cells[2:]):
            expect = [r[field] if dec is None else f"{float(r[field]):.{dec}f}"]
            bad += cell_says(f"tab:progression / {r['date']} / {field}", expect, cell)
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
        ["fp16", "awq", "llvq_q8"],
        CAMPAIGN8B_SPEC,
        # The f16-embedding 8B arm is measured and cited in prose (34.4 tok/s,
        # 6.62 GB, \S integration) but deliberately not tabulated.
        absent={"llvq_f16emb": "cited in prose, not tabulated"},
    )
    bad += check_scale_gap_csv()
    bad += check_scale()
    bad += check_phases()
    bad += check_progression()
    if bad:
        print("table/CSV mismatch:", file=sys.stderr)
        for b in bad:
            print(f"  {b}", file=sys.stderr)
        return 1
    print(
        "tables agree with docs/data/*.csv "
        "(layouts 7 arms, campaign 4 arms, campaign8b 3 arms, scale 3 models "
        "x {ppl, b/param, paired MMLU gap}, phases 3 profiles, "
        "progression 4 steps); all CSVs rectangular"
    )
    # Still not covered, and named rather than left implicit:
    #   tab:lit          — no CSV, and there should not be one: rows 1 and 3–6
    #                      are transcribed from the original paper's Table 6,
    #                      not measured here. Its rows 2 and 7 are our own, and
    #                      they repeat cells this script does check in
    #                      tab:campaign and tab:campaign8b.
    #   tab:attribution  — its five stages come from two logs
    #                      (attribution-cuda, fusion-qkv-cuda), and no CSV of
    #                      that shape exists in docs/data/.
    # Two CSV facts the covered tables leave on the floor, by design and not by
    # omission: phases.csv's (q8, dense) profile is measured and plotted but
    # not tabulated, and tableau-8b.csv's llvq_f16emb arm is cited in prose.
    return 0


if __name__ == "__main__":
    sys.exit(main())
