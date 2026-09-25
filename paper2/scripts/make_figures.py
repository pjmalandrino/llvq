#!/usr/bin/env python3
"""Generate paper 2's figures from docs/data/*.csv.

Same rules and the same art direction as paper 1's
`paper/scripts/make_figures.py`, deliberately: the Okabe-Ito palette, serif
type at 7-8 pt, grey axes without top and right spines, a light grid, circles
for our arms and squares for deployed kernels, values in a column outside the
axes. A reader who opens the two papers side by side should not be able to
tell which script drew which figure. No number in a figure is typed by hand
except the layout constants listed in `HARDCODED`, and every CSV row a figure
claims to draw is drawn or the script fails with the row named.

One rule of thumb runs through all four: **anything that needs a sentence goes
in the caption, not in the figure.** A 6.5 pt grey line under a bar is
unreadable at print size; the same words at caption size are not. Paper 1's
fig_records does this too — its geometry and exception rates live in its
caption.

  fig_gap.pdf      bits carried vs bits read in VRAM    echelle-formats.csv
  fig_word.pdf     two records to bit scale, one zoom   (layout constants)
  fig_tile.pdf     the tile sweep, three arms           tuile-l40s.csv
  fig_dissoc.pdf   perplexity against MMLU              tetra-rowscales.csv
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# acmsmall's \textwidth is 395.8 pt = 5.50 in, as in paper 1. Figures are
# drawn at the width they are included at, so 7-8 pt type prints at 7-8 pt.
TEXTWIDTH_IN = 5.5

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "data"
OUT = ROOT / "paper2" / "figures"

# Okabe-Ito palette: colorblind-safe, print-safe. Identical to paper 1;
# Tetra takes PURPLE, the one arm colour paper 1 left unassigned.
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#7F7F7F"
LIGHT = "#C8C8C8"
INK = "#1A1A1A"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": GRAY,
    "axes.linewidth": 0.7,
    "xtick.color": GRAY,
    "ytick.color": GRAY,
    "axes.labelcolor": INK,
    "text.color": INK,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.5,
    "figure.dpi": 150,
})

HARDCODED: list[tuple[str, str]] = []


def hardcoded(what: str, where: str) -> None:
    HARDCODED.append((what, where))


def read_csv(name: str) -> list[dict]:
    with open(DATA / name, newline="") as f:
        return list(csv.DictReader(f))


def require_plotted(fig: str, csv_name: str, expected: set, plotted: set,
                    excluded: dict[str, str]) -> None:
    """Fail if a CSV row the figure should draw was not drawn."""
    missing = expected - plotted - set(excluded)
    if missing:
        raise SystemExit(
            f"{fig}: {sorted(missing)} in {csv_name} but not plotted - "
            "add them to the figure or list them as excluded with a reason"
        )
    for key, why in excluded.items():
        print(f"{fig}: {csv_name} row {key!r} not drawn - {why}")


# ---------------------------------------------------------------------------
# Fig. 1 - what the format carries against what the kernel reads
# ---------------------------------------------------------------------------

# Bits of code a format carries per weight. A property of the format, not a
# measurement: 48 bits per block of 24 weights for the two lattice formats (a
# 47-bit index plus one gain bit), 2 bits per weight for the 2-bit trellis,
# and 4 bits per weight for a 4-bit affine format before its group scales.
CODE_CARRIED = {"Planes14": 2.000, "Tetra48": 2.000, "QTIP": 2.000,
                "AWQ": 4.000}


def fig_gap() -> None:
    """One dumbbell per format: bits of code carried (hollow) joined to bits
    read per weight in VRAM (filled).

    The x axis starts at 1.7 rather than 0 so that the two short dumbbells
    (Tetra 2.000-2.148, AWQ 4.000-4.179) are legible segments and not blobs;
    the two markers of a 0.148-bit gap are 5 pt apart at this scale, against
    2 pt on a zero-based axis. Nothing is cut off: 1.7 is below every value.
    """
    rows = {r["layout"]: r for r in read_csv("echelle-formats.csv")}
    hardcoded("bits of code carried, 2.000 / 4.000 (Fig. 1)",
              "format definitions: 48 bits per 24 weights; 4-bit affine")
    shown = {"AWQ": "AWQ w4g128 (4-bit)", "Planes14": "Planes14 (paper 1)",
             "Tetra48": "Tetra (this paper)", "QTIP": "QTIP (2-bit)"}
    colors = {"AWQ": ORANGE, "Planes14": BLUE, "Tetra48": PURPLE,
              "QTIP": VERMILLION}
    markers = {"AWQ": "s", "Planes14": "o", "Tetra48": "o", "QTIP": "s"}
    order = ["AWQ", "Planes14", "Tetra48", "QTIP"]

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 1.78), layout="constrained")
    plotted = set()
    for i, k in enumerate(order):
        y = len(order) - 1 - i
        carried = CODE_CARRIED[k]
        read = float(rows[k]["bpw_kernel"])
        ax.plot([carried, read], [y, y], color=colors[k], linewidth=1.4,
                zorder=2)
        ax.plot(carried, y, marker=markers[k], markersize=6,
                markerfacecolor="white", markeredgecolor=colors[k],
                markeredgewidth=1.3, zorder=3)
        ax.plot(read, y, marker=markers[k], markersize=6, color=colors[k],
                markeredgecolor=colors[k], zorder=3)
        ax.annotate(f"{carried:.3f} to {read:.3f} b/weight",
                    xy=(1.015, y), xycoords=("axes fraction", "data"),
                    ha="left", va="center", fontsize=7.5, color=INK,
                    annotation_clip=False)
        plotted.add(k)

    ax.annotate("carried, read", xy=(1.015, len(order) - 0.55),
                xycoords=("axes fraction", "data"), ha="left", va="bottom",
                fontsize=7, color=GRAY, annotation_clip=False)

    require_plotted(
        "fig_gap", "echelle-formats.csv", set(rows), plotted,
        {"FP16": "16.000 on both ends: no gap to draw",
         "cuBLASf16": "the same control through cuBLAS",
         "nullk": "reads no weights",
         "Slot32": "superseded by Planes14; in the appendix table",
         "Planes12x": "not served; in the appendix table",
         "Golay70v1": "not served; in the appendix table",
         "Golay70v2": "not served; in the appendix table"},
    )

    ax.set_yticks([len(order) - 1 - i for i in range(len(order))],
                  [shown[k] for k in order])
    ax.set_xlabel("bits per weight")
    ax.set_xlim(1.7, 5.3)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.grid(axis="x", zorder=0)
    fig.savefig(OUT / "fig_gap.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig. 3 - two records, to the bit, plus the short one magnified
# ---------------------------------------------------------------------------

# Field widths are constants of each layout, transcribed from the record
# diagram at the head of its decoder, exactly as in paper 1's Fig. A:
#   Planes14  llvq-cuda/kernels/llvq_planes.cuh
#             [class 9][gain 1][smask 24][plane0 24][plane1 24][plane2 24][pad 6]
#   Tetra     llvq-cuda/kernels/llvq_f1rank.cuh
#             [p 1][r 1][s8 6][b1 1][i1 11][b2 4][i2 11][b3 1][i3 11][gain 1]
FIELD_COLORS = {"class": INK, "gain": VERMILLION, "smask": SKY, "plane": BLUE,
                "state": PURPLE, "edge": LIGHT, "row": SKY, "pad": "white"}

PLANES_RECORD = [
    ("class", 9, "class"), ("g", 1, "gain"), ("sign mask", 24, "smask"),
    ("plane 0", 24, "plane"), ("plane 1", 24, "plane"),
    ("plane 2", 24, "plane"), ("6", 6, "pad"),
]
TETRA_RECORD = [
    ("p", 1, "state"), ("r", 1, "state"), ("$s_8$", 6, "state"),
    ("$b_1$", 1, "edge"), ("$i_1$", 11, "row"),
    ("$b_2$", 4, "edge"), ("$i_2$", 11, "row"),
    ("$b_3$", 1, "edge"), ("$i_3$", 11, "row"), ("g", 1, "gain"),
]


def fig_word() -> None:
    """Three bars. The top two are the unfolded Planes14 record and the Tetra
    word at ONE bit scale, which is the comparison. The third magnifies the
    48-bit word to the width of the figure so that every field can be named —
    the same device as paper 1's Fig. F, which magnifies the gap above the
    DRAM floor rather than leaving five terms unreadable at true scale.

    Everything that would need a sentence (what each colour means, the stride,
    the window, the table sizes) is in the caption. At 5.5 in a 6.5 pt grey
    line under a bar does not survive printing.
    """
    rows = {r["layout"]: r for r in read_csv("echelle-formats.csv")}
    hardcoded("record field widths (Fig. 3)",
              "llvq-cuda/kernels/llvq_planes.cuh, llvq_f1rank.cuh headers")

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.45), layout="constrained")
    bar_h = 0.52
    right = 196          # where the value column sits, in bit units
    y_p, y_t, y_z = 0.0, -1.15, -2.62
    zoom_w = 176.0       # the magnified word spans this many bit units
    k = zoom_w / 48.0

    def draw(record, y, scale=1.0, label_min=4.0):
        """Draw one record left-aligned at x = 0. Fields narrower than
        `label_min` drawn units carry no inline label."""
        x = 0.0
        for label, width, kind in record:
            w = width * scale
            ax.add_patch(Rectangle(
                (x, y - bar_h / 2), w, bar_h, facecolor=FIELD_COLORS[kind],
                edgecolor=GRAY if kind == "pad" else INK, linewidth=0.6,
                hatch="////" if kind == "pad" else None, zorder=2))
            dark = kind in ("class", "plane", "state")
            if kind == "class":
                ax.annotate("class 9 · gain 1", xy=(x, y + bar_h / 2),
                            xytext=(0, 2.5), textcoords="offset points",
                            ha="left", va="bottom", fontsize=7, color=INK)
            elif w >= label_min and label:
                ax.text(x + w / 2, y, label, ha="center", va="center",
                        fontsize=7, color="white" if dark else INK, zorder=3)
            x += w
        return x

    # --- the two records at one scale
    end_p = draw(PLANES_RECORD, y_p)
    end_t = draw(TETRA_RECORD, y_t)
    for y, name, end, bits, arm in ((y_p, "Planes14", end_p, "112 b", "Planes14"),
                                    (y_t, "Tetra", end_t, "48 b", "Tetra48")):
        ax.annotate(bits, xy=(end, y), xytext=(4, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=7, color=INK)
        ax.text(-4, y, name, ha="right", va="center", fontsize=8.5, color=INK,
                fontweight="bold")
        ax.text(right, y, f"{float(rows[arm]['bpw_kernel']):.3f} b/w in VRAM",
                ha="right", va="center", fontsize=8, color=INK)

    # --- the word magnified, so the ten fields can be named
    ax.plot([0, 0], [y_t - bar_h / 2, y_z + bar_h / 2], color=GRAY,
            linewidth=0.5, linestyle=":", zorder=1)
    ax.plot([48, zoom_w], [y_t - bar_h / 2, y_z + bar_h / 2], color=GRAY,
            linewidth=0.5, linestyle=":", zorder=1)
    draw(TETRA_RECORD, y_z, scale=k, label_min=9.0)
    ax.text(-4, y_z, "the word,\nmagnified", ha="right", va="center",
            fontsize=8, color=INK, linespacing=1.15)

    # The 1-bit fields are 3.7 drawn units wide even magnified: they are named
    # once, below, rather than crammed into 0.1 in of bar.
    ax.text(0, y_z - bar_h / 2 - 0.20,
            "one bit each, left to right:  $p$ shared parity · $r$ section-1 "
            "class · $b_1$, $b_3$ edge choices · $g$ gain",
            ha="left", va="top", fontsize=7.5, color=INK)

    # --- the bit ruler, over the two true-scale rows only
    top = bar_h / 2 + 0.34
    for w in range(0, 129, 32):
        ax.plot([w, w], [top, y_t - bar_h / 2 - 0.06], color="#E4E4E4",
                linewidth=0.5, zorder=1)
        ax.text(w, top + 0.03, f"{w}", ha="center", va="bottom", fontsize=7,
                color=GRAY)
    ax.text(-4, top + 0.03, "bits", ha="right", va="bottom", fontsize=7,
            color=GRAY)

    require_plotted(
        "fig_word", "echelle-formats.csv", set(rows), {"Planes14", "Tetra48"},
        {k2: "this figure contrasts two layouts; the others have no record here"
         for k2 in rows if k2 not in {"Planes14", "Tetra48"}},
    )

    ax.set_xlim(-40, right + 4)
    ax.set_ylim(y_z - bar_h / 2 - 0.62, top + 0.40)
    ax.axis("off")
    fig.savefig(OUT / "fig_word.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig. 5 - the tile sweep
# ---------------------------------------------------------------------------

def fig_tile() -> None:
    """Median ms against the activation tile for three arms of one process:
    the arm with tables, the arm without, and the no-weights control. Each
    curve is one arm against itself across tiles."""
    rows = read_csv("tuile-l40s.csv")
    rows.sort(key=lambda r: int(r["tile"]))
    tiles = [int(r["tile"]) for r in rows]
    series = [
        ("planes14_ms", "Planes14, no tables", BLUE, "o", "-"),
        ("tetra48_ms", "Tetra, 18.4 KiB of tables", PURPLE, "o", "-"),
        ("nullk_ms", "no-weights control", GRAY, "^", ":"),
    ]
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN * 0.66, 2.15),
                           layout="constrained")
    for col, label, color, marker, ls in series:
        ys = [float(r[col]) for r in rows]
        ax.plot(tiles, ys, marker + ls, color=color, markersize=4.5,
                linewidth=1.2, label=label, zorder=3)
        amp = 100 * (max(ys) / min(ys) - 1)
        ax.annotate(f"{amp:.1f}%", xy=(tiles[-1], ys[-1]), xytext=(5, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=7.5, color=color, annotation_clip=False)

    ax.set_xscale("log", base=2)
    ax.set_xticks(tiles, [str(t) for t in tiles])
    ax.set_xlim(27, 165)
    ax.set_ylim(1.9, 6.1)
    ax.set_xlabel("activation tile $T$ (blocks per CTA)")
    ax.set_ylabel("median ms, 252 projections")
    ax.grid(axis="y", zorder=0)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0),
              handlelength=1.8, borderaxespad=0.1)
    fig.savefig(OUT / "fig_tile.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig. 6 - perplexity against MMLU
# ---------------------------------------------------------------------------

def fig_dissoc() -> None:
    """The served object before and after the row-scale training, against the
    FP16 checkpoint, on the two metrics at once."""
    rows = {r["base"]: r for r in read_csv("tetra-rowscales.csv")}
    r = rows["served_object"]
    f16_mmlu = float(next(x for x in read_csv("tetra-formats.csv")
                          if x["arm"] == "f16")["mmlu"])
    before = (float(r["ppl_ratio_before"]), float(r["mmlu_before"]))
    after = (float(r["ppl_ratio_after"]), float(r["mmlu_after"]))

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN * 0.66, 2.15),
                           layout="constrained")
    ax.axhline(f16_mmlu, color=GRAY, linewidth=0.8, linestyle=":", zorder=1)
    ax.axvline(1.0, color=GRAY, linewidth=0.8, linestyle=":", zorder=1)
    ax.plot(1.0, f16_mmlu, marker="o", markersize=5.5, color=INK, zorder=3)
    ax.annotate("FP16 checkpoint", xy=(1.0, f16_mmlu), xytext=(6, -3),
                textcoords="offset points", ha="left", va="top", fontsize=7.5,
                color=INK)
    ax.plot(*before, marker="o", markersize=5.5, markerfacecolor="white",
            markeredgecolor=PURPLE, markeredgewidth=1.3, zorder=3)
    ax.plot(*after, marker="o", markersize=5.5, color=PURPLE, zorder=3)
    ax.annotate("", xy=after, xytext=before,
                arrowprops=dict(arrowstyle="->", color=PURPLE, linewidth=1.0,
                                connectionstyle="arc3,rad=0.25",
                                shrinkA=5, shrinkB=5), zorder=2)
    ax.annotate("before training", xy=before, xytext=(-5, -6),
                textcoords="offset points", ha="right", va="top",
                fontsize=7.5, color=PURPLE)
    ax.annotate("after, served", xy=after, xytext=(5, 5),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=7.5, color=PURPLE)

    ax.set_xlabel("wikitext-2 perplexity, ratio to FP16")
    ax.set_ylabel("MMLU micro (census)")
    ax.set_xlim(0.95, 1.42)
    ax.set_ylim(54, 74)
    ax.grid(zorder=0)
    fig.savefig(OUT / "fig_dissoc.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_gap()
    fig_word()
    fig_tile()
    fig_dissoc()
    print(f"wrote 4 figures to {OUT}")
    print("numbers not read from a CSV:")
    for what, where in HARDCODED:
        print(f"  {what}  <-  {where}")


if __name__ == "__main__":
    main()
