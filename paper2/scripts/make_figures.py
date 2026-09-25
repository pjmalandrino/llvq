#!/usr/bin/env python3
"""Generate paper 2's figures from docs/data/*.csv.

The art direction is paper 1's (`paper/scripts/make_figures.py`), copied rather
than reinterpreted: Okabe-Ito palette, serif type at 7-8 pt, grey axes without
top and right spines, a light grid, figures drawn at the 5.5 in text width they
are printed at. Our arms are circles in the cool colours, blue for the object
of the paper; deployed kernels are squares in the warm ones. The scale figure
is paper 1's `fig_scale` with the same three series styles.

No number in a figure is typed by hand except the layout constants listed in
`HARDCODED`, and every CSV row a figure claims to draw is drawn or the script
fails with the row named.

  fig_gap.pdf     bits carried vs bits read in VRAM     echelle-formats.csv
  fig_word.pdf    two records to bit scale, one zoom    (layout constants)
  fig_tile.pdf    the tile sweep, three arms            tuile-l40s.csv
  fig_scale.pdf   three sizes, three panels             paper2-gaps.csv,
                                                         paper2-chain.csv,
                                                         paper2-results.csv,
                                                         echelle-4b-8b.csv
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# The text block is 5.5 in wide, as in paper 1. Figures are drawn at the width
# they are included at, so 7-8 pt type prints at 7-8 pt.
TEXTWIDTH_IN = 5.5

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "data"
OUT = ROOT / "paper2" / "figures"

# Okabe-Ito palette, identical to paper 1.
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
                    excluded: dict) -> None:
    """Fail if a CSV row the figure should draw was not drawn."""
    missing = expected - plotted - set(excluded)
    if missing:
        raise SystemExit(
            f"{fig}: {sorted(missing)} in {csv_name} but not plotted: "
            "add them to the figure or list them as excluded with a reason"
        )
    for key, why in excluded.items():
        print(f"{fig}: {csv_name} row {key!r} not drawn ({why})")


# ---------------------------------------------------------------------------
# What the format carries against what the kernel reads
# ---------------------------------------------------------------------------

# Bits of code a format carries per weight. A property of the format, not a
# measurement: 48 bits per block of 24 weights for the two lattice formats (a
# 47-bit index plus one gain bit), 2 bits per weight for the 2-bit trellis,
# and 4 bits per weight for a 4-bit affine format before its group scales.
CODE_CARRIED = {"Planes14": 2.000, "Tetra48": 2.000, "QTIP": 2.000,
                "AWQ": 4.000}


def fig_gap() -> None:
    """One dumbbell per format: bits of code carried (hollow) joined to bits
    read per weight in VRAM (filled). The x axis starts at 1.7 so that the two
    short dumbbells stay legible segments."""
    rows = {r["layout"]: r for r in read_csv("echelle-formats.csv")}
    hardcoded("bits of code carried, 2.000 / 4.000 (fig_gap)",
              "format definitions: 48 bits per 24 weights; 4-bit affine")
    shown = {"AWQ": "AWQ w4g128 (4-bit)", "Planes14": "Planes14 (our earlier layout)",
             "Tetra48": "Tetra (this paper)", "QTIP": "QTIP (2-bit)"}
    colors = {"AWQ": ORANGE, "Planes14": SKY, "Tetra48": BLUE,
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
        {"FP16": "16.000 on both ends, no gap to draw",
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
# Two records, to the bit, plus the short one magnified
# ---------------------------------------------------------------------------

# Field widths are constants of each layout, transcribed from the record
# diagram at the head of its decoder, exactly as in paper 1's record figure:
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
    """The unfolded Planes14 record and the Tetra word at one bit scale, then
    the word magnified so that every field can be named. Everything that
    would need a sentence is in the caption."""
    rows = {r["layout"]: r for r in read_csv("echelle-formats.csv")}
    hardcoded("record field widths (fig_word)",
              "llvq-cuda/kernels/llvq_planes.cuh, llvq_f1rank.cuh headers")

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.45), layout="constrained")
    bar_h = 0.52
    right = 196          # where the value column sits, in bit units
    y_p, y_t, y_z = 0.0, -1.15, -2.62
    zoom_w = 176.0       # the magnified word spans this many bit units
    k = zoom_w / 48.0

    def draw(record, y, scale=1.0, label_min=4.0):
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

    ax.plot([0, 0], [y_t - bar_h / 2, y_z + bar_h / 2], color=GRAY,
            linewidth=0.5, linestyle=":", zorder=1)
    ax.plot([48, zoom_w], [y_t - bar_h / 2, y_z + bar_h / 2], color=GRAY,
            linewidth=0.5, linestyle=":", zorder=1)
    draw(TETRA_RECORD, y_z, scale=k, label_min=9.0)
    ax.text(-4, y_z, "the word,\nmagnified", ha="right", va="center",
            fontsize=8, color=INK, linespacing=1.15)
    ax.text(0, y_z - bar_h / 2 - 0.20,
            "one bit each, left to right:  $p$ shared parity · $r$ section-1 "
            "class · $b_1$, $b_3$ edge choices · $g$ gain",
            ha="left", va="top", fontsize=7.5, color=INK)

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
        {k2: "this figure contrasts two layouts" for k2 in rows
         if k2 not in {"Planes14", "Tetra48"}},
    )

    ax.set_xlim(-40, right + 4)
    ax.set_ylim(y_z - bar_h / 2 - 0.62, top + 0.40)
    ax.axis("off")
    fig.savefig(OUT / "fig_word.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# The tile sweep
# ---------------------------------------------------------------------------

def fig_tile() -> None:
    """Median ms against the activation tile for three arms of one process:
    the arm with tables, the arm without, and the no-weights control."""
    rows = read_csv("tuile-l40s.csv")
    rows.sort(key=lambda r: int(r["tile"]))
    tiles = [int(r["tile"]) for r in rows]
    series = [
        ("planes14_ms", "Planes14, no tables", SKY, "o", "-"),
        ("tetra48_ms", "Tetra, 18.4 KiB of tables", BLUE, "o", "-"),
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
    ax.set_ylim(1.9, 7.4)
    ax.set_xlabel("activation tile $T$ (blocks per CTA)")
    ax.set_ylabel("median ms, 252 projections")
    ax.grid(axis="y", zorder=0)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0),
              handlelength=1.8, borderaxespad=0.1)
    fig.savefig(OUT / "fig_tile.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Three sizes, three panels (paper 1's fig_scale, same series styles)
# ---------------------------------------------------------------------------

def fig_scale() -> None:
    """Against model size: the paired MMLU gaps, whole-model bits per
    parameter, and decode speed at batch 1, each engine in its own series."""
    gaps = read_csv("paper2-gaps.csv")
    chain = read_csv("paper2-chain.csv")
    results = read_csv("paper2-results.csv")
    scale = read_csv("echelle-4b-8b.csv")

    sizes = {r["model"]: int(r["params_total"]) / 1e9 for r in scale}
    models = sorted(sizes, key=sizes.get)
    xs = [sizes[m] for m in models]
    short = {m: m.replace("Qwen3-", "") for m in models}

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 1.95),
                                     layout="constrained")

    def line(ax, pts, color, label, marker="o", ls="-"):
        """pts: (x, y, lo, hi); lo == hi == y draws no bar."""
        pts = sorted(pts)
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts],
                    yerr=[[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]],
                    fmt=marker + ls, color=color, markersize=4, linewidth=1.1,
                    capsize=2.5, elinewidth=0.9, capthick=0.9, label=label,
                    zorder=3)

    # (i) MMLU gap in points, paired 95% CI, zero line.
    done_gap = set()
    a1.axhline(0, color=GRAY, linewidth=0.7, linestyle=":", zorder=1)
    for pair, color, label, dx, marker, ls in (
            ("f16_minus_tetra", BLUE, "FP16 − Tetra", 0, "o", "-"),
            ("awq4_minus_tetra", VERMILLION, "AWQ − Tetra", 0.18, "s", "--"),
            ("f16_minus_awq4", GREEN, "FP16 − AWQ", -0.18, "^", ":")):
        rows = [r for r in gaps if r["pair"] == pair]
        line(a1, [(sizes[r["model"]] + dx, float(r["delta_pp"]),
                   float(r["ci_lo_pp"]), float(r["ci_hi_pp"])) for r in rows],
             color, label, marker, ls)
        done_gap |= {(r["model"], r["pair"]) for r in rows}
    a1.set_ylabel("MMLU gap (points)")
    a1.set_ylim(-1, 9)
    a1.legend(frameon=False, loc="upper right")

    # (ii) whole-model bits per parameter, served file against AWQ.
    tetra_bpp = {r["model"]: float(r["bparam"]) for r in chain
                 if r["stage"] == "sealed"}
    awq_bpp = {r["model"]: float(r["vram_bits_per_param"]) for r in scale
               if r["arm"] == "awq4"}
    line(a2, [(sizes[m], v, v, v) for m, v in tetra_bpp.items()], BLUE,
         "Tetra, served (ours)")
    line(a2, [(sizes[m], v, v, v) for m, v in awq_bpp.items()], GREEN,
         "4-bit AWQ, official", marker="s", ls="--")
    for m, v in tetra_bpp.items():
        a2.annotate(f"{v:.3f}", xy=(sizes[m], v), xytext=(0, -9),
                    textcoords="offset points", ha="center", fontsize=6.5,
                    color=BLUE)
    for m, v in awq_bpp.items():
        a2.annotate(f"{v:.3f}", xy=(sizes[m], v), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=6.5,
                    color=GREEN)
    a2.set_ylabel("b/param, whole model")
    a2.set_ylim(2.0, 6.9)
    a2.legend(frameon=False, loc="center right")

    # (iii) decode tokens per second at batch 1, each engine its own series.
    done_speed = set()
    pending = {}
    for arm, engine, color, label, marker, ls in (
            ("tetra", "ours", BLUE, "Tetra (ours)", "o", "-"),
            ("awq", "vLLM 0.26.0", GREEN, "AWQ (vLLM)", "s", "--"),
            ("fp16", "vLLM 0.26.0", GRAY, "FP16 (vLLM)", "^", ":")):
        rows = [r for r in results if r["arm"] == arm and r["engine"] == engine]
        have = [r for r in rows if r["toks"]]
        for r in rows:
            if not r["toks"]:
                pending[(r["model"], arm, engine)] = "served run not back yet"
        if have:
            line(a3, [(sizes[r["model"]], float(r["toks"]), float(r["toks"]),
                       float(r["toks"])) for r in have], color, label, marker, ls)
        done_speed |= {(r["model"], r["arm"], r["engine"]) for r in have}
    a3.set_ylabel("decode tok/s, batch 1")
    a3.set_ylim(0, 230)
    a3.legend(frameon=False, loc="upper right")

    for ax in (a1, a2, a3):
        ax.set_xticks(xs, [short[m] for m in models])
        ax.set_xlim(min(xs) - 1.2, max(xs) + 1.2)
        ax.set_xlabel("model size")
        ax.grid(axis="y", zorder=0)

    require_plotted("fig_scale", "paper2-gaps.csv",
                    {(r["model"], r["pair"]) for r in gaps}, done_gap, {})
    require_plotted(
        "fig_scale", "paper2-results.csv",
        {(r["model"], r["arm"], r["engine"]) for r in results}, done_speed,
        {**pending,
         **{(r["model"], r["arm"], r["engine"]): "not in the speed panel"
            for r in results if r["arm"] == "iq2xxs"
            or (r["arm"] == "fp16" and r["engine"] == "ours dense")}})
    fig.savefig(OUT / "fig_scale.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_gap()
    fig_word()
    fig_tile()
    fig_scale()
    print(f"wrote 4 figures to {OUT}")
    print("numbers not read from a CSV:")
    for what, where in HARDCODED:
        print(f"  {what}  <-  {where}")


if __name__ == "__main__":
    main()
