#!/usr/bin/env python3
"""Generate paper 2's figures from docs/data/*.csv.

The art direction is paper 1's (`paper/scripts/make_figures.py`), copied rather
than reinterpreted: Okabe-Ito palette, serif type at 7-8 pt, grey axes without
top and right spines, a light grid, figures drawn at the 5.5 in text width they
are printed at. Our arms are circles in the cool colours, blue for the object
of the paper; deployed kernels are squares in the warm ones.

No number in a figure is typed by hand except the layout constants listed in
`HARDCODED`, and every CSV row a figure claims to draw is drawn or the script
fails with the row named.

  fig_gap.pdf     bits carried vs bits read in VRAM     echelle-formats.csv
  fig_word.pdf    two records to bit scale, one zoom    (layout constants)
  fig_frontier.pdf  MMLU and GSM8K against weight GB    paper2-results.csv, paper2-gsm8k.csv
  fig_speed.pdf     decode speed at batch 1             paper2-results.csv
  fig_context.pdf   decode speed and first token        paper2-context.csv, paper2-duel.csv
  fig_peak.pdf      peak card memory, 5 and 8,192       paper2-peak.csv
  fig_ablation.pdf  where the 4B loses its MMLU points  paper2-ablation.csv, paper2-gaps.csv
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
             "Tetra48": "Tetra (this paper)", "QTIP": "QTIP HYB (2-bit)"}
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
    ("class", 9, "class"), ("g", 1, "gain"), ("signs", 24, "smask"),
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
    end_t = draw(TETRA_RECORD, y_t, label_min=5.0)
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
# The files against the baselines, speed, long prompts
# ---------------------------------------------------------------------------

SIZES = ["Qwen3-4B", "Qwen3-8B", "Qwen3-14B"]
SHORT = {"Qwen3-4B": "4B", "Qwen3-8B": "8B", "Qwen3-14B": "14B"}
# One colour and one marker per arm, in every figure below. Circles are ours.
ARM_STYLE = {
    "fp16": (GRAY, "s", "FP16, vLLM"),
    "awq": (ORANGE, "s", "AWQ 4-bit, vLLM"),
    "iq2s": (VERMILLION, "s", "IQ2_S, llama.cpp"),
    "tetra": (BLUE, "o", "Tetra (this paper)"),
}


def fig_frontier() -> None:
    """Quality against weight memory, one line per format through its three
    sizes. The x axis is logarithmic: the formats span 1.4 to 29.5 GB."""
    res = read_csv("paper2-results.csv")
    gsm = {(r["model"], r["arm"]): float(r["gsm8k"])
           for r in read_csv("paper2-gsm8k.csv") if "dense" not in r["engine"]}
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.35),
                             layout="constrained")
    plotted = set()
    for ax, metric in zip(axes, ("MMLU", "GSM8K")):
        for arm, (color, marker, label) in ARM_STYLE.items():
            xs, ys = [], []
            for m in SIZES:
                row = next(r for r in res if r["model"] == m and r["arm"] == arm
                           and r["engine"] != "ours dense")
                xs.append(float(row["gb"]))
                ys.append(float(row["mmlu"]) if metric == "MMLU"
                          else gsm[(m, arm)])
                plotted.add((m, arm))
            ours = arm == "tetra"
            ax.plot(xs, ys, color=color, marker=marker, markersize=5.5 if ours else 4.5,
                    linewidth=1.8 if ours else 1.1, label=label,
                    zorder=4 if ours else 3)
            if ours:
                for x, y, m in zip(xs, ys, SIZES):
                    ax.annotate(SHORT[m], xy=(x, y), xytext=(0, 6),
                                textcoords="offset points", ha="center",
                                va="bottom", fontsize=7, color=BLUE,
                                fontweight="bold")
        ax.set_xscale("log")
        ax.set_xticks([1.5, 3, 6, 12, 24], ["1.5", "3", "6", "12", "24"])
        ax.set_xticks([], minor=True)
        ax.set_xlim(1.15, 36)
        ax.set_xlabel("weights, GB (log scale)")
        ax.set_ylabel(f"{metric}, %")
        ax.grid(zorder=0)
    axes[0].legend(loc="lower right", frameon=False, handlelength=1.6)
    expected = {(r["model"], r["arm"]) for r in res}
    require_plotted(
        "fig_frontier", "paper2-results.csv", expected, plotted,
        {**{(m, "fp16"): "drawn once, from the vLLM row" for m in ()},
         ("Qwen3-4B", "iq2xxs"): "39.78 MMLU, below the axis; in the table",
         **{(m, "tetra_permatrix"): "a speed row, same file" for m in SIZES}},
    )
    fig.savefig(OUT / "fig_frontier.pdf")
    plt.close(fig)


def fig_speed() -> None:
    """Decode speed at batch 1 after a short prompt, one group of bars a size."""
    res = read_csv("paper2-results.csv")
    arms = [("fp16", "vLLM 0.26.0", GRAY, "FP16, vLLM"),
            ("awq", "vLLM 0.26.0", ORANGE, "AWQ 4-bit, vLLM"),
            ("iq2s", None, VERMILLION, "IQ2_S, llama.cpp"),
            ("tetra_permatrix", None, SKY, "Tetra, one launch per matrix"),
            ("tetra", None, BLUE, "Tetra, one launch per token (served)")]
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.9), layout="constrained")
    h, gap = 0.8, 1.3
    y, yticks, plotted = 0.0, [], set()
    for m in SIZES:
        top = y
        for arm, engine, color, _ in arms:
            row = next(r for r in res if r["model"] == m and r["arm"] == arm
                       and (engine is None or r["engine"] == engine))
            v = float(row["toks"])
            ax.barh(y, v, height=h, color=color, zorder=3)
            ax.text(v + 3, y, f"{v:.1f}", ha="left", va="center", fontsize=7,
                    color=INK, fontweight="bold" if arm == "tetra" else "normal")
            plotted.add((m, arm, row["engine"]))
            y -= 1.0
        yticks.append(((top + y + 1.0) / 2, SHORT[m]))
        y -= gap
    for arm, _, color, label in arms:
        ax.barh(0, 0, color=color, label=label)
    expected = {(r["model"], r["arm"], r["engine"]) for r in res}
    require_plotted(
        "fig_speed", "paper2-results.csv", expected, plotted,
        {**{(m, "fp16", "ours dense"): "the dense reconstruction; in the table"
            for m in SIZES},
         ("Qwen3-4B", "iq2xxs", "llama.cpp"): "a lower rate; in the table"},
    )
    ax.set_yticks([t for t, _ in yticks], [s for _, s in yticks])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("tokens per second, batch 1, one L40S")
    ax.set_xlim(0, 335)
    ax.grid(axis="x", zorder=0)
    ax.legend(loc="lower right", frameon=False, handlelength=1.2)
    fig.savefig(OUT / "fig_speed.pdf")
    plt.close(fig)


def fig_context() -> None:
    """Left: decode speed after a prompt of 5 to 8,192 tokens. Right: seconds
    to the first token, the eager prefill against the chain prefill at 4B."""
    ctx = read_csv("paper2-context.csv")
    duel = read_csv("paper2-duel.csv")
    shade = {"Qwen3-4B": BLUE, "Qwen3-8B": "#3F93C6", "Qwen3-14B": SKY}
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.75),
                               layout="constrained")
    plotted = set()
    for m in SIZES:
        rows = [r for r in ctx if r["model"] == m and r["config"] == "v9"]
        xs = [int(r["prompt"]) for r in rows]
        a.plot(xs, [float(r["toks"]) for r in rows], color=shade[m],
               marker="o", markersize=4.5, linewidth=1.5, zorder=4)
        a.annotate(f"Tetra {SHORT[m]}", xy=(xs[-1], float(rows[-1]["toks"])),
                   xytext=(4, {"Qwen3-4B": 6, "Qwen3-8B": 6, "Qwen3-14B": 0}[m]),
                   textcoords="offset points", ha="left",
                   va="center", fontsize=7, color=shade[m], fontweight="bold")
        d = next(r for r in duel if r["model"] == m and r["arm"] == "iq2s")
        a.plot([5, 8192], [float(d["toks5"]), float(d["toks8192"])],
               color=VERMILLION, marker="s", markersize=4, linewidth=1.0,
               linestyle=(0, (3, 2)), zorder=3)
        a.annotate(f"IQ2_S {SHORT[m]}", xy=(8192, float(d["toks8192"])),
                   xytext=(4, {"Qwen3-4B": 2, "Qwen3-8B": -6, "Qwen3-14B": -6}[m]),
                   textcoords="offset points", ha="left",
                   va="center", fontsize=7, color=VERMILLION)
        b.plot(xs[1:], [float(r["first_token_s"]) for r in rows[1:]],
               color=shade[m], marker="o", markersize=4.5, linewidth=1.5,
               zorder=3)
        b.annotate(SHORT[m],
                   xy=(xs[-1], float(rows[-1]["first_token_s"])),
                   xytext=(5, {"Qwen3-4B": -3, "Qwen3-8B": 2, "Qwen3-14B": 0}[m]),
                   textcoords="offset points", ha="left",
                   va="center", fontsize=7, color=shade[m], fontweight="bold")
        plotted |= {(m, "v9", r["prompt"]) for r in rows}
    ch = [r for r in ctx if r["config"] == "chain" and r["prompt"] != "5"]
    b.plot([int(r["prompt"]) for r in ch], [float(r["first_token_s"]) for r in ch],
           color=GREEN, marker="o", markersize=4.5, linewidth=1.8, zorder=4)
    b.annotate("4B", xy=(8192, float(ch[-1]["first_token_s"])),
               xytext=(5, 0), textcoords="offset points", ha="left",
               va="center", fontsize=7, color=GREEN, fontweight="bold")
    b.text(1100, 60, "per-matrix prefill", ha="left", va="center", fontsize=7,
           color=BLUE)
    b.text(2300, 0.5, "chain prefill", ha="left", va="center", fontsize=7,
           color=GREEN)
    plotted |= {(r["model"], "chain", r["prompt"]) for r in ch}
    expected = {(r["model"], r["config"], r["prompt"]) for r in ctx}
    require_plotted(
        "fig_context", "paper2-context.csv", expected, plotted,
        {**{k: "the v9 arm of the chain job; quoted in the text"
            for k in expected if k[1] == "v9_same_job"},
         ("Qwen3-4B", "chain", "5"): "0.02 s, as the eager prefill"},
    )
    a.set_xlim(-300, 11400)
    a.set_ylim(0, 310)
    a.set_xticks([5, 2048, 4096, 8192], ["5", "2,048", "4,096", "8,192"])
    a.set_xlabel("prompt length, tokens")
    a.set_ylabel("decode speed, tokens per second")
    a.grid(zorder=0)
    b.set_yscale("log")
    b.set_yticks([0.5, 1, 2, 5, 10, 20, 50, 100],
                 ["0.5", "1", "2", "5", "10", "20", "50", "100"])
    b.set_yticks([], minor=True)
    b.set_ylim(0.3, 130)
    b.set_xscale("log", base=2)
    b.set_xlim(850, 12500)
    b.set_xticks([1024, 2048, 4096, 8192], ["1,024", "2,048", "4,096", "8,192"])
    b.set_xticks([], minor=True)
    b.set_xlabel("prompt length, tokens")
    b.set_ylabel("seconds to the first token (log scale)")
    b.grid(zorder=0)
    fig.savefig(OUT / "fig_context.pdf")
    plt.close(fig)


def fig_peak() -> None:
    """Peak card memory after a prompt of 5 and of 8,192 tokens."""
    rows = read_csv("paper2-peak.csv")
    val = {(r["model"], r["arm"], r["prompt"]): float(r["peak_gb"]) for r in rows}
    series = [("iq2s", "5", VERMILLION, 0.45, "IQ2_S, after 5 tokens"),
              ("iq2s", "8192", VERMILLION, 1.0, "IQ2_S, after 8,192"),
              ("tetra", "5", BLUE, 0.45, "Tetra, after 5 tokens"),
              ("tetra", "8192", BLUE, 1.0, "Tetra, after 8,192"),
              ("tetra_chain", "8192", GREEN, 1.0, "Tetra, chain prefill, after 8,192")]
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.1), layout="constrained")
    w, plotted = 0.17, set()
    for i, m in enumerate(SIZES):
        present = [s for s in series if (m, s[0], s[1]) in val]
        x0 = i - w * (len(present) - 1) / 2
        for j, (arm, prompt, color, alpha, _) in enumerate(present):
            v = val[(m, arm, prompt)]
            x = x0 + j * w
            ax.bar(x, v, width=w * 0.9, color=color, alpha=alpha, zorder=3)
            ax.text(x, v + 0.25, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=6.5, color=INK)
            plotted.add((m, arm, prompt))
            if arm == "tetra" and prompt == "8192":
                before = val[(m, "tetra_before", "8192")]
                ax.bar(x, before, width=w * 0.9, facecolor="none",
                       edgecolor=BLUE, linewidth=0.8, linestyle=(0, (2, 2)),
                       zorder=2)
                ax.text(x, before + 0.25, f"{before:.2f}", ha="center",
                        va="bottom", fontsize=6.5, color=GRAY)
                plotted.add((m, "tetra_before", "8192"))
    for arm, prompt, color, alpha, label in series:
        ax.bar(0, 0, color=color, alpha=alpha, label=label)
    ax.bar(0, 0, facecolor="none", edgecolor=BLUE, linewidth=0.8,
           linestyle=(0, (2, 2)), label="Tetra, after 8,192, before the fix")
    require_plotted("fig_peak", "paper2-peak.csv", set(val), plotted, {})
    ax.set_xticks(range(len(SIZES)), [SHORT[m] for m in SIZES])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("peak card memory, GB")
    ax.set_ylim(0, 21.5)
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="upper left", frameon=False, ncols=2, handlelength=1.2,
              columnspacing=1.2)
    fig.savefig(OUT / "fig_peak.pdf")
    plt.close(fig)


def fig_ablation() -> None:
    """MMLU points the sealed 4B recovers when one group of projections is
    restored to f16, with 95 % intervals, against its whole gap to FP16."""
    rows = read_csv("paper2-ablation.csv")
    gap = next(float(r["delta_pp"]) for r in read_csv("paper2-gaps.csv")
               if r["model"] == "Qwen3-4B" and r["pair"] == "f16_minus_tetra")
    order = ["qkv", "o", "attn", "gateup", "down", "mlp", "all"]
    color = {"qkv": SKY, "o": SKY, "attn": SKY, "gateup": BLUE, "down": BLUE,
             "mlp": BLUE, "all": INK}
    by = {r["group"]: r for r in rows}
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.0), layout="constrained")
    plotted = set()
    for i, g in enumerate(order):
        r = by[g]
        y = len(order) - 1 - i
        d, lo, hi = (float(r[k]) for k in ("delta_pp", "ci_lo_pp", "ci_hi_pp"))
        ax.barh(y, d, height=0.62, color=color[g], zorder=3)
        ax.plot([lo, hi], [y, y], color=INK if g != "all" else GRAY,
                linewidth=1.0, zorder=4)
        for e in (lo, hi):
            ax.plot([e, e], [y - 0.16, y + 0.16],
                    color=INK if g != "all" else GRAY, linewidth=1.0, zorder=4)
        ax.text(max(hi, 0) + 0.12, y, f"{d:+.2f}  [{lo:.2f}, {hi:.2f}]",
                ha="left", va="center", fontsize=7, color=INK)
        plotted.add(g)
    require_plotted("fig_ablation", "paper2-ablation.csv", set(by), plotted, {})
    ax.axvline(gap, color=VERMILLION, linewidth=1.0, linestyle=(0, (3, 2)),
               zorder=2)
    ax.text(gap + 0.08, len(order) - 1.0, f"gap to FP16, {gap:.2f}", ha="left",
            va="center", fontsize=7, color=VERMILLION)
    ax.axvline(0, color=GRAY, linewidth=0.7, zorder=2)
    ax.set_yticks([len(order) - 1 - i for i in range(len(order))],
                  [by[g]["label"] for g in order])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("MMLU points recovered when the group is restored to f16")
    ax.set_xlim(-0.8, 9.6)
    ax.grid(axis="x", zorder=0)
    fig.savefig(OUT / "fig_ablation.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_gap()
    fig_word()
    fig_frontier()
    fig_speed()
    fig_context()
    fig_peak()
    fig_ablation()
    print(f"wrote 7 figures to {OUT}")
    print("numbers not read from a CSV:")
    for what, where in HARDCODED:
        print(f"  {what}  <-  {where}")


if __name__ == "__main__":
    main()
