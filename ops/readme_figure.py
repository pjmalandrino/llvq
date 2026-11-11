#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8"]
# ///
"""The README's figure: MMLU against file size, Qwen3 4B, 8B and 14B.

Reads `docs/data/readme-frontier.csv` and writes `docs/figures/frontier-light.svg`
and `docs/figures/frontier-dark.svg`. The README shows them through a <picture>
element, so GitHub picks the one that matches the reader's theme.

File size is bits per parameter x parameters / 8: the whole file, embedding and
tables included, the accounting of the duel of 2026-11-08. No number in the
figure is typed here; every point comes from the CSV, whose `source` column
names the journal.

The hues are paper 2's Okabe-Ito blue and orange, with reddish purple for
llama.cpp, stepped for GitHub's dark surface. Both sets pass the colour-vision
checks on every pair; the light orange sits under 3:1 on white, so every series
also carries its own marker and a direct label.

    uv run ops/readme_figure.py
    LLVQ_FIG_PREVIEW=/some/dir uv run ops/readme_figure.py   # also opaque PNG previews
"""

import csv
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data" / "readme-frontier.csv"
OUT = ROOT / "docs" / "figures"

ARMS = ["fp16", "awq", "iq2s", "tetra"]  # drawing order: ours on top
SIZES = ["Qwen3-4B", "Qwen3-8B", "Qwen3-14B"]

LEGEND = {
    "tetra": "Tetra, this repository (2.7–2.8 b/param)",
    "awq": "AWQ w4g128, vLLM (5.3–6.0 b/param)",
    "iq2s": "IQ2_S, llama.cpp (2.7–2.8 b/param)",
    "fp16": "FP16, vLLM (16 b/param)",
}
SHORT = {"tetra": "Tetra 14B", "awq": "AWQ 4-bit", "iq2s": "IQ2_S", "fp16": "FP16"}
MARKER = {"tetra": "o", "awq": "s", "iq2s": "D", "fp16": "^"}

THEMES = {
    "light": {
        "surface": "#ffffff", "ink": "#1f2328", "muted": "#59636e", "grid": "#d8dee4",
        "tetra": "#0072B2", "awq": "#E69F00", "iq2s": "#CC79A7", "fp16": "#7f7f7f",
    },
    "dark": {
        "surface": "#0d1117", "ink": "#e6edf3", "muted": "#9198a1", "grid": "#30363d",
        "tetra": "#3987e5", "awq": "#c98500", "iq2s": "#d55181", "fp16": "#8b949e",
    },
}

# Where each direct label sits, relative to the arm's 14B point, in points.
LABEL_AT = {
    "tetra": ("Qwen3-14B", (-9, 9), "right"),
    "awq": ("Qwen3-14B", (10, -4), "left"),
    "iq2s": ("Qwen3-8B", (14, -16), "left"),
    "fp16": ("Qwen3-14B", (0, 11), "center"),
}


def load() -> dict[str, dict[str, tuple[float, float]]]:
    points: dict[str, dict[str, tuple[float, float]]] = {a: {} for a in ARMS}
    with open(DATA, newline="") as f:
        for row in csv.DictReader(f):
            gb = float(row["bparam"]) * int(row["params"]) / 8 / 1e9
            points[row["arm"]][row["model"]] = (gb, float(row["mmlu"]))
    for arm in ARMS:
        missing = [s for s in SIZES if s not in points[arm]]
        if missing:
            raise SystemExit(f"{DATA.name}: {arm} has no row for {missing}")
    return points


def draw(points, theme: str) -> Path:
    c = THEMES[theme]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "svg.fonttype": "path",
        "text.color": c["ink"],
        "axes.labelcolor": c["ink"],
        "axes.edgecolor": c["muted"],
        "xtick.color": c["muted"],
        "ytick.color": c["muted"],
    })
    fig, ax = plt.subplots(figsize=(8.0, 5.1))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    fig.subplots_adjust(left=0.085, right=0.975, top=0.81, bottom=0.20)

    for arm in ARMS:
        xs = [points[arm][s][0] for s in SIZES]
        ys = [points[arm][s][1] for s in SIZES]
        ours = arm == "tetra"
        ax.plot(xs, ys, color=c[arm], lw=2.8 if ours else 2.0, zorder=4 if ours else 3,
                solid_capstyle="round")
        ax.plot(xs, ys, ls="none", marker=MARKER[arm], ms=9 if ours else 8,
                mfc=c[arm], mec=c["surface"], mew=2.0, zorder=5 if ours else 4,
                label=LEGEND[arm])
        size, (dx, dy), ha = LABEL_AT[arm]
        x, y = points[arm][size]
        ax.annotate(SHORT[arm], (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", fontsize=11, color=c["ink"],
                    fontweight="bold" if ours else "normal", zorder=6)

    # Model size beside our 4B and 8B (the 14B carries the series label); every
    # line runs 4B, 8B, 14B from left to right.
    for size, (dx, dy), ha in [("Qwen3-4B", (0, -15), "center"),
                               ("Qwen3-8B", (0, 13), "center")]:
        x, y = points["tetra"][size]
        ax.annotate(size.removeprefix("Qwen3-"), (x, y), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="center", fontsize=9.5,
                    color=c["muted"], zorder=6)

    ax.set_xscale("log")
    ax.set_xlim(1.1, 40)
    ax.xaxis.set_major_locator(FixedLocator([1.25, 2.5, 5, 10, 20, 40]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["1.25", "2.5", "5", "10", "20", "40"])
    lo = 5 * math.floor(min(p[1] for a in ARMS for p in points[a].values()) / 5)
    ax.set_ylim(lo, 82)
    ax.set_xlabel("File size, GB (log scale)", color=c["muted"], fontsize=10.5, labelpad=6)
    ax.set_ylabel("MMLU, 5-shot", color=c["muted"], fontsize=10.5, labelpad=6)
    ax.grid(axis="y", color=c["grid"], lw=0.8)
    ax.grid(axis="x", color=c["grid"], lw=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["grid"])
    ax.tick_params(length=0, pad=6)

    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles[::-1], labels[::-1], loc="lower right", frameon=False,
                    fontsize=9.5, handlelength=1.2, borderaxespad=0.4, labelspacing=0.45)
    for text in leg.get_texts():
        text.set_color(c["ink"])

    fig.text(0.085, 0.935, "Qwen3: more model per gigabyte", fontsize=16, fontweight="bold",
             color=c["ink"], ha="left")
    fig.text(0.085, 0.875,
             "At about the same size, Tetra's 8B scores above AWQ's 4B, "
             "and its 14B above AWQ's 8B.",
             fontsize=10.5, color=c["muted"], ha="left")
    fig.text(0.085, 0.048,
             "MMLU measured on all 14,042 test questions. File size computed as bits per "
             "parameter × parameters, whole file.",
             fontsize=8.5, color=c["muted"], ha="left")
    fig.text(0.085, 0.016, "Data and journals: docs/data/readme-frontier.csv.",
             fontsize=8.5, color=c["muted"], ha="left")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"frontier-{theme}.svg"
    fig.savefig(path, format="svg", metadata={"Date": None})
    preview = os.environ.get("LLVQ_FIG_PREVIEW")
    if preview:  # an opaque PNG on the theme's surface, to look at; never committed
        fig.patch.set_alpha(1.0)
        fig.patch.set_facecolor(c["surface"])
        fig.savefig(Path(preview) / f"frontier-{theme}.png", dpi=110,
                    facecolor=c["surface"])
    plt.close(fig)
    return path


def main() -> None:
    points = load()
    for theme in THEMES:
        print("wrote", draw(points, theme).relative_to(ROOT))


if __name__ == "__main__":
    main()
