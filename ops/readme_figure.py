#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8"]
# ///
"""The README's figures, Qwen3 4B, 8B and 14B, each in a light and a dark SVG.

  frontier   MMLU against file size            docs/data/readme-frontier.csv
  speed      decode speed at batch 1           docs/data/paper2-results.csv
  memory     card memory after a prompt        docs/data/paper2-peak.csv

They are written to `docs/figures/<name>-light.svg` and `<name>-dark.svg`. The
README shows them through a <picture> element, so GitHub picks the one that
matches the reader's theme.

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


def themed(theme: str, size: tuple[float, float]):
    """A figure on a transparent surface with the theme's type and axis colours."""
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
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["grid"])
    ax.tick_params(length=0, pad=6)
    return c, fig, ax


def save(fig, name: str, theme: str) -> Path:
    c = THEMES[theme]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}-{theme}.svg"
    fig.savefig(path, format="svg", metadata={"Date": None})
    preview = os.environ.get("LLVQ_FIG_PREVIEW")
    if preview:
        fig.patch.set_alpha(1.0)
        fig.patch.set_facecolor(c["surface"])
        fig.savefig(Path(preview) / f"{name}-{theme}.png", dpi=110,
                    facecolor=c["surface"])
    plt.close(fig)
    return path


def read(name: str) -> list[dict]:
    with open(ROOT / "docs" / "data" / name, newline="") as f:
        return list(csv.DictReader(f))


SPEED_ARMS = [("fp16", "vLLM 0.26.0", "FP16, vLLM"),
              ("awq", "vLLM 0.26.0", "AWQ w4g128, vLLM"),
              ("iq2s", None, "IQ2_S, llama.cpp"),
              ("tetra", None, "Tetra, this repository")]


def draw_speed(theme: str) -> Path:
    """Decode speed at batch 1, one group of bars a size."""
    rows = read("paper2-results.csv")
    c, fig, ax = themed(theme, (8.0, 4.7))
    fig.subplots_adjust(left=0.075, right=0.975, top=0.80, bottom=0.21)
    y, ticks = 0.0, []
    for size in SIZES:
        top = y
        for arm, engine, _ in SPEED_ARMS:
            row = next(r for r in rows if r["model"] == size and r["arm"] == arm
                       and (engine is None or r["engine"] == engine))
            v = float(row["toks"])
            ours = arm == "tetra"
            ax.barh(y, v, height=0.78, color=c[arm], zorder=3)
            ax.text(v + 4, y, f"{v:.1f}", va="center", ha="left", fontsize=10,
                    color=c["ink"], fontweight="bold" if ours else "normal")
            y -= 1.0
        ticks.append(((top + y + 1.0) / 2, size.removeprefix("Qwen3-")))
        y -= 0.9
    for arm, _, label in SPEED_ARMS[::-1]:
        ax.barh(0, 0, color=c[arm], label=label)
    ax.set_yticks([t for t, _ in ticks], [s for _, s in ticks])
    ax.set_xlim(0, 330)
    ax.set_xlabel("Tokens per second, batch 1, greedy", color=c["muted"],
                  fontsize=10.5, labelpad=6)
    ax.grid(axis="x", color=c["grid"], lw=0.8)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9.5,
                    handlelength=1.2, labelspacing=0.45)
    for text in leg.get_texts():
        text.set_color(c["ink"])
    fig.text(0.075, 0.925, "Decode speed on one NVIDIA L40S", fontsize=16,
             fontweight="bold", color=c["ink"], ha="left")
    fig.text(0.075, 0.862,
             "Tetra decodes above 4-bit AWQ in vLLM and below llama.cpp. "
             "Three engines, so compare levels only.",
             fontsize=10.5, color=c["muted"], ha="left")
    fig.text(0.075, 0.048,
             "Measured, medians of five rounds. Tetra, FP16 and AWQ include a 5-token "
             "prompt; IQ2_S is the decode alone.",
             fontsize=8.5, color=c["muted"], ha="left")
    fig.text(0.075, 0.016, "Data and journals: docs/data/paper2-results.csv.",
             fontsize=8.5, color=c["muted"], ha="left")
    return save(fig, "speed", theme)


def draw_memory(theme: str) -> Path:
    """Peak card memory after a prompt of 5 and of 8,192 tokens."""
    val = {(r["model"], r["arm"], r["prompt"]): float(r["peak_gb"])
           for r in read("paper2-peak.csv")}
    series = [("iq2s", "5", 0.45, "IQ2_S, llama.cpp, after 5 tokens"),
              ("iq2s", "8192", 1.0, "IQ2_S, llama.cpp, after 8,192"),
              ("tetra", "5", 0.45, "Tetra, after 5 tokens"),
              ("tetra", "8192", 1.0, "Tetra, after 8,192")]
    c, fig, ax = themed(theme, (8.0, 4.4))
    fig.subplots_adjust(left=0.075, right=0.975, top=0.78, bottom=0.20)
    w = 0.2
    for i, size in enumerate(SIZES):
        for j, (arm, prompt, alpha, _) in enumerate(series):
            v = val[(size, arm, prompt)]
            x = i + (j - 1.5) * w
            ax.bar(x, v, width=w * 0.9, color=c[arm], alpha=alpha, zorder=3)
            ax.text(x, v + 0.2, f"{v:.2f}", ha="center", va="bottom", fontsize=9.5,
                    color=c["ink"],
                    fontweight="bold" if arm == "tetra" and prompt == "8192" else "normal")
    for arm, _, alpha, label in series:
        ax.bar(0, 0, color=c[arm], alpha=alpha, label=label)
    ax.set_xticks(range(len(SIZES)), [s.removeprefix("Qwen3-") for s in SIZES])
    ax.set_ylim(0, 12.5)
    ax.set_ylabel("Peak card memory, GB", color=c["muted"], fontsize=10.5, labelpad=6)
    ax.grid(axis="y", color=c["grid"], lw=0.8)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9.5, ncols=2,
                    handlelength=1.2, columnspacing=1.4, labelspacing=0.45)
    for text in leg.get_texts():
        text.set_color(c["ink"])
    fig.text(0.075, 0.925, "Card memory after a prompt", fontsize=16,
             fontweight="bold", color=c["ink"], ha="left")
    fig.text(0.075, 0.858,
             "About the same after a short prompt. After 8,192 tokens, llama.cpp "
             "holds 2.4 to 3.6 GB less.",
             fontsize=10.5, color=c["muted"], ha="left")
    fig.text(0.075, 0.048,
             "Measured on one L40S, driver memory sampled every 100 ms, peak minus idle.",
             fontsize=8.5, color=c["muted"], ha="left")
    fig.text(0.075, 0.016, "Data and journals: docs/data/paper2-peak.csv.",
             fontsize=8.5, color=c["muted"], ha="left")
    return save(fig, "memory", theme)


def main() -> None:
    points = load()
    for theme in THEMES:
        print("wrote", draw(points, theme).relative_to(ROOT))
        print("wrote", draw_speed(theme).relative_to(ROOT))
        print("wrote", draw_memory(theme).relative_to(ROOT))


if __name__ == "__main__":
    main()
