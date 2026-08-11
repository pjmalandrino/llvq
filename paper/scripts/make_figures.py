#!/usr/bin/env python3
"""Generate the paper's figures from docs/data/*.csv.

The paper rebuilds from the measurements, like everything else in the repo:
no number in a figure is typed by hand. Run `make figures` in paper/.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "data"
OUT = ROOT / "paper" / "figures"

# Okabe-Ito palette: colorblind-safe, print-safe.
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
VERMILLION = "#D55E00"
GRAY = "#7F7F7F"
INK = "#1A1A1A"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
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


def read_csv(name: str) -> list[dict]:
    with open(DATA / name, newline="") as f:
        return list(csv.DictReader(f))


def fig1_layout_scale() -> None:
    """Speedup vs FP16 against in-VRAM rate, six kernels plus the ceiling.

    The dashed hyperbola is the whole point of the figure: an arm that read
    `x` bits per weight at the bandwidth the FP16 control achieves on these
    same shapes would run `16/x` times faster than it. Every marker's
    vertical distance to that curve is therefore the arm's *fraction of its
    own byte bound* — the quantity the campaign compares, since a ratio
    against FP16 mechanically favours whoever reads least.
    """
    rows = read_csv("echelle-formats.csv")
    styles = {
        "Slot32": (GRAY, "o"),
        "Planes14": (BLUE, "o"),
        "Planes12x": (SKY, "o"),
        "Golay70v1": (VERMILLION, "o"),
        "Golay70v2": (VERMILLION, "o"),
        # The competitor: a different marker, because it is not one of ours.
        "AWQ": (GREEN, "s"),
    }
    shown = {
        "Slot32": "Slot32", "Planes14": "Planes14", "Planes12x": "Planes12x",
        "Golay70v1": "Golay70", "Golay70v2": "Golay70, hoisted",
        "AWQ": "AWQ w4g128 (4-bit)",
    }
    # (dx, dy, horizontal alignment) — hand-placed so no two labels overlap
    # at this figure size; the two Golay70 points share an x, so theirs go
    # right and the arrow between them carries no text of its own.
    label_offsets = {
        "Slot32": (0, -17, "center"), "Planes14": (9, 8, "left"),
        "Planes12x": (-9, 11, "right"), "Golay70v1": (9, -5, "left"),
        # AWQ's label goes LEFT: to its right it would cross the ceiling
        # curve, and a label lying on the line it is being compared to is
        # exactly the wrong place for it.
        "Golay70v2": (9, 5, "left"), "AWQ": (-9, 0, "right"),
    }
    fig, ax = plt.subplots(figsize=(4.8, 3.4), layout="constrained")

    # The memory-bound ceiling, and the FP16 control it is anchored on.
    xs = [3.2 + 0.02 * i for i in range(156)]
    ax.plot(xs, [16.0 / x for x in xs], linestyle="--", linewidth=0.9,
            color=GRAY, zorder=1)
    ax.annotate("ceiling: this rate at the FP16 arm's bandwidth",
                xy=(3.75, 16.0 / 3.75), xytext=(4, 7),
                textcoords="offset points", fontsize=7, color=GRAY)
    ax.axhline(1.0, color=GRAY, linewidth=0.8, linestyle=":", zorder=1)
    ax.annotate("FP16 control (16 b/weight)", xy=(6.25, 1.0),
                xytext=(0, 4), textcoords="offset points",
                ha="right", fontsize=7.5, color=GRAY)

    pts = {}
    for r in rows:
        name = r["layout"]
        if name not in styles:
            continue
        color, marker = styles[name]
        x = float(r["bpw_kernel"])
        y = float(r["ratio_vs_fp16"])
        lo, hi = float(r["ratio_lo"]), float(r["ratio_hi"])
        pts[name] = (x, y)
        ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt=marker,
                    color=color, markersize=6, capsize=3,
                    elinewidth=1, capthick=1, zorder=3)
        dx, dy, ha = label_offsets[name]
        ax.annotate(f"{shown[name]}\n{x:.2f} b/w · {y:.2f}× · "
                    f"{r['pct_byte_bound']}%",
                    xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=7.5, va="center", ha=ha, color=INK)

    # The second attack on the same format: same bytes, hoisted decode. The
    # arrow needs no label — two markers of one colour at one rate, joined,
    # is the statement, and the caption says what moved.
    if "Golay70v1" in pts and "Golay70v2" in pts:
        (x1, y1), (x2, y2) = pts["Golay70v1"], pts["Golay70v2"]
        ax.annotate("", xy=(x2, y2 - 0.07), xytext=(x1, y1 + 0.07),
                    arrowprops=dict(arrowstyle="->", color=VERMILLION,
                                    linewidth=0.9, shrinkA=0, shrinkB=0),
                    zorder=2)

    ax.set_xlabel("in-VRAM rate (bits per weight, kernel accounting)")
    ax.set_ylabel("speedup vs FP16 matvec")
    ax.set_xlim(3.2, 6.3)
    ax.set_ylim(0.8, 4.9)
    ax.grid(axis="y", zorder=0)
    fig.savefig(OUT / "fig1_layout_scale.pdf")
    plt.close(fig)


def fig2_phase_attribution() -> None:
    """Per-token phase times, three arms, grouped horizontal bars."""
    rows = read_csv("phases.csv")
    # Arms as shown in the publication dossier: dense f16 and fused f16 from
    # the f16 invocation, fused q8 from the q8 invocation.
    arms = [
        ("dense f16 (reference engine)", "f16", "dense", GRAY),
        ("fused kernel, f16 head", "f16", "fused", BLUE),
        ("fused kernel, q8 head", "q8", "fused", GREEN),
    ]
    phases = [
        ("embed", "embedding"),
        ("blocks_norm", "transformer blocks"),
        ("lm_head", "lm_head"),
        ("argmax_misc", "argmax + misc"),
    ]
    vals = {
        (inv, arm, ph): float(r["median_ms"])
        for r in rows
        for inv, arm, ph in [(r["invocation"], r["arm"], r["phase"])]
    }
    fig, ax = plt.subplots(figsize=(4.8, 2.9), layout="constrained")
    bar_h, group_gap = 0.24, 1.0
    yticks, ylabels = [], []
    for pi, (pkey, plabel) in enumerate(phases):
        base = -pi * group_gap
        yticks.append(base - bar_h)
        ylabels.append(plabel)
        for ai, (alabel, inv, arm, color) in enumerate(arms):
            v = vals[(inv, arm, pkey)]
            y = base - ai * bar_h
            ax.barh(y, v, height=bar_h * 0.92, color=color,
                    label=alabel if pi == 0 else None, zorder=3)
            ax.annotate(f"{v:.2f}" if v >= 0.1 else f"{v:.3f}",
                        xy=(v, y), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=7, color=INK)
    ax.set_yticks(yticks, ylabels)
    ax.set_xlabel("median ms per token (sync-bounded)")
    ax.set_xlim(0, 31)
    ax.grid(axis="x", zorder=0)
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(OUT / "fig2_phase_attribution.pdf")
    plt.close(fig)


def fig3_progression() -> None:
    """VRAM and throughput across the four integration steps: two panels
    sharing one x axis (never a dual-axis chart)."""
    rows = read_csv("progression.csv")
    steps = [
        "no kernel\n(Aug 5)",
        "fused kernel\nSlot32 (Aug 6)",
        "Planes14\nlayout (Aug 6)",
        "int8 embedding\n+ q8 head (Aug 7)",
    ]
    vram = [float(r["vram_gb"]) for r in rows]
    tokps = [float(r["tokps"]) for r in rows]
    x = range(len(rows))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.8, 3.4),
                                   sharex=True, layout="constrained")
    ax1.plot(x, vram, "o-", color=BLUE, markersize=5, linewidth=1.4, zorder=3)
    for xi, v in zip(x, vram):
        ax1.annotate(f"{v:.2f}", xy=(xi, v), xytext=(0, 6),
                     textcoords="offset points", ha="center", fontsize=7.5)
    ax1.set_ylabel("VRAM (GB)")
    ax1.set_ylim(0, 9.6)
    ax1.grid(axis="y", zorder=0)

    ax2.plot(x, tokps, "o-", color=GREEN, markersize=5, linewidth=1.4, zorder=3)
    for xi, v in zip(x, tokps):
        ax2.annotate(f"{v:.1f}", xy=(xi, v), xytext=(0, 6),
                     textcoords="offset points", ha="center", fontsize=7.5)
    ax2.set_ylabel("throughput (tok/s)")
    ax2.set_ylim(0, 105)
    ax2.set_xticks(list(x), steps)
    ax2.grid(axis="y", zorder=0)
    fig.savefig(OUT / "fig3_progression.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_layout_scale()
    fig2_phase_attribution()
    fig3_progression()
    print(f"wrote 3 figures to {OUT}")


if __name__ == "__main__":
    main()
