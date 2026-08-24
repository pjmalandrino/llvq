#!/usr/bin/env python3
"""Generate the paper's figures from docs/data/*.csv.

The paper rebuilds from the measurements, like everything else in the repo:
no number in a figure is typed by hand. Run `make figures` in paper/.

Five figures, one function each, all under the same guard: every CSV row a
figure claims to draw is drawn or the script fails with the row named. The
few constants that are not measurements (record field widths, which are
layout definitions; the embedding share of each model, which is architecture
arithmetic) are declared once, commented with where they come from, and
listed in `HARDCODED` so a reader can audit them without reading the code.

  fig1_layout_scale.pdf  speedup vs in-VRAM rate, ten arms   echelle-formats.csv
  fig_records.pdf        the four records as bit-scale maps  echelle-formats.csv
  fig_scale.pdf          three model sizes, three panels     ppl-appariee.csv,
                                                             mmlu-appariee.csv,
                                                             echelle-4b-8b.csv
  fig_a100.pdf           achieved GB/s on two cards          echelle-formats*.csv
  fig_attribution.pdf    the Slot32 time, term by term       attribution-slot32.csv
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# acmsmall's \textwidth is 395.8 pt = 5.50 in (main.log). Figures are drawn
# at the width they are included at, so 6–8 pt type prints at 6–8 pt.
TEXTWIDTH_IN = 5.5

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "data"
OUT = ROOT / "paper" / "figures"

# Okabe-Ito palette: colorblind-safe, print-safe.
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

# Every number drawn by this script that is NOT read from a CSV, with its
# provenance. Printed by main() so the list is visible on every build.
HARDCODED: list[tuple[str, str]] = []


def hardcoded(what: str, where: str) -> None:
    HARDCODED.append((what, where))


def read_csv(name: str) -> list[dict]:
    with open(DATA / name, newline="") as f:
        return list(csv.DictReader(f))


def require_plotted(fig: str, csv_name: str, expected: set, plotted: set,
                    excluded: dict[str, str]) -> None:
    """Fail if a CSV row the figure should draw was not drawn.

    `excluded` names the rows deliberately left out, each with its reason,
    so a row can only vanish from a figure by being listed here — never by
    a renamed key falling through a lookup with exit code 0.
    """
    missing = expected - plotted - set(excluded)
    if missing:
        raise SystemExit(
            f"{fig}: {sorted(missing)} in {csv_name} but not plotted — "
            "add them to the figure or list them as excluded with a reason"
        )
    for key, why in excluded.items():
        print(f"{fig}: {csv_name} row {key!r} not drawn — {why}")


# ---------------------------------------------------------------------------
# Fig. 1 — the layout scale
# ---------------------------------------------------------------------------

def fig1_layout_scale() -> None:
    """Speedup vs FP16 against in-VRAM rate, seven arms, the byte bound and
    the no-weights control.

    The dashed hyperbola is the whole point of the figure: an arm reading `x`
    bits per weight at the bandwidth the FP16 control achieves on these same
    shapes would run `16/x` times faster than it, since
    speedup = (16/x)·(B/B_fp16). An arm's *fraction of its own byte bound* is
    therefore y ÷ (16/x) — a RATIO, not a gap.

    Which is why the y axis is logarithmic. On a linear axis the distance to
    the hyperbola is (16/x)·(1−f): it shrinks as the rate grows even at
    constant f, so the eye reads the wrong ordering — AWQ, the arm converting
    the MOST of its bound, sits closest to the curve and would look worst.
    In log space the gap becomes log(16/x) − log(y) = −log(f), so equal
    fractions are equal distances.

    Vocabulary (fixed by the paper): the hyperbola is the *byte bound*; the
    horizontal rule is the *no-weights control*, a kernel of ours over the
    same 252 launches that reads no weight bytes. Ours are circles in a cool
    palette; the two deployed kernels are squares in a warm one.
    """
    rows = read_csv("echelle-formats.csv")
    styles = {
        "Slot32": (GRAY, "o"),
        "Planes14": (BLUE, "o"),
        "Planes12x": (SKY, "o"),
        "Golay70v1": (GREEN, "o"),
        "Golay70v2": (GREEN, "o"),
        # The competitors: warm colours and a square marker, not ours.
        "AWQ": (ORANGE, "s"),
        "QTIP": (VERMILLION, "s"),
    }
    shown = {
        "Slot32": "Slot32", "Planes14": "Planes14", "Planes12x": "Planes12x",
        "Golay70v1": "Golay70", "Golay70v2": "Golay70, hoisted",
        "AWQ": "AWQ w4g128 (4-bit)", "QTIP": "QTIP (2-bit)",
    }
    # (dx, dy, horizontal alignment) in points — hand-placed so no two labels
    # overlap at this figure size. The two Golay70 points share an x: the
    # hoisted one is labelled to its LEFT (to the right it collided with the
    # Planes12x label), the original to its right; the arrow between them
    # carries no text of its own.
    label_offsets = {
        "Slot32": (0, -17, "center"), "Planes14": (9, 8, "left"),
        "Planes12x": (-9, 11, "right"), "Golay70v1": (9, -5, "left"),
        "Golay70v2": (-9, 0, "right"),
        # AWQ's label goes LEFT: to its right it would cross the byte-bound
        # curve, and a label lying on the line it is compared to is exactly
        # the wrong place for it.
        "AWQ": (-9, 0, "right"),
        # QTIP is the leftmost and highest point; its label goes below-right
        # so it clears both the byte bound and the control rule.
        "QTIP": (7, -13, "left"),
    }
    fig, ax = plt.subplots(figsize=(4.8, 3.4), layout="constrained")

    # The byte bound, anchored on the FP16 control. Its label sits on the
    # upper-left stretch of the curve, where nothing else is drawn; it used
    # to sit at x ≈ 3.75, under the QTIP point and on top of the control
    # rule.
    xs = [1.85 + 0.02 * i for i in range(224)]
    ax.plot(xs, [16.0 / x for x in xs], linestyle="--", linewidth=0.9,
            color=GRAY, zorder=1)
    ax.annotate("byte bound: this rate at the\nFP16 control's bandwidth",
                xy=(2.45, 16.0 / 2.45), xytext=(11, 6),
                textcoords="offset points", fontsize=7, color=GRAY,
                va="bottom", ha="left")
    ax.axhline(1.0, color=GRAY, linewidth=0.8, linestyle=":", zorder=1)
    ax.annotate("FP16 control (16 b/weight)", xy=(6.45, 1.0),
                xytext=(0, 4), textcoords="offset points",
                ha="right", fontsize=7.5, color=GRAY)
    # The no-weights control: our own launch geometry with nothing to read.
    # Every arm of ours is below it; the 2-bit competitor is above.
    control = next((float(r["ratio_vs_fp16"]) for r in rows
                    if r["layout"] == "nullk"), None)
    if control is not None:
        ax.axhline(control, color=INK, linewidth=0.8, linestyle=":",
                   zorder=1)
        ax.annotate("no-weights control (our launch geometry), "
                    f"{control:.2f}$\\times$",
                    xy=(6.45, control), xytext=(0, 4),
                    textcoords="offset points", ha="right", fontsize=7.5,
                    color=INK)

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

    # The second attack on the same format: same bytes, hoisted decode.
    if "Golay70v1" in pts and "Golay70v2" in pts:
        (x1, y1), (x2, y2) = pts["Golay70v1"], pts["Golay70v2"]
        ax.annotate("", xy=(x2, y2 - 0.07), xytext=(x1, y1 + 0.07),
                    arrowprops=dict(arrowstyle="->", color=GREEN,
                                    linewidth=0.9, shrinkA=0, shrinkB=0),
                    zorder=2)

    ax.text(1.7, 1.06, "circles: ours · squares: deployed kernels",
            fontsize=7, color=GRAY, va="bottom", ha="left")

    # Three rows are deliberately absent as points: the two FP16 controls
    # (x = 16 is off the axis) and the no-weights control, which reads
    # 0.159 b/weight and would sit off the other end — it is drawn as a
    # horizontal rule, because what matters about it is its height.
    require_plotted(
        "fig1", "echelle-formats.csv", {r["layout"] for r in rows}, set(pts),
        {"FP16": "the 1.00× anchor, drawn as the dotted rule at 1×",
         "cuBLASf16": "16 b/weight is off the rate axis; it checks the control",
         "nullk": "drawn as the no-weights control rule, not as a point"},
    )

    ax.set_xlabel("in-VRAM rate (bits per weight, kernel accounting)")
    ax.set_ylabel("speedup vs FP16 matvec (log)")
    ax.set_xlim(1.6, 6.5)
    ax.set_yscale("log")
    ax.set_ylim(0.9, 9.5)
    ticks = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    ax.set_yticks(ticks, [f"{t:g}×" for t in ticks])
    ax.set_yticks([], minor=True)
    ax.grid(axis="y", zorder=0)
    fig.savefig(OUT / "fig1_layout_scale.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig. A — the four records, to the bit
# ---------------------------------------------------------------------------

# Field widths are constants of each layout, transcribed from the record
# diagram at the head of its decoder:
#   Slot32    llvq-cuda/kernels/llvq_slot.cuh      [class 9][gain 1][smask 24][m1..m_{L-1} @24]
#   Planes14  llvq-cuda/kernels/llvq_planes.cuh    [class 9][gain 1][smask 24][plane0 24][plane1 24][plane2 24][pad 6]
#   Planes12x llvq-cuda/kernels/llvq_planes12.cuh  [class 9][gain 1][smask 24][plane0 24][plane1 24][pad 14]
#   Golay70   llvq-cuda/kernels/llvq_golay.cuh     [class 9][gain 1][golay 12][A 24][B 24][pad 2]
# Strides and read windows come from the same headers ("The read window"
# section of each). The exception rates are the bench's own count on the
# 4B artifact (docs/mesures/f2-p3-qtip-banc-2026-08-21.txt, construction
# block: 3.3824 % of blocks at L = 5 for Planes12x; 7.4357 % for Golay70).
FIELD_COLORS = {
    "class": INK, "gain": VERMILLION, "smask": SKY, "mask": BLUE,
    "plane": BLUE, "golay": PURPLE, "AB": GREEN, "pad": "white",
}


def fig_records() -> None:
    """Four horizontal bit-scale bars: one record of each layout, field by
    field, with stride, read window and side channel under each bar; at
    right, what the layout costs and buys, read from echelle-formats.csv
    (bpw_kernel, ratio_vs_fp16)."""
    rows = {r["layout"]: r for r in read_csv("echelle-formats.csv")}
    arms = {  # figure row → CSV row
        "Slot32": "Slot32", "Planes14": "Planes14",
        "Planes12x": "Planes12x", "Golay70": "Golay70v2",
    }
    # (label, width in bits, kind). `optional` marks the field that exists
    # only at L = 5 (Slot32's m4): dashed outline, lighter fill.
    records = {
        "Slot32": [
            ("class", 9, "class"), ("g", 1, "gain"), ("sign mask", 24, "smask"),
            ("m1", 24, "mask"), ("m2", 24, "mask"), ("m3", 24, "mask"),
            ("m4 (L = 5)", 24, "optional"),
        ],
        "Planes14": [
            ("class", 9, "class"), ("g", 1, "gain"), ("sign mask", 24, "smask"),
            ("plane 0", 24, "plane"), ("plane 1", 24, "plane"),
            ("plane 2", 24, "plane"), ("6", 6, "pad"),
        ],
        "Planes12x": [
            ("class", 9, "class"), ("g", 1, "gain"), ("sign mask", 24, "smask"),
            ("plane 0", 24, "plane"), ("plane 1", 24, "plane"),
            ("14", 14, "pad"),
        ],
        "Golay70": [
            ("class", 9, "class"), ("g", 1, "gain"), ("Golay\nrank", 12, "golay"),
            ("A", 24, "AB"), ("B", 24, "AB"), ("", 2, "pad"),
        ],
    }
    width_label = {  # printed to the right of each bar
        "Slot32": "106–130 b", "Planes14": "112 b", "Planes12x": "96 b",
        "Golay70": "72 b",
    }
    # Stride and read window only; the side channels (base table, exception
    # tables, codeword table) and the exception rates are in the caption,
    # so the under-record line stays one short line at 6.5 pt.
    geometry = {  # stride, read window — from each decoder's header
        "Slot32": "stride: widest record of its 32-block group · window 5 words (20 B)",
        "Planes14": "stride 14 B, uniform (byte offset ≡ 0 or 2 mod 4) · window 4 words (16 B)",
        "Planes12x": "stride 12 B, word-aligned · window 3 words (12 B) · exception side channel",
        "Golay70": "stride 9 B · window 3 words (12 B) · codeword table and exception side channel",
    }
    hardcoded("record field widths, strides, windows (Fig. A)",
              "llvq-cuda/kernels/llvq_{slot,planes,planes12,golay}.cuh headers")

    order = ["Slot32", "Planes14", "Planes12x", "Golay70"]
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.48), layout="constrained")
    bar_h = 0.5
    pitch = 1.42
    right = 196
    plotted = set()
    for i, name in enumerate(order):
        y = -i * pitch
        x = 0
        for label, width, kind in records[name]:
            optional = kind == "optional"
            color = FIELD_COLORS["mask" if optional else kind]
            face = "#DCE9F4" if optional else color
            ax.add_patch(Rectangle(
                (x, y - bar_h / 2), width, bar_h, facecolor=face,
                edgecolor=GRAY if kind == "pad" else INK,
                linewidth=0.6, linestyle="--" if optional else "-",
                hatch="////" if kind == "pad" else None, zorder=2))
            dark = kind in ("class", "mask", "plane", "golay", "AB")
            # The 9-bit class field and the 1-bit gain field are too narrow
            # for 6.5 pt type: one annotation above the bar names both.
            if kind == "class":
                ax.annotate("class 9 · gain 1", xy=(x, y + bar_h / 2),
                            xytext=(0, 2.5), textcoords="offset points",
                            ha="left", va="bottom", fontsize=6.5,
                            color=INK)
            elif kind != "gain" and label:
                ax.text(x + width / 2, y, label, ha="center", va="center",
                        fontsize=6.5, color="white" if dark else INK,
                        zorder=3, linespacing=0.9)
            x += width
        ax.annotate(width_label[name], xy=(x, y), xytext=(3, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=6.5, color=INK)
        ax.text(-3, y, name, ha="right", va="center", fontsize=8,
                color=INK, fontweight="bold")
        ax.text(0, y - bar_h / 2 - 0.05, geometry[name], ha="left", va="top",
                fontsize=6.5, color=GRAY, linespacing=1.15)
        r = rows[arms[name]]
        plotted.add(arms[name])
        # Golay70's row is the hoisted decoder's (Table 1 lists both), and
        # the label says so: the stored bytes are the same, the speed is not.
        speed = f"{float(r['ratio_vs_fp16']):.2f}× FP16"
        if name == "Golay70":
            v1 = float(rows["Golay70v1"]["ratio_vs_fp16"])
            speed = f"{v1:.2f}× / {float(r['ratio_vs_fp16']):.2f}× hoisted"
        ax.text(right, y, f"{float(r['bpw_kernel']):.2f} b/w\n{speed}",
                ha="right", va="center", fontsize=7.5, color=INK,
                linespacing=1.0)

    # Word ruler: a faint line every 32 bits (one 32-bit word).
    top = bar_h / 2 + 0.28
    bottom = -(len(order) - 1) * pitch - bar_h / 2 - 0.8
    for w in range(0, 129, 32):
        ax.plot([w, w], [top, bottom], color="#E4E4E4", linewidth=0.5,
                zorder=1)
        ax.text(w, top + 0.02, f"{w}", ha="center", va="bottom", fontsize=6.5,
                color=GRAY)
    ax.text(-3, top + 0.02, "bits", ha="right", va="bottom", fontsize=6.5,
            color=GRAY)
    ax.text(0, bottom + 0.02, "hatched: padding to the byte stride · "
            "dashed: present only in 5-level classes",
            ha="left", va="bottom", fontsize=6.5, color=GRAY)
    ax.text(right, top + 0.02, "in VRAM (kernel\naccounting) · vs FP16",
            ha="right", va="bottom", fontsize=6.5, color=GRAY)

    # Rows deliberately absent: the figure draws the four record formats;
    # the two controls, the no-weights control, the un-hoisted Golay70
    # decoder (same bytes as the hoisted one) and the two competitors have
    # no record of ours to draw.
    require_plotted(
        "fig_records", "echelle-formats.csv", set(rows), plotted,
        {"FP16": "no record: the f16 control",
         "cuBLASf16": "no record: the f16 control via cuBLAS",
         "nullk": "no record: reads no weights",
         "Golay70v1": "same bytes as Golay70v2; the hoisted decoder's row is shown",
         "AWQ": "competitor format, not ours to draw",
         "QTIP": "competitor format, not ours to draw"},
    )

    ax.set_xlim(-34, right + 2)
    ax.set_ylim(bottom, top + 0.55)
    ax.axis("off")
    fig.savefig(OUT / "fig_records.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig. C — three model sizes
# ---------------------------------------------------------------------------

# Embedding share of each model's parameters, in %: vocabulary × hidden over
# params_total, both tables counted where the head is untied (8B, 14B).
# Architecture arithmetic, not a measurement; the paper's own text carries
# these three values (CLAUDE.md §6, "Memory, at all three sizes").
EMBED_SHARE_PCT = {"Qwen3-4B": 9.7, "Qwen3-8B": 15.2, "Qwen3-14B": 10.5}


def fig_scale() -> None:
    """Three panels against model size: paired perplexity excess, paired
    MMLU gap, whole-model bits per parameter."""
    ppl = read_csv("ppl-appariee.csv")
    mmlu = read_csv("mmlu-appariee.csv")
    scale = read_csv("echelle-4b-8b.csv")
    hardcoded("embedding share 9.7 / 15.2 / 10.5 % (Fig. C, panel iii)",
              "CLAUDE.md §6 — architecture arithmetic, no CSV carries it")

    # x is the parameter count in billions, from echelle-4b-8b.csv.
    sizes = {}
    for r in scale:
        sizes[r["model"]] = int(r["params_total"]) / 1e9
    models = sorted(sizes, key=sizes.get)
    xs = [sizes[m] for m in models]
    short = {m: m.replace("Qwen3-", "") for m in models}

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 1.95),
                                     layout="constrained")

    # Marker AND line style differ per series so the three curves of a panel
    # stay apart in a grayscale print (colour alone does not survive one).
    def series(ax, rows, key_col, key, val, lo, hi, color, label, dx,
               marker="o", ls="-"):
        pts = [(sizes[r["model"]] + dx, float(r[val]), float(r[lo]), float(r[hi]))
               for r in rows if r[key_col] == key]
        pts.sort()
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts],
                    yerr=[[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]],
                    fmt=marker + ls, color=color, markersize=4, linewidth=1.1,
                    capsize=2.5, elinewidth=0.9, capthick=0.9, label=label,
                    zorder=3)
        return {(r["model"], r[key_col]) for r in rows if r[key_col] == key}

    # (i) perplexity excess over the reference, %, paired 95% CI.
    done_ppl = set()
    done_ppl |= series(a1, ppl, "pair", "llvq2_over_f16", "excess_pct",
                       "ci_lo_pct", "ci_hi_pct", BLUE, "2-bit / FP16", 0)
    done_ppl |= series(a1, ppl, "pair", "llvq2_over_awq4", "excess_pct",
                       "ci_lo_pct", "ci_hi_pct", VERMILLION, "2-bit / 4-bit", 0.18,
                       marker="s", ls="--")
    done_ppl |= series(a1, ppl, "pair", "awq4_over_f16", "excess_pct",
                       "ci_lo_pct", "ci_hi_pct", GREEN, "4-bit / FP16", -0.18,
                       marker="^", ls=":")
    a1.set_ylabel("perplexity excess (%)")
    a1.set_ylim(0, 48)
    a1.legend(frameon=False, loc="upper right")

    # (ii) MMLU gap in points, paired 95% CI, zero line.
    done_mmlu = set()
    a2.axhline(0, color=GRAY, linewidth=0.7, linestyle=":", zorder=1)
    done_mmlu |= series(a2, mmlu, "pair", "f16_minus_llvq2", "delta_pp",
                        "ci_lo_pp", "ci_hi_pp", BLUE, "FP16 − 2-bit", 0)
    done_mmlu |= series(a2, mmlu, "pair", "awq4_minus_llvq2", "delta_pp",
                        "ci_lo_pp", "ci_hi_pp", VERMILLION, "4-bit − 2-bit", 0.18,
                        marker="s", ls="--")
    done_mmlu |= series(a2, mmlu, "pair", "f16_minus_awq4", "delta_pp",
                        "ci_lo_pp", "ci_hi_pp", GREEN, "FP16 − 4-bit", -0.18,
                        marker="^", ls=":")
    a2.set_ylabel("MMLU gap (points)")
    a2.set_ylim(-3, 19)
    a2.legend(frameon=False, loc="upper right")

    # (iii) whole-model bits per parameter, served configuration vs AWQ.
    done_bpp = set()
    done_bpp |= series(a3, scale, "arm", "llvq2", "vram_bits_per_param",
                       "vram_bits_per_param", "vram_bits_per_param", BLUE,
                       "2-bit, served (ours)", 0)
    done_bpp |= series(a3, scale, "arm", "awq4", "vram_bits_per_param",
                       "vram_bits_per_param", "vram_bits_per_param", GREEN,
                       "4-bit AWQ, official", 0, marker="s", ls="--")
    for r in scale:
        if r["arm"] == "llvq2":
            x, y = sizes[r["model"]], float(r["vram_bits_per_param"])
            a3.annotate(f"{y:.3f}", xy=(x, y), xytext=(0, -9),
                        textcoords="offset points", ha="center", fontsize=6.5,
                        color=BLUE)
        if r["arm"] == "awq4":
            x, y = sizes[r["model"]], float(r["vram_bits_per_param"])
            a3.annotate(f"{y:.3f}", xy=(x, y), xytext=(0, 5),
                        textcoords="offset points", ha="center", fontsize=6.5,
                        color=GREEN)
    a3.set_ylabel("b/param, whole model")
    a3.set_ylim(4.9, 6.5)
    a3.legend(frameon=False, loc="upper right")

    for ax in (a1, a2, a3):
        ax.set_xticks(xs, [short[m] for m in models])
        ax.set_xlim(min(xs) - 1.2, max(xs) + 1.2)
        ax.set_xlabel("model size")
        ax.grid(axis="y", zorder=0)
    # The embedding share rides on the tick labels of the memory panel: it
    # is the mechanism behind the non-monotone margin, and a model property.
    a3.set_xticks(xs, [f"{short[m]}\n{EMBED_SHARE_PCT[m]:.1f}%"
                       for m in models])
    a3.set_xlabel("size; embedding share")

    require_plotted("fig_scale", "ppl-appariee.csv",
                    {(r["model"], r["pair"]) for r in ppl}, done_ppl, {})
    require_plotted("fig_scale", "mmlu-appariee.csv",
                    {(r["model"], r["pair"]) for r in mmlu}, done_mmlu, {})
    require_plotted(
        "fig_scale", "echelle-4b-8b.csv",
        {(r["model"], r["arm"]) for r in scale}, done_bpp,
        {(m, "f16"): "16.000 b/param by construction, off the axis"
         for m in models})

    fig.savefig(OUT / "fig_scale.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig. D — two memory hierarchies
# ---------------------------------------------------------------------------

def fig_a100() -> None:
    """Dumbbell per arm: achieved GB/s on the L40S (filled) and the A100
    (hollow), joined; vertical rules for the FP16 control and cuBLAS on each
    card. GB/s is read from the `gbps` column of both CSVs, which the bench
    forms from each arm's fastest kept round (the paper's stated
    convention); nothing is recomputed here."""
    l40s = {r["layout"]: r for r in read_csv("echelle-formats.csv")}
    a100 = {r["layout"]: r for r in read_csv("echelle-formats-a100.csv")}
    shown = {
        "AWQ": "AWQ w4g128 (4-bit)", "Slot32": "Slot32", "Planes14": "Planes14",
        "Planes12x": "Planes12x", "Golay70v2": "Golay70, hoisted",
        "Golay70v1": "Golay70",
    }
    colors = {"AWQ": ORANGE, "Slot32": GRAY, "Planes14": BLUE,
              "Planes12x": SKY, "Golay70v2": GREEN, "Golay70v1": GREEN}
    order = [k for k in shown if k in l40s and k in a100]
    for k in a100:
        if k not in l40s:
            raise SystemExit(f"fig_a100: {k!r} on the A100 but not on the L40S")
    for k in l40s:
        if k not in a100:
            print(f"fig_a100: {k!r} has no A100 point and is excluded")

    # Drawn at full text width: at 0.8·linewidth the value column and the
    # six row labels leave the GB/s axis under two inches, and the labels of
    # the four control rules (661, 672, 1052, 1204) cannot be kept apart.
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.52), layout="constrained")
    card_color = {"L40S": INK, "A100": GRAY}
    ytop = len(order) - 0.35
    # Vertical rules: the FP16 control and cuBLAS on each card. Their labels
    # sit above the plot area on two staggered rows (FP16 on the upper row,
    # cuBLAS on the lower), each anchored to its rule: the two L40S rules
    # are 11 GB/s apart and the two A100 labels would otherwise collide.
    rules = (("L40S", "FP16", "FP16", "right", -2, 1),
             ("L40S", "cuBLASf16", "cuBLAS", "left", 2, 0),
             ("A100", "FP16", "FP16", "left", 2, 0),
             ("A100", "cuBLASf16", "cuBLAS", "right", -2, 1))
    row_y = {0: len(order) + 0.05, 1: len(order) + 0.55}
    for card, key, lab, ha, dx, row in rules:
        table = l40s if card == "L40S" else a100
        g = float(table[key]["gbps"])
        ax.axvline(g, ymax=0.87, color=card_color[card], linewidth=0.8,
                   linestyle="--" if card == "L40S" else ":", zorder=1)
        ax.annotate(f"{lab} {card} {g:.0f}", xy=(g, row_y[row]), xytext=(dx, 0),
                    textcoords="offset points", ha=ha, va="center",
                    fontsize=6.5, color=card_color[card],
                    annotation_clip=False)

    plotted = set()
    for i, k in enumerate(order):
        y = len(order) - 1 - i
        gl, ga = float(l40s[k]["gbps"]), float(a100[k]["gbps"])
        rl, ra = l40s[k]["ratio_vs_fp16"], a100[k]["ratio_vs_fp16"]
        ax.plot([ga, gl], [y, y], color=colors[k], linewidth=1.2, zorder=2)
        ax.plot(gl, y, marker="o", markersize=6, color=colors[k],
                markeredgecolor=colors[k], zorder=3)
        ax.plot(ga, y, marker="o", markersize=6, markerfacecolor="white",
                markeredgecolor=colors[k], markeredgewidth=1.3, zorder=3)
        # Values in a column outside the axes: inside, the labels of the
        # four fastest arms ran across the L40S control rules.
        ax.annotate(f"{gl:.0f} → {ga:.0f} GB/s\n"
                    f"{float(rl):.2f}× → {float(ra):.2f}× FP16",
                    xy=(1.015, y), xycoords=("axes fraction", "data"),
                    ha="left", va="center", fontsize=7, color=INK,
                    linespacing=1.0, annotation_clip=False)
        plotted.add(k)
    ax.annotate("L40S → A100", xy=(1.015, ytop), xycoords=("axes fraction", "data"),
                xytext=(0, 30), textcoords="offset points", ha="left",
                va="bottom", fontsize=6.5, color=GRAY, annotation_clip=False)

    # The no-weights control has no bandwidth to show (0.07 GB); its move
    # between cards is a time, drawn as a note from the two med_ms columns.
    n_l, n_a = float(l40s["nullk"]["med_ms"]), float(a100["nullk"]["med_ms"])
    ax.text(15, -0.85,
            f"no-weights control, not plotted:\n"
            f"{n_l:.3f} → {n_a:.3f} ms (×{n_a / n_l:.2f})",
            ha="left", va="center", fontsize=6.5, color=GRAY, linespacing=0.95)
    # Legend at lower right, where no arm reaches and no rule passes.
    ax.text(1275, -0.85, "● L40S (GDDR6)\n○ A100 (HBM2e)", ha="right",
            va="center", fontsize=6.5, color=INK, linespacing=1.1)

    ax.set_yticks([len(order) - 1 - i for i in range(len(order))],
                  [shown[k] for k in order])
    ax.set_xlabel("achieved bandwidth (GB/s, fastest kept round)")
    ax.set_xlim(0, 1290)
    ax.set_ylim(-1.25, len(order) + 0.8)
    ax.grid(axis="x", zorder=0)

    excluded = {"FP16": "drawn as a vertical rule on each card",
                "cuBLASf16": "drawn as a vertical rule on each card",
                "nullk": "no bandwidth to show; its ms move is the note"}
    require_plotted("fig_a100", "echelle-formats-a100.csv", set(a100),
                    plotted, excluded)
    require_plotted("fig_a100", "echelle-formats.csv", set(l40s), plotted,
                    {**excluded,
                     "QTIP": "no A100 point; absent rather than half-filled"})
    fig.savefig(OUT / "fig_a100.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig. F — the Slot32 attribution
# ---------------------------------------------------------------------------

def fig_attribution() -> None:
    """Two stacked horizontal bars, in ms. Top: the Slot32 time as the DRAM
    floor (grey) plus the five attributed terms. Bottom: the same five terms
    magnified to the width of the figure, labelled, with the fusion recovery
    as a bracket over the term it removes."""
    rows = {r["term"]: r for r in read_csv("attribution-slot32.csv")}
    terms = [  # (key, label, colour) in stacking order
        ("payload_streaming", "payload streaming\n(bases + 5-word window)", SKY),
        ("class_table_gather", "class-table\ngather", PURPLE),
        ("shared_x_reads", "24 shared-memory\nreads of x", GREEN),
        ("residual_decode", "residual decode\n(by difference)", BLUE),
        ("launch_latency", "unmasked latency / occupancy\n(the fusion term)", VERMILLION),
    ]
    total = float(rows["total"]["ms"])
    floor = float(rows["dram_floor"]["ms"])
    gap = total - floor
    summed = floor + sum(float(rows[k]["ms"]) for k, _, _ in terms)
    if abs(summed - total) > 0.002:
        raise SystemExit(f"fig_attribution: terms sum to {summed:.3f}, total is {total:.3f}")
    pct = 100 * floor / total
    slot32 = next(r for r in read_csv("echelle-formats.csv") if r["layout"] == "Slot32")
    if abs(pct - float(slot32["pct_byte_bound"])) > 1:
        raise SystemExit(f"fig_attribution: floor/total = {pct:.1f}% but "
                         f"echelle-formats.csv says {slot32['pct_byte_bound']}%")

    hardcoded("floor definition 2.50 GB at 662 GB/s (Fig. F, in-bar text)",
              "docs/mesures/attribution-cuda-2026-08-05.txt §2 — the CSV's "
              "dram_floor source field names the same two quantities")
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.0), layout="constrained")
    plotted = {"dram_floor", "total"}
    y_main, y_zoom, bar_h = 2.1, 0.55, 0.56
    # --- the whole time, to scale
    ax.barh(y_main, floor, height=bar_h, color=LIGHT, edgecolor="white",
            linewidth=0.6, zorder=2)
    ax.text(floor / 2, y_main,
            f"DRAM floor {floor:.3f} ms = {pct:.0f}% of the arm's {total:.3f} ms\n"
            f"(2.50 GB at the FP16 control's 662 GB/s)",
            ha="center", va="center", fontsize=6.3, color=INK, zorder=3,
            linespacing=1.0)
    x = floor
    for key, _, color in terms:
        w = float(rows[key]["ms"])
        ax.barh(y_main, w, left=x, height=bar_h, color=color,
                edgecolor="white", linewidth=0.4, zorder=2)
        x += w
    ax.annotate(f"{total:.3f} ms", xy=(total, y_main), xytext=(4, 0),
                textcoords="offset points", ha="left", va="center",
                fontsize=7, color=INK)
    # --- the gap, magnified
    z0, z1 = 0.25, 6.05
    k = (z1 - z0) / gap
    ax.plot([floor, z0], [y_main - bar_h / 2, y_zoom + bar_h / 2],
            color=GRAY, linewidth=0.5, linestyle=":", zorder=1)
    ax.plot([total, z1], [y_main - bar_h / 2, y_zoom + bar_h / 2],
            color=GRAY, linewidth=0.5, linestyle=":", zorder=1)
    ax.text(0.05, (y_main + y_zoom) / 2 + 0.1,
            f"gap above the floor:\n{gap:.3f} ms, magnified ×{k:.1f}",
            ha="left", va="center", fontsize=6.3, color=GRAY,
            linespacing=0.95)
    x = z0
    for key, label, color in terms:
        w = float(rows[key]["ms"])
        wz = w * k
        ax.barh(y_zoom, wz, left=x, height=bar_h, color=color,
                edgecolor="white", linewidth=0.6, zorder=2)
        mid = x + wz / 2
        value = f"{w:.3f} ms · {rows[key]['share_pct']}%"
        if wz >= 0.9:
            ax.text(mid, y_zoom, value, ha="center", va="center",
                    fontsize=6.5, color="white", zorder=3)
            ax.text(mid, y_zoom - bar_h / 2 - 0.08, label, ha="center",
                    va="top", fontsize=6.3, color=INK, linespacing=0.95)
        else:
            # The two thin terms: label on a second tier below, offset to
            # the free side (gather left, shared-x right), with a leader.
            side = -1 if key == "class_table_gather" else 1
            lx = mid + side * 0.38
            ax.annotate(f"{label}\n{value}", xy=(mid, y_zoom - bar_h / 2),
                        xytext=(lx, y_zoom - bar_h / 2 - 0.58),
                        ha="center", va="top", fontsize=6.3, color=INK,
                        linespacing=0.95,
                        arrowprops=dict(arrowstyle="-", color=GRAY,
                                        linewidth=0.5, shrinkA=0, shrinkB=1))
        x += wz
        plotted.add(key)

    # The bracket: the latency term is what fusing q/k/v and gate/up
    # recovers, measured on Slot32 in the ten-arm run (and on Planes14).
    rec = float(rows["recovered_by_fusion_slot32"]["ms"])
    rec14 = float(rows["recovered_by_fusion_planes14"]["ms"])
    plotted |= {"recovered_by_fusion_slot32", "recovered_by_fusion_planes14"}
    yb = y_zoom + bar_h / 2 + 0.1
    ax.plot([z1 - rec * k, z1 - rec * k, z1, z1], [yb - 0.08, yb, yb, yb - 0.08],
            color=INK, linewidth=0.8, zorder=3)
    ax.annotate("fused launches (144 instead of 252) recover "
                f"{rec:.3f} ms on Slot32 ({rows['recovered_by_fusion_slot32']['share_pct']}% of its time)\n"
                f"and {rec14:.3f} ms on Planes14 ({rows['recovered_by_fusion_planes14']['share_pct']}%)",
                xy=(z1 - 0.28, yb + 0.05), ha="right", va="bottom",
                fontsize=6.3, color=INK, linespacing=0.95)

    require_plotted("fig_attribution", "attribution-slot32.csv", set(rows),
                    plotted, {})

    ax.set_xlim(0, 6.45)
    ax.set_ylim(-0.95, y_main + bar_h / 2 + 0.05)
    ax.axis("off")
    fig.savefig(OUT / "fig_attribution.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_layout_scale()
    fig_records()
    fig_scale()
    fig_a100()
    fig_attribution()
    print(f"wrote 5 figures to {OUT}")
    print("numbers not read from a CSV:")
    for what, where in HARDCODED:
        print(f"  {what}  <-  {where}")


if __name__ == "__main__":
    main()
