#!/usr/bin/env python3
"""Figures of the r/LocalLLaMA post. Every number is copied from a journal named beside it."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

OUT = Path(__file__).parent / "fig"
TETRA, IQ2S, AWQ, UD, FP16 = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#8A8A8A"
INK, MUTED = "#1A1A1A", "#666666"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#BBBBBB",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False,
})
SIZES = ["Qwen3-4B", "Qwen3-8B", "Qwen3-14B"]

def save(fig, name, note, y=-0.04):
    fig.text(0.01, y, note, fontsize=8.5, color=MUTED, ha="left", va="bottom")
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)

# --- 1. MMLU vs bits per parameter (ETAT.md section 3; duel-iq2-2026-11-08) -----------------
# b/param: whole model, file bytes x 8 / params, embedding tables included.
arms = {  # size: [(label, b/param, MMLU, colour, marker)]
    "Qwen3-4B":  [("FP16", 16.00, 70.14, FP16, "s"), ("AWQ 4-bit", 5.30, 68.14, AWQ, "D"),
                  ("IQ2_S", 2.8188, 54.29, IQ2S, "^"), ("Unsloth UD-IQ2_XXS", 2.4967, 36.03, UD, "v"),
                  ("Tetra (ours)", 2.8206, 63.37, TETRA, "o")],
    "Qwen3-8B":  [("FP16", 16.00, 75.05, FP16, "s"), ("AWQ 4-bit", 5.96, 73.79, AWQ, "D"),
                  ("IQ2_S", 2.7980, 65.45, IQ2S, "^"), ("Unsloth UD-IQ2_XXS", 2.5446, 60.96, UD, "v"),
                  ("Tetra (ours)", 2.7495, 69.58, TETRA, "o")],
    "Qwen3-14B": [("FP16", 16.00, 78.88, FP16, "s"), ("AWQ 4-bit", 5.40, 78.12, AWQ, "D"),
                  ("IQ2_S", 2.6886, 72.77, IQ2S, "^"), ("Unsloth UD-IQ2_XXS", 2.4215, 67.78, UD, "v"),
                  ("Tetra (ours)", 2.7556, 75.66, TETRA, "o")],
}
fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
for ax, s in zip(axs, SIZES):
    for lab, b, m, c, mk in arms[s]:
        ax.scatter(b, m, s=110 if "ours" in lab else 70, color=c, marker=mk, zorder=3,
                   edgecolor="white", linewidth=1.5)
        dx, ha = (1.06, "left")
        if lab.startswith("Unsloth"):
            dx = 1.06
        ax.annotate(f"{lab}\n{m:.1f}", (b, m), xytext=(b * dx, m), fontsize=8.5, va="center", ha=ha,
                    color=INK, fontweight="bold" if "ours" in lab else "normal")
    ax.set_xscale("log"); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([2.5, 3, 4, 5, 8, 16]); ax.set_xticklabels(["2.5", "3", "4", "5", "8", "16"])
    ax.set_xlim(2.25, 30)
    ax.set_title(s); ax.set_xlabel("bits per parameter, whole model (log)")
axs[0].set_ylabel("MMLU, 5-shot, all 14,042 questions (%)")
fig.suptitle("MMLU vs bits per parameter. Up and left is better", x=0.01, ha="left", fontsize=14, fontweight="bold")
save(fig, "01_mmlu_vs_bits.png",
     "Measured on one L40S. Tetra MMLU on the dense reconstruction of the sealed file; GGUF through llama-server; "
     "AWQ and FP16 through vLLM. Unsloth UD-IQ2_XXS 4B loops under greedy decoding.")

# --- 2. Same bits: Tetra vs IQ2_S vs UD-IQ2_XXS (duel-iq2-2026-11-08) ----------------------
duel = {  # metric: {arm: [4B, 8B, 14B]}
    "MMLU (%)  higher is better": {"Tetra": [63.37, 69.58, 75.66], "IQ2_S": [54.29, 65.45, 72.77],
                                   "UD-IQ2_XXS": [36.03, 60.96, 67.78]},
    "GSM8K (%)  higher is better": {"Tetra": [82.49, 88.63, 92.04], "IQ2_S": [68.23, 85.60, 91.89],
                                    "UD-IQ2_XXS": [9.17, 70.51, 85.52]},
    "WikiText-2 perplexity  lower is better": {"Tetra": [12.58, 9.71, 8.44], "IQ2_S": [16.78, 10.75, 9.47],
                                               "UD-IQ2_XXS": [34.51, 13.44, 10.47]},
}
cols = {"Tetra": TETRA, "IQ2_S": IQ2S, "UD-IQ2_XXS": UD}
names = {"Tetra": "Tetra (ours) 2.75-2.82 b/p", "IQ2_S": "IQ2_S imatrix 2.69-2.82 b/p",
         "UD-IQ2_XXS": "Unsloth UD-IQ2_XXS 2.42-2.54 b/p"}
fig, axs = plt.subplots(1, 3, figsize=(13, 4.4))
w = 0.26
for ax, (metric, d) in zip(axs, duel.items()):
    for i, (arm, vals) in enumerate(d.items()):
        xs = [k + (i - 1) * w for k in range(3)]
        bars = ax.bar(xs, vals, width=w - 0.03, color=cols[arm], label=names[arm])
        for x, v in zip(xs, vals):
            ax.text(x, v, f"{v:.1f}" if v < 100 else f"{v:.0f}", ha="center", va="bottom", fontsize=7.5,
                    color=INK)
    ax.set_xticks(range(3)); ax.set_xticklabels(["4B", "8B", "14B"]); ax.set_title(metric, fontsize=11)
    ax.grid(axis="x", visible=False)
axs[2].set_ylim(0, 38)
fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.01))
axs[0].set_ylim(0, 85); axs[1].set_ylim(0, 100)
fig.suptitle("Around 2.7 bits per parameter, same day, same L40S", x=0.01, ha="left", fontsize=14,
             fontweight="bold")
save(fig, "02_same_bits_quality.png",
     "Measured. GSM8K: all 1,319 problems, zero-shot, greedy, 1,024 new tokens max. Perplexity: 12 windows of 4,096. "
     "llama.cpp cd959bae9, IQ2_S imatrix on dclm-edu. 14B GSM8K vs IQ2_S: not separated (+0.15 [-1.27, +1.58]).", y=-0.1)

# --- 3. One size up at about the same memory (sizeup-2026-09-28) --------------------------
pairs = [("AWQ 4-bit\nQwen3-4B", 2.67, 68.14, 89.01, AWQ), ("Tetra\nQwen3-8B", 2.76, 69.58, 88.63, TETRA),
         ("AWQ 4-bit\nQwen3-8B", 6.10, 73.79, 92.95, AWQ), ("Tetra\nQwen3-14B", 5.04, 75.66, 92.04, TETRA)]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
for ax, idx, title, lo in [(axs[0], 2, "MMLU (%)", 0), (axs[1], 3, "GSM8K (%)", 0)]:
    xs = [0, 0.9, 2.3, 3.2]
    for x, p in zip(xs, pairs):
        ax.bar(x, p[idx], width=0.75, color=p[4])
        ax.text(x, p[idx] + 0.2, f"{p[idx]:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(x, 3, f"{p[1]:.2f} GB", ha="center", va="bottom", fontsize=8.5, color="white")
    ax.set_xticks(xs); ax.set_xticklabels([p[0] for p in pairs], fontsize=9)
    ax.set_ylim(0, 108); ax.set_title(title); ax.grid(axis="x", visible=False)
axs[0].text(0.45, 76, "+1.44 [+0.71, +2.19]", ha="center", fontsize=9, color=MUTED)
axs[0].text(2.75, 82, "+1.87 [+1.20, +2.57]", ha="center", fontsize=9, color=MUTED)
axs[1].text(0.45, 96, "tie, -0.38 [-2.14, +1.39]", ha="center", fontsize=9, color=MUTED)
axs[1].text(2.75, 100, "tie, -0.91 [-2.35, +0.53]", ha="center", fontsize=9, color=MUTED)
fig.suptitle("Same memory, one size up: our 8B vs AWQ's 4B, our 14B vs AWQ's 8B", x=0.01, ha="left",
             fontsize=14, fontweight="bold")
save(fig, "03_size_up.png",
     "Measured, paired per question (95% intervals). Post hoc, not preregistered. GB = weights on the card for Tetra, "
     "file size for AWQ.")

# --- 4. Gap to FP16 shrinks with size (paper2-gaps.csv, paper2-gsm8k-gaps.csv) --------------
mm = [(6.77, 6.05, 7.50), (5.48, 4.83, 6.10), (3.22, 2.69, 3.75)]
gs = [(9.63, 7.69, 11.57), (4.62, 3.00, 6.25), (3.26, 1.93, 4.59)]
fig, ax = plt.subplots(figsize=(7.5, 4.3))
x = [0, 1, 2]
for vals, c, lab, off, mk in [(mm, TETRA, "MMLU", -0.08, "o"), (gs, IQ2S, "GSM8K", 0.08, "s")]:
    ax.errorbar([k + off for k in x], [v[0] for v in vals],
                yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], fmt=mk + "-", color=c,
                ms=8, lw=2, capsize=4, label=lab)
    for k, v in zip(x, vals):
        ax.text(k + off + (0.07 if off > 0 else -0.07), v[0], f"{v[0]:.1f}", fontsize=9, va="center",
                ha="left" if off > 0 else "right")
ax.set_xticks(x); ax.set_xticklabels(SIZES); ax.set_ylim(0, 12.5)
ax.set_ylabel("points lost vs FP16"); ax.legend()
fig.suptitle("What 2.7 bits costs, by size. Lower is better", x=0.01, ha="left", fontsize=14,
             fontweight="bold")
save(fig, "04_gap_to_fp16.png", "Measured, paired per question, 95% intervals. FP16 through vLLM.")

# --- 5. Decode speed, honest (duel-iq2-2026-11-08) -----------------------------------------
sp5 = {"Tetra": [227.4, 144.5, 90.0], "IQ2_S": [287.5, 208.4, 123.5], "UD-IQ2_XXS": [306.3, 222.6, 134.0]}
sp8k = {"Tetra": [160.8, 109.6, 71.7], "IQ2_S": [195.4, 154.2, 99.7], "UD-IQ2_XXS": [202.6, 162.3, 106.2]}
fig, axs = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
for ax, d, t in [(axs[0], sp5, "after a 5-token prompt"), (axs[1], sp8k, "after an 8,192-token prompt")]:
    for i, (arm, vals) in enumerate(d.items()):
        xs = [k + (i - 1) * w for k in range(3)]
        ax.bar(xs, vals, width=w - 0.03, color=cols[arm], label=names[arm].split(" ")[0] if arm != "UD-IQ2_XXS"
               else "Unsloth UD-IQ2_XXS")
        for xx, v in zip(xs, vals):
            ax.text(xx, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels(["4B", "8B", "14B"]); ax.set_title(t, fontsize=11)
    ax.grid(axis="x", visible=False)
axs[0].set_ylabel("decode tok/s, batch 1"); axs[0].legend(fontsize=9)
fig.suptitle("Decode speed: llama.cpp is faster. Higher is better", x=0.01, ha="left", fontsize=14,
             fontweight="bold")
save(fig, "05_speed.png",
     "Measured, one L40S, 256 new tokens, median of 5 rounds. Ours: own engine (one CUDA launch per token). "
     "llama.cpp: llama-bench, flash attention.")

# --- 6. Card memory (duel-iq2-2026-11-08; pic-vram-8k-2026-11-08) --------------------------
m5 = {"Tetra": [2.01, 3.42, 5.71], "IQ2_S": [2.43, 3.62, 5.73]}
m8k = {"Tetra": [5.95, 7.39, 10.54], "IQ2_S": [3.52, 4.71, 6.90]}
fig, axs = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
w2 = 0.36
for ax, d, t in [(axs[0], m5, "after a 5-token prompt (IQ2_S: card + host-held embedding)"),
                 (axs[1], m8k, "peak during an 8,192-token prompt (card only)")]:
    for i, (arm, vals) in enumerate(d.items()):
        xs = [k + (i - 0.5) * w2 for k in range(3)]
        ax.bar(xs, vals, width=w2 - 0.04, color=cols[arm], label="Tetra (ours)" if arm == "Tetra" else "IQ2_S")
        for xx, v in zip(xs, vals):
            ax.text(xx, v + 0.15, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(["4B", "8B", "14B"]); ax.set_title(t, fontsize=10.5)
    ax.grid(axis="x", visible=False)
axs[0].set_ylabel("GB"); axs[0].legend(fontsize=9)
fig.suptitle("Memory: a tie at short context, we lose on long prompts. Lower is better", x=0.01, ha="left",
             fontsize=14, fontweight="bold")
save(fig, "06_memory.png",
     "Measured, one L40S, peak minus idle. Our long-prompt peak comes from the prefill's attention scores, not "
     "from the weights.")
print("ok", sorted(p.name for p in OUT.iterdir()))
