# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
"""One panel a metric of docs/data/inkling-stats.csv, the readings in order on the abscissa, a line a card.

    uv run ops/inkling/plot_stats.py [out.png]
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[2]
rows = list(csv.DictReader(open(root / "docs/data/inkling-stats.csv")))
series = defaultdict(lambda: defaultdict(list))
for r in rows:
    if r["config"].startswith("control:"):
        continue  # a reference arm read later than the path it precedes: kept in the file, not drawn
    series[r["metric"]][r["card"]].append((r["date"], int(r["seq"]), float(r["value"]), r["unit"], r["label"]))
metrics = sorted(series)
cols = 3
fig, axes = plt.subplots(-(-len(metrics) // cols), cols, figsize=(5 * cols, 3.2 * -(-len(metrics) // cols)))
for ax, m in zip(axes.flat, metrics):
    for card, pts in sorted(series[m].items()):
        pts.sort()
        xs = [f"{d[5:]}#{s}" for d, s, *_ in pts]
        ax.plot(xs, [p[2] for p in pts], marker="o", label=card)
        ax.set_ylabel(pts[0][3])
    ax.set_title(m, fontsize=10)
    ax.tick_params(axis="x", labelrotation=60, labelsize=7)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
for ax in list(axes.flat)[len(metrics):]:
    ax.axis("off")
fig.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else str(root / "docs/data/inkling-stats.png")
fig.savefig(out, dpi=110)
print(out)
