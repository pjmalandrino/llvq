#!/usr/bin/env python3
"""Sensitivity of the perplexity knee to the 4B calibration draw.

Writes docs/data/knee-seeds.csv from the per-window NLL journals already in
the repository: the three-arm 4B campaign (f16, AWQ, LLVQ sealed), the 8B and
14B campaigns (same three arms, same 12 windows, fingerprint
3f1baca9033bf251), and the three F5 re-draws of the 4B (docs/data/f5-nll/).

The test is the one of docs/mesures/ppl-appariee-4b-2026-08-17.txt, replayed
with each F5 artifact in place of the published 4B: per window, the log ratio
LLVQ/f16 at each size; step1 = log ratio(8B) - log ratio(4B); step2 =
log ratio(14B) - log ratio(8B); knee = step1 - step2; Student t on 11 degrees
of freedom (t_{0.975,11} = 2.200985). The 'published' row must reproduce
ppl-genou.csv's knee (-0.100991706, t = -6.06) to the digit; the script
refuses to write otherwise.
"""

import csv
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T11 = 2.200985


def nll_blocks(path: Path) -> list[list[float]]:
    blocks, cur = [], []
    for line in path.read_text().splitlines():
        m = re.search(r"window\s+(\d+)/12\s+nll\s+([\d.]+)", line)
        if m:
            cur.append(float(m.group(2)))
            if int(m.group(1)) == 12:
                blocks.append(cur)
                cur = []
    return blocks


def paired(d: list[float]) -> tuple[float, float, float, float, float]:
    n = len(d)
    m = sum(d) / n
    s = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) / math.sqrt(n)
    return m, s, m / s, m - T11 * s, m + T11 * s


def main() -> None:
    f16_4, _awq_4, llvq_4 = nll_blocks(
        ROOT / "docs/mesures/a4-campagne-4b-ppl-BRUT-2026-08-06.txt"
    )
    f16_8, _awq_8, llvq_8 = nll_blocks(
        ROOT / "docs/mesures/campagne-8b-qualite-2026-08-08.txt"
    )
    # The 14B journal prints its arms in the order AWQ, f16, LLVQ.
    _awq_14, f16_14, llvq_14 = nll_blocks(
        ROOT / "docs/mesures/campagne-14b-qualite-2026-08-10.txt"
    )
    arms = [("published", llvq_4)] + [
        (f"seed{i}", nll_blocks(ROOT / f"docs/data/f5-nll/seed{i}-nll.txt")[0])
        for i in (1, 2, 3)
    ]
    r8 = [l - f for l, f in zip(llvq_8, f16_8)]
    r14 = [l - f for l, f in zip(llvq_14, f16_14)]
    rows = []
    for name, l4 in arms:
        r4 = [l - f for l, f in zip(l4, f16_4)]
        step1 = [b - a for a, b in zip(r4, r8)]
        step2 = [c - b for b, c in zip(r8, r14)]
        knee = [a - b for a, b in zip(step1, step2)]
        m1, _, t1, lo1, hi1 = paired(step1)
        m2, _, t2, lo2, hi2 = paired(step2)
        mk, _, tk, lok, hik = paired(knee)
        rows.append(
            {
                "arm_4b": name,
                "ppl_4b": f"{math.exp(sum(l4) / len(l4)):.4f}",
                "step1_logratio": f"{m1:.4f}",
                "step1_t": f"{t1:.2f}",
                "step2_logratio": f"{m2:.4f}",
                "step2_t": f"{t2:.2f}",
                "knee_logratio": f"{mk:.4f}",
                "knee_lo": f"{lok:.4f}",
                "knee_hi": f"{hik:.4f}",
                "knee_t": f"{tk:.2f}",
                "knee_excludes_zero": "yes" if (lok > 0 or hik < 0) else "no",
                "source": (
                    "mesures/a4-campagne-4b-ppl-BRUT-2026-08-06.txt"
                    if name == "published"
                    else f"data/f5-nll/{name}-nll.txt (mesures/f5-graines-4b-2026-08-19.txt)"
                ),
            }
        )
    pub = rows[0]
    if pub["knee_logratio"] != "-0.1010" or pub["knee_t"] != "-6.06":
        sys.exit(
            f"published knee does not replay ppl-genou.csv: {pub['knee_logratio']} t={pub['knee_t']}"
        )
    out = ROOT / "docs/data/knee-seeds.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out.relative_to(ROOT)}: {len(rows)} rows")
    for r in rows:
        print(
            f"  {r['arm_4b']:10s} ppl {r['ppl_4b']}  step1 {r['step1_logratio']} (t {r['step1_t']})"
            f"  knee {r['knee_logratio']} [{r['knee_lo']}, {r['knee_hi']}] t {r['knee_t']}"
        )


if __name__ == "__main__":
    main()
