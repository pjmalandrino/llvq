# /// script
# requires-python = ">=3.10"
# ///
"""Split the two FULL dumps into the selection set and its complement.

The exploration of 2026-09-16 that selected down_proj scored 2,280 questions.
Those questions are what makes its +3.79 pp a selection and not a result, so
the primary is the complement: the 11,762 that took no part in the choice.
Same construction as the o_proj confirmation of 2026-09-17.
"""
import hashlib
from pathlib import Path

FULL = Path("docs/mesures/downproj-int4-full-2026-09-18-brut")
EXPL = Path("docs/data/mmlu-dumps/mmlu-shipped.csv")
OUT = Path("/tmp/downproj-split")
OUT.mkdir(parents=True, exist_ok=True)


def load(p):
    head, rows, trailer = [], [], None
    for l in p.read_text().splitlines():
        if l.startswith("# end"):
            trailer = l
        elif l.startswith("#") or l.startswith("subject,"):
            head.append(l)
        elif l:
            f = l.split(",")
            rows.append(((f[0], int(f[1])), f[3], l))
    return head, rows, trailer


hs, ship, ts = load(FULL / "mmlu-shipped-FULL.csv")
hd, down, _ = load(FULL / "mmlu-downproj-FULL.csv")
_, ex, _ = load(EXPL)
sel = {k for k, _, _ in ex}
sel_hash = {h for _, h, _ in ex}
fp0 = ts.split("fingerprint=")[1].split()[0]
subsets = {
    "heldout": lambda k, h: k not in sel,
    "heldout-strict": lambda k, h: k not in sel and h not in sel_hash,
    "selection": lambda k, h: k in sel,
}
for name, keep in subsets.items():
    fp = hashlib.sha256((fp0 + name).encode()).hexdigest()[:16]
    n = 0
    for arm, head, rows in [("shipped", hs, ship), ("downproj", hd, down)]:
        body = [l for k, h, l in rows if keep(k, h)]
        n = len(body)
        (OUT / f"{arm}-{name}.csv").write_text(
            "\n".join(head + body + [f"# end fingerprint={fp} questions={n}"]) + "\n"
        )
    print(f"{name}: {n} questions")
