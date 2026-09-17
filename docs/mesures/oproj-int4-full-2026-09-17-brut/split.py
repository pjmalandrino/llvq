# /// script
# requires-python = ">=3.10"
# ///
import hashlib
from pathlib import Path
FULL = Path.home() / "llvq-oproj-full-2026-09-16"; EXPL = Path.home() / "llvq-q5-alloc-2026-09-16"
OUT = Path("/tmp/claude-501/oproj")
def load(p):
    head, rows, trailer = [], [], None
    for l in p.read_text().splitlines():
        if l.startswith("# end"): trailer = l
        elif l.startswith("#") or l.startswith("subject,"): head.append(l)
        elif l:
            f = l.split(","); rows.append(((f[0], int(f[1])), f[3], l))
    return head, rows, trailer
hs, ship, ts = load(FULL / "mmlu-shipped-FULL.csv"); ho, oprj, _ = load(FULL / "mmlu-oproj-FULL.csv")
_, ex, _ = load(EXPL / "mmlu-shipped.csv")
sel = {k for k, _, _ in ex}; sel_hash = {h for _, h, _ in ex}
fp0 = ts.split("fingerprint=")[1].split()[0]
subsets = {
    "heldout": lambda k, h: k not in sel,
    "heldout-strict": lambda k, h: k not in sel and h not in sel_hash,
    "selection": lambda k, h: k in sel,
}
for name, keep in subsets.items():
    fp = hashlib.sha256((fp0 + name).encode()).hexdigest()[:16]
    for arm, head, rows in [("shipped", hs, ship), ("oproj", ho, oprj)]:
        body = [l for k, h, l in rows if keep(k, h)]
        (OUT / f"{arm}-{name}.csv").write_text("\n".join(head + body + [f"# end fingerprint={fp} questions={len(body)}"]) + "\n")
    print(f"{name}: {len(body)} questions")
