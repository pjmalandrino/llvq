"""Read the duel's raw outputs into one table: size, speed, peak VRAM, perplexity.

    python3 ops/duel/analyze.py <raw dir>

<raw dir> holds `4b/`, `8b/`, `14b/` (the llama.cpp jobs) and `tetra/` (our side), as the bucket
`duel-iq2-2026-11-08/` lays them out. Quality on the census (MMLU, GSM8K) is paired by `bin/mmlupair` and
`bin/gsm8kpair`, not here.

Perplexity is paired window by window against our sealed files' reading of 2026-10-04
(`docs/mesures/ppl-scelles-2026-10-04-brut/ppl-<size>.txt`): same 12 windows of 4,096 tokens, every
position scored on both sides. The ratio is exp(mean of the per-window NLL differences), its interval
the paired t at 11 degrees of freedom.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from pathlib import Path

SIZES = {"4b": 4022468096, "8b": 8190735360, "14b": 14768307200}
OUR_BYTES = {"4b": 1418224685, "8b": 2815098745, "14b": 5087000541}
OUR_PPL_DIR = Path(__file__).resolve().parents[2] / "docs/mesures/ppl-scelles-2026-10-04-brut"
T95_DF11 = 2.200985
WIN = re.compile(r"window +(\d+)/12 +nll ([0-9.]+)")


def windows(path: Path) -> list[float]:
    got = {int(m.group(1)): float(m.group(2)) for m in WIN.finditer(path.read_text())}
    assert sorted(got) == list(range(1, 13)), f"{path}: windows {sorted(got)}"
    return [got[i] for i in range(1, 13)]


def paired(ours: list[float], theirs: list[float]) -> str:
    d = [t - o for o, t in zip(ours, theirs)]
    m, sd = statistics.mean(d), statistics.stdev(d)
    h = T95_DF11 * sd / math.sqrt(len(d))
    return f"x{math.exp(m):.4f} [{math.exp(m - h):.4f} ; {math.exp(m + h):.4f}]"


def mib_gb(mib: float) -> float:
    return mib * 1048576 / 1e9


def vram_rows(path: Path) -> dict[str, tuple[int, int]]:
    out = {}
    for line in path.read_text().splitlines():
        m = re.match(r"vram (\S+) idle (\d+) MiB peak (\d+) MiB", line)
        if m:
            out[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return out


def main(root: Path) -> int:
    t = root / "tetra"
    tv = vram_rows(t / "vram.txt") if (t / "vram.txt").exists() else {}
    for size, params in SIZES.items():
        d = root / size
        if not d.exists():
            print(f"== {size}: no output")
            continue
        print(f"== {size} ({params:,} parameters)")
        gate = (d / "gate-f16.txt").read_text().strip() if (d / "gate-f16.txt").exists() else "gate missing"
        print("  " + gate)
        rung = [l for l in (d / "rungs.txt").read_text().splitlines() if l.startswith("rung")]
        print("  " + (rung[0] if rung else "rung missing"))
        ours_w = windows(OUR_PPL_DIR / f"ppl-{size}.txt")
        ours_ppl = math.exp(statistics.mean(ours_w))
        vr = vram_rows(d / "vram.txt") if (d / "vram.txt").exists() else {}
        speed = (d / "speed.txt").read_text() if (d / "speed.txt").exists() else ""
        hdr = f"  {'arm':22} {'b/param':>8} {'file GB':>8} {'peak GB d5':>10} {'peak GB d8k':>11} " \
              f"{'tok/s d5 [range]':>22} {'tok/s d8k [range]':>22} {'ppl':>8}  ppl / Tetra (paired)"
        print(hdr)
        for line in (d / "sizes.txt").read_text().splitlines():
            m = re.match(r"size (\S+) (\d+) B", line)
            if not m:
                continue
            arm, b = m.group(1), int(m.group(2))
            sp = {}
            for dep in ("5", "8192"):
                s = re.search(rf"speed {re.escape(arm)} depth {dep}: tg256 ([0-9.]+) tok/s \[([0-9.]+)-([0-9.]+)\]", speed)
                sp[dep] = f"{s.group(1)} [{s.group(2)}-{s.group(3)}]" if s else "-"
            pk = {dep: vr.get(f"bench-{arm}-d{dep}") for dep in ("5", "8192")}
            pks = {k: f"{mib_gb(v[1] - v[0]):.2f}" if v else "-" for k, v in pk.items()}
            pf = d / f"ppl-{arm}.err"
            if pf.exists():
                w = windows(pf)
                ppl, pr = f"{math.exp(statistics.mean(w)):.4f}", paired(ours_w, w)
            else:
                ppl, pr = "-", "-"
            print(f"  {arm:22} {b * 8 / params:8.4f} {b / 1e9:8.3f} {pks['5']:>10} {pks['8192']:>11} "
                  f"{sp['5']:>22} {sp['8192']:>22} {ppl:>8}  {pr}")
        # our side
        sp = {}
        for dep in ("5", "8192"):
            f = t / f"sweep-{size}-p{dep}.txt"
            s = re.search(rf"^ +{dep} +[0-9.]+ \[[^\]]+\] +([0-9.]+) \[([0-9.]+)[^0-9]+([0-9.]+)\]",
                          f.read_text(), re.M) if f.exists() else None
            sp[dep] = f"{s.group(1)} [{s.group(2)}-{s.group(3)}]" if s else "-"
        pks = {}
        for dep in ("5", "8192"):
            v = tv.get(f"tetra-{size}-p{dep}")
            pks[dep] = f"{mib_gb(v[1] - v[0]):.2f}" if v else "-"
        b = OUR_BYTES[size]
        print(f"  {'tetra-v9 (ours)':22} {b * 8 / params:8.4f} {b / 1e9:8.3f} {pks['5']:>10} {pks['8192']:>11} "
              f"{sp['5']:>22} {sp['8192']:>22} {ours_ppl:8.4f}  (2026-10-04)")
        bt = d / "batched.txt"
        if bt.exists():
            print("  " + bt.read_text().strip().replace("\n", "\n  "))
    print("peak GB = sampled memory.used peak minus idle, 100 ms, 1 GB = 1e9 B (measured); "
          "b/param = file bytes x 8 / parameters (computed)")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
