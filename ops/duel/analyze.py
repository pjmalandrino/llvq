"""Read the duel's raw outputs into one table a size: memory, speed, perplexity, MMLU, GSM8K, and the verdicts.

    python3 ops/duel/analyze.py <raw dir>

<raw dir> holds, as the bucket `duel-iq2-2026-11-08/` lays them out: `<size>/` (run 1: the gate, the rungs, the
GGUF sizes, and the perplexity of our IQ2_XXS), `<size>-r3/` (retry 3: IQ2_S and unsloth, every measure) and
`tetra/` (our speed and peak memory). A file is read from `<size>-r3/` first, then from `<size>/`.

Perplexity is paired window by window against our sealed files' reading of 2026-10-04
(`docs/mesures/ppl-scelles-2026-10-04-brut/ppl-<size>.txt`): same 12 windows of 4,096 tokens, every position scored
on both sides. The ratio is exp(mean of the per-window NLL differences), its interval the paired t at 11 degrees of
freedom. MMLU and GSM8K are paired by `target/release/mmlupair` and `gsm8kpair` against the census dumps the
preregistration names; MMLU's interval is the McNemar normal interval on the 14,042 pairs (the census bootstrap with
finite-population correction has zero width by construction).

Decision rule: section 6 of `proofs/preregistration-duel-iq2-2026-11-08.md`.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SIZES = {"4b": 4022468096, "8b": 8190735360, "14b": 14768307200}
OUR_BYTES = {"4b": 1418224685, "8b": 2815098745, "14b": 5087000541}
OUR_PPL_DIR = REPO / "docs/mesures/ppl-scelles-2026-10-04-brut"
OUR_MMLU = {
    "4b": "docs/mesures/embed-q4-swap-2026-09-23-brut/mmlu-4b-embed-q4-swap-FULL.csv",
    "8b": "docs/mesures/sealed-8b-27-2026-09-24-brut/mmlu-B/mmlu-8b-B-sealed-FULL.csv",
    "14b": "docs/mesures/sealed-8b-14b-2026-09-23-brut/mmlu-14b/mmlu-14b-sealed-FULL.csv",
}
OUR_GSM8K = {
    "4b": "docs/mesures/gsm8k-wave1-2026-09-26-brut/run1-served/gsm8k-4b-served.jsonl",
    "8b": "docs/mesures/gsm8k-wave2-2026-09-26-brut/run2-served-8b/gsm8k-8b-served.jsonl",
    "14b": "docs/mesures/gsm8k-wave2-2026-09-26-brut/run3-served-14b/gsm8k-14b-served.jsonl",
}
T95_DF11 = 2.200985
Z95 = 1.959964
WIN = re.compile(r"window +(\d+)/12 +nll ([0-9.]+)")
ARMS = ["own-iq2_s", "unsloth-ud-iq2_xxs", "own-iq2_xxs"]


def windows(path: Path) -> list[float]:
    got = {int(m.group(1)): float(m.group(2)) for m in WIN.finditer(path.read_text())}
    assert sorted(got) == list(range(1, 13)), f"{path}: windows {sorted(got)}"
    return [got[i] for i in range(1, 13)]


def mib_gb(mib: float) -> float:
    return mib * 1048576 / 1e9


def vram_rows(path: Path) -> dict[str, tuple[int, int]]:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            m = re.match(r"vram (\S+) idle (\d+) MiB peak (\d+) MiB", line)
            if m:
                out[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return out


def bench_json_speed(path: Path) -> tuple[float, float, float] | None:
    """Median and range of a llama-bench JSON's `samples_ts`. Run 1's bench file carries the stray peak line of D2
    after the array, so the array is cut out first."""
    if not path.exists():
        return None
    t = path.read_text()
    i, j = t.find("["), t.rfind("]")
    if i < 0 or j < i:
        return None
    ts = json.loads(t[i:j + 1])[0]["samples_ts"]
    return statistics.median(ts), min(ts), max(ts)


def host_gb(path: Path | None) -> float | None:
    """The model bytes llama.cpp keeps in host memory (`CPU_Mapped model buffer`), outside the card's peak."""
    if path is None or not path.exists():
        return None
    m = re.search(r"CPU_Mapped model buffer size = +([0-9.]+) MiB", path.read_text())
    return mib_gb(float(m.group(1))) if m else None


def find(dirs: list[Path], name: str) -> Path | None:
    for d in dirs:
        if (d / name).exists():
            return d / name
    return None


def verdict_interval(lo: float, hi: float, ours_better_if_positive: bool) -> str:
    if not ours_better_if_positive:
        lo, hi = -hi, -lo
    return "Tetra" if lo > 0 else "llama.cpp" if hi < 0 else "not separated"


def run(*args: str) -> str:
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True, check=True).stdout


def mmlu(size: str, dump: Path) -> tuple[float, float, str]:
    out = run(str(REPO / "target/release/mmlupair"), OUR_MMLU[size], str(dump))
    m = re.search(r"stratified micro +A +([0-9.]+) % +B +([0-9.]+) %", out)
    b = re.search(r"A✓B✗ = (\d+) +A✗B✓ = (\d+) +concordant = (\d+)", out)
    a_acc, b_acc = float(m.group(1)), float(m.group(2))
    nb, nc, nn = int(b.group(1)), int(b.group(2)), int(b.group(3))
    n = nb + nc + nn
    d = (nb - nc) / n
    se = math.sqrt(nb + nc - (nb - nc) ** 2 / n) / n
    lo, hi = 100 * (d - Z95 * se), 100 * (d + Z95 * se)
    gap = f"{a_acc - b_acc:+.2f} pp [{lo:+.2f} ; {hi:+.2f}]"
    return a_acc, b_acc, f"{gap} {verdict_interval(lo, hi, True)}"


def gsm8k(size: str, dump: Path) -> tuple[float, float, str, int]:
    out = run(str(REPO / "target/release/gsm8kpair"), OUR_GSM8K[size], str(dump))
    acc = re.search(r"accuracy A = ([0-9.]+) %, B = ([0-9.]+) %", out)
    gap = re.search(r"A − B = ([+-][0-9.]+) pp, 95 % \[([+-][0-9.]+) ; ([+-][0-9.]+)\]", out)
    cap = re.findall(r"(\d+) stopped at the cap", out)
    lo, hi = float(gap.group(2)), float(gap.group(3))
    txt = f"{gap.group(1)} pp [{gap.group(2)} ; {gap.group(3)}] {verdict_interval(lo, hi, True)}"
    return float(acc.group(1)), float(acc.group(2)), txt, int(cap[1])


def speed_verdict(ours: tuple[float, float, float] | None, theirs: tuple[float, float, float] | None) -> str:
    if not ours or not theirs:
        return "-"
    if ours[1] > theirs[2]:
        return "Tetra"
    if ours[2] < theirs[1]:
        return "llama.cpp"
    return "not separated"


def lower_wins(ours: float, theirs: float, tie: float) -> str:
    return "Tetra" if theirs - ours > tie else "llama.cpp" if ours - theirs > tie else "tie"


def main(root: Path) -> int:
    t = root / "tetra"
    tv = vram_rows(t / "vram.txt")
    for size, params in SIZES.items():
        dirs = [root / f"{size}-r3", root / size]
        if not dirs[1].exists():
            print(f"== {size}: no output")
            continue
        print(f"\n== {size} ({params:,} parameters)")
        for name in ("gate-f16.txt",):
            p = find(dirs, name)
            print("  " + (p.read_text().strip() if p else f"{name} missing"))
        rung = [line for line in find(dirs, "rungs.txt").read_text().splitlines() if line.startswith("rung")]
        print("  " + (rung[0] if rung else "rung missing"))

        # our side
        ours_w = windows(OUR_PPL_DIR / f"ppl-{size}.txt")
        osp: dict[str, tuple[float, float, float] | None] = {}
        for dep in ("5", "8192"):
            f = t / f"sweep-{size}-p{dep}.txt"
            s = re.search(rf"^ +{dep} +[0-9.]+ \[[^\]]+\] +([0-9.]+) \[([0-9.]+)[^0-9]+([0-9.]+)\]",
                          f.read_text(), re.M) if f.exists() else None
            osp[dep] = tuple(float(s.group(i)) for i in (1, 2, 3)) if s else None
        opk = {dep: mib_gb(v[1] - v[0]) if (v := tv.get(f"tetra-{size}-p{dep}")) else None for dep in ("5", "8192")}
        ob = OUR_BYTES[size]
        rows = [("tetra-v9 (ours)", ob, osp, opk, math.exp(statistics.mean(ours_w)), None)]

        sizes = {}
        for line in find(dirs, "sizes.txt").read_text().splitlines():
            m = re.match(r"size (\S+) (\d+) B", line)
            if m:
                sizes[m.group(1)] = int(m.group(2))
        verdicts = []
        for arm in ARMS:
            b = sizes[arm]
            sp: dict[str, tuple[float, float, float] | None] = {}
            spf = find(dirs, "speed.txt")
            txt = spf.read_text() if spf else ""
            for dep in ("5", "8192"):
                s = re.search(rf"speed {re.escape(arm)} depth {dep}: tg256 ([0-9.]+) tok/s "
                              rf"\[([0-9.]+)-([0-9.]+)\]", txt)
                sp[dep] = tuple(float(s.group(i)) for i in (1, 2, 3)) if s else None
            vr = vram_rows(find(dirs, "vram.txt") or Path("/nonexistent"))
            if arm == "own-iq2_xxs":
                # Deviation D2: run 1 stopped inside its first bench, after llama-bench had written its five rounds
                # at depth 5 and the sampler its peak; nothing else of this arm is measured.
                sp = {"5": bench_json_speed(dirs[1] / "bench-own-iq2_xxs-d5.json"), "8192": None}
                vr = vram_rows(dirs[1] / "vram.txt")
            pk = {dep: mib_gb(v[1] - v[0]) if (v := vr.get(f"bench-{arm}-d{dep}")) else None for dep in ("5", "8192")}
            pf = find(dirs, f"ppl-{arm}.err")
            w = windows(pf) if pf else None
            hm = host_gb(pf)
            rows.append((arm, b, sp, pk, math.exp(statistics.mean(w)) if w else None, w))

            v = [f"  vs {arm}:"]
            v.append(f"b/param {lower_wins(ob * 8 / params, b * 8 / params, 0.01)}"
                     f" ({ob * 8 / params:.4f} vs {b * 8 / params:.4f})")
            v.append(f"file GB {lower_wins(ob / 1e9, b / 1e9, 0.05)}")
            for dep in ("5", "8192"):
                if opk[dep] is not None and pk[dep] is not None:
                    v.append(f"peak d{dep} {lower_wins(opk[dep], pk[dep], 0.05)}"
                             f" ({opk[dep]:.2f} vs {pk[dep]:.2f} GB)")
                    if hm is not None:
                        v.append(f"peak d{dep} + host {lower_wins(opk[dep], pk[dep] + hm, 0.05)}"
                                 f" ({opk[dep]:.2f} vs {pk[dep]:.2f} + {hm:.2f} GB held in host memory)")
                if sp[dep]:
                    v.append(f"tok/s d{dep} {speed_verdict(osp[dep], sp[dep])}"
                             f" ({osp[dep][0]:.1f} vs {sp[dep][0]:.1f})")
            if w:
                d = [o - x for o, x in zip(ours_w, w)]
                m, sd = statistics.mean(d), statistics.stdev(d)
                h = T95_DF11 * sd / math.sqrt(12)
                v.append(f"ppl Tetra/arm x{math.exp(m):.4f} [{math.exp(m - h):.4f} ; {math.exp(m + h):.4f}] "
                         f"{verdict_interval(m - h, m + h, False)}")
            mf = find(dirs, f"mmlu-{size}-{arm}-FULL.csv")
            if mf:
                a, bb, g = mmlu(size, mf)
                v.append(f"MMLU {a:.2f} vs {bb:.2f}, Tetra - arm {g}")
            gf = find(dirs, f"gsm8k-{size}-{arm}.jsonl")
            if gf:
                a, bb, g, cap = gsm8k(size, gf)
                v.append(f"GSM8K {a:.2f} vs {bb:.2f}, Tetra - arm {g}; arm at the 1,024 cap {cap}/1319")
            verdicts.append("\n    ".join(v))

        print(f"  {'arm':20} {'b/param':>7} {'file GB':>7} {'peak d5':>7} {'peak d8k':>8} "
              f"{'tok/s d5 [range]':>21} {'GB/s d5':>7} {'tok/s d8k [range]':>21} {'GB/s d8k':>8} {'ppl':>8}")
        for arm, b, sp, pk, ppl, _ in rows:
            def f_sp(x):
                return f"{x[0]:.1f} [{x[1]:.1f}-{x[2]:.1f}]" if x else "-"

            def f_bw(x):
                return f"{b * x[0] / 1e9:.1f}" if x else "-"

            def f_pk(x):
                return f"{x:.2f}" if x is not None else "-"
            print(f"  {arm:20} {b * 8 / params:7.4f} {b / 1e9:7.3f} {f_pk(pk['5']):>7} {f_pk(pk['8192']):>8} "
                  f"{f_sp(sp['5']):>21} {f_bw(sp['5']):>7} {f_sp(sp['8192']):>21} {f_bw(sp['8192']):>8} "
                  f"{(f'{ppl:.4f}' if ppl else '-'):>8}")
        print("\n".join(verdicts))
        bt = find(dirs, "batched.txt")
        if bt:
            print("  " + bt.read_text().strip().replace("\n", "\n  "))
    print("\npeak GB = sampled memory.used peak minus idle, 100 ms, 1 GB = 1e9 B (measured); tok/s median of 5 "
          "[range] (measured); b/param = file bytes x 8 / parameters, GB/s = file bytes x tok/s (computed); "
          "host = llama.cpp's CPU_Mapped model buffer, read from the arm's perplexity log (measured), which the "
          "card's peak does not see; Tetra holds every byte on the card")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
