# DRAFT, never run. Written read-only on 2026-11-02 for lead "bigmodel"
# (prereg-bigmodel-1.md §5-§7), revised the same day after review. Not part of
# the repository; meant to land as ops/k32_project.py in the commit carrying
# the prereg. Standard library only; run with `python3 -I` on paths passed as
# arguments, after review.
#
# What it computes, and on whose authority:
#   * micro and SE of every arm, with the HOUSE formula of bin/mmlu
#     (`micro_stderr`, mmlu.rs:397-415): w^2 * p(1-p)/(n-1) * (1 - n/N);
#   * paired same-model differences (E14 - f16-14B, ours-14B - R14), analytic
#     stratified SE with the same finite population correction. `mmlupair` is
#     AUTHORITATIVE for these (bootstrap, mmlupair.rs:537); this script
#     cross-checks them and is checked against mmlupair before use (`check`);
#   * cross-size quantities (R32 - AWQ14, Delta = P32 - (AWQ14 + 1.0)) with SEs
#     in quadrature, METHODE §3 ("mmlupair does not pair two models" across
#     sizes). This script is authoritative for those;
#   * the case of prereg §7: V, C, D or "stage B runs" after stage A; A, B or U
#     after stage B.
#
# FALLBACK (operator's word only): if ops/vllm_score.py does not gain the
# trailer, mmlupair cannot read the GGUF dumps and this script is
# authoritative for every quantity, after `check` passes.
#
# References, subset to the plan by (subject, index) with qhash checked:
#   ours-14B  docs/mesures/sealed-8b-14b-2026-09-23-brut/mmlu-14b/mmlu-14b-sealed-FULL.csv
#   awq-14B   docs/mesures/census-14b-ref-2026-09-22-brut/mmlu-14b-awq-FULL.csv
#   f16-14B   docs/data/mmlu-dumps/mmlu-14b-f16.csv (already on the plan)
# On 2026-11-02 (wave5-deep/calc/bigmodel/k32_numbers.py): micro 75.85 /
# 78.21 / 78.97, SE 1.23 / 1.20 / 1.19; paired ours - AWQ -2.36 (SE 0.92).
#
# Usage:
#   python3 -I k32_project.py check  PLAN AWQ_CENSUS F16_PLAN
#   python3 -I k32_project.py stageA PLAN OURS_CENSUS AWQ_CENSUS F16_PLAN E14 R14
#   python3 -I k32_project.py stageB PLAN OURS_CENSUS AWQ_CENSUS F16_PLAN E14 R14 R32
# PLAN is docs/data/mmlu-dumps/mmlu-4b-f16.csv (keys, qhash, order).

import csv
import math
import sys
from collections import defaultdict

PLAN_FINGERPRINT = "65dcd53655e8bfa5"   # trailer of both plan dumps (4B, 14B)
ENGINE_PP = 0.6                          # gate 3, paired micro
ENGINE_PICKS = 46                        # gate 3, 2 % of 2,280 (9 at 4B)
CLAIM_MARGIN = 1.0                       # plan D: AWQ 14B + 1.0
Z = 1.96


def load(path):
    rows, trailer = {}, None
    with open(path, encoding="utf-8") as f:
        body = []
        for line in f:
            if line.startswith("# end "):
                kv = dict(x.split("=", 1) for x in line[len("# end "):].split())
                trailer = (kv.get("fingerprint"), int(kv.get("questions", "-1")))
            elif not line.startswith("#"):
                body.append(line)
    for r in csv.DictReader(body):
        key = (r["subject"], int(r["index"]))
        if key in rows:
            raise SystemExit(f"{path}: {key} twice")
        rows[key] = r
    return rows, trailer


def on_plan(name, rows, plan):
    bad = [k for k in plan if k not in rows or rows[k]["qhash"] != plan[k]["qhash"]]
    if bad:
        raise SystemExit(f"{name}: {len(bad)} plan rows missing or with another qhash; first {bad[0]}")


def micro_se(rows, keys):
    by, pop = defaultdict(list), {}
    for k in keys:
        by[k[0]].append(int(rows[k]["correct"]))
        pop[k[0]] = int(rows[k]["population"])
    n_tot = sum(pop.values())
    acc = sum(pop[s] * sum(v) / len(v) for s, v in by.items()) / n_tot
    var = 0.0
    for s, v in by.items():
        n, p = len(v), sum(v) / len(v)
        if n > 1:
            var += (pop[s] / n_tot) ** 2 * p * (1 - p) / (n - 1) * (1 - n / pop[s])
    return 100 * acc, 100 * math.sqrt(var)


def paired(a, b, keys):
    """a - b, micro, analytic stratified SE with the finite population correction."""
    by, pop = defaultdict(list), {}
    for k in keys:
        by[k[0]].append(int(a[k]["correct"]) - int(b[k]["correct"]))
        pop[k[0]] = int(a[k]["population"])
    n_tot = sum(pop.values())
    d = sum(pop[s] * sum(v) / len(v) for s, v in by.items()) / n_tot
    var = 0.0
    for s, v in by.items():
        n = len(v)
        if n > 1:
            mu = sum(v) / n
            var += (pop[s] / n_tot) ** 2 * sum((x - mu) ** 2 for x in v) / (n - 1) / n * (1 - n / pop[s])
    picks = sum(1 for k in keys if a[k]["pick"] != b[k]["pick"])
    return 100 * d, 100 * math.sqrt(var), picks


def gguf_trailer_ok(name, trailer):
    if trailer is None:
        print(f"  {name}: NO `# end` trailer: mmlupair will refuse this dump (fallback rules apply)")
        return False
    fp, q = trailer
    if fp != PLAN_FINGERPRINT or q != 2280:
        raise SystemExit(f"{name}: trailer {fp} / {q}, the plan's is {PLAN_FINGERPRINT} / 2280 (gate 4)")
    return True


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__ or "see the header")
    mode, paths = argv[1], argv[2:]
    plan, _ = load(paths[0])
    keys = list(plan.keys())
    if len(keys) != 2280:
        raise SystemExit(f"plan holds {len(keys)} questions, expected 2,280")

    if mode == "check":
        awq, _ = load(paths[1])
        f16, _ = load(paths[2])
        on_plan("awq census", awq, plan)
        on_plan("f16 plan", f16, plan)
        d, se, picks = paired(awq, f16, keys)
        print(f"awq - f16 on the plan: {d:+.3f} pp, SE {se:.3f}, picks different {picks}")
        print("Compare with: mmlupair --intersect <awq census> docs/data/mmlu-dumps/mmlu-14b-f16.csv")
        print("This script may be used only if estimate and SE agree within 0.05 pp (prereg §3 item 2).")
        return

    ours, _ = load(paths[1])
    awq, _ = load(paths[2])
    f16, _ = load(paths[3])
    e14, t_e14 = load(paths[4])
    r14, t_r14 = load(paths[5])
    arms = [("ours-14B", ours), ("awq-14B", awq), ("f16-14B", f16), ("E14 BF16", e14), ("R14 IQ2_M", r14)]
    r32 = t_r32 = None
    if mode == "stageB":
        r32, t_r32 = load(paths[6])
        arms.append(("R32 IQ2_M", r32))
    elif mode != "stageA":
        raise SystemExit(f"unknown mode {mode}")

    se_of = {}
    for name, rows in arms:
        on_plan(name, rows, plan)
        m, se = micro_se(rows, keys)
        se_of[name] = (m, se)
        print(f"{name:10s} micro {m:6.2f}  SE {se:4.2f}")
    ready = all(gguf_trailer_ok(n, t) for n, t in [("E14", t_e14), ("R14", t_r14)] + ([("R32", t_r32)] if r32 else []))
    print("mmlupair can read the GGUF dumps" if ready else "FALLBACK: this script is authoritative (after `check`)")

    # Gate 3, the engine at 14B.
    eng, eng_se, eng_picks = paired(e14, f16, keys)
    gate3 = abs(eng) <= ENGINE_PP and eng_picks <= ENGINE_PICKS
    print(f"engine  E14 - f16      {eng:+.2f} (SE {eng_se:.2f}), picks different {eng_picks}"
          f"  gate: |d| <= {ENGINE_PP} and picks <= {ENGINE_PICKS} -> {'pass' if gate3 else 'FAIL'}")

    # D_ours, paired; mmlupair --intersect <ours census> <R14> is authoritative.
    d, d_se, d_picks = paired(ours, r14, keys)
    lo, hi = d - Z * d_se, d + Z * d_se
    print(f"D_ours  ours - R14     {d:+.2f} (SE {d_se:.2f})  95% [{lo:+.2f}, {hi:+.2f}]  picks different {d_picks}")

    if not gate3:
        print("case V: no statement about our format; stage B does not run")
        return
    if hi < 0:
        print("case D: the rival rung beats our sealed 14B; stage B does not run")
        return
    if lo <= 0:
        print("case C: not separated; stage B does not run")
        return
    if mode == "stageA":
        print("stage A done: lo > 0, so stage B (R32) runs, on the operator's word")
        return

    # Stage B: Delta = P32 - (AWQ14 + 1.0), SEs in quadrature (METHODE §3).
    m32, s32 = se_of["R32 IQ2_M"]
    ma, sa = se_of["awq-14B"]
    size, size_se = m32 - ma, math.hypot(s32, sa)
    p32 = m32 + d
    delta = p32 - (ma + CLAIM_MARGIN)
    delta_se = math.sqrt(s32 ** 2 + sa ** 2 + d_se ** 2)
    dlo, dhi = delta - Z * delta_se, delta + Z * delta_se
    print(f"size    R32 - AWQ14    {size:+.2f} (SE {size_se:.2f}, quadrature)")
    print(f"P32 = R32 + D_ours     {p32:.2f}")
    print(f"Delta = P32 - (AWQ14 + {CLAIM_MARGIN})  {delta:+.2f} (SE {delta_se:.2f})  95% [{dlo:+.2f}, {dhi:+.2f}]")
    if dlo > 0:
        print("case A: propose the 14B->32B gap gate and pricing the 32B chain (operator)")
    elif dhi < 0:
        print("case B: the projected 32B misses AWQ14 + 1.0: no 32B encode on this evidence")
    else:
        print("case U: unresolved on the plan. Operator: R32 on the full split (about 6.2x its plan time),"
              " R14 and R32 on the full split, or stop")


if __name__ == "__main__":
    main(sys.argv)
