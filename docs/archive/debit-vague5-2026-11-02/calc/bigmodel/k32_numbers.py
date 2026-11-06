# Read-only computation for the K32 revision (wave5-deep/bigmodel.md, prereg-bigmodel-1.md).
# Standard library only; run with python3 -I <repo>. Reads repository dumps, writes nothing.
# SE "fpc" follows llvq-llm/src/bin/mmlu.rs micro_stderr: w^2 * p(1-p)/(n-1) * (1 - n/N).
# Paired SE "fpc" is the analytic counterpart of mmlupair's bootstrap: w^2 * s_d^2/n * (1 - n/N).
import csv, math, sys
from collections import defaultdict

R = sys.argv[1]

def load(p):
    with open(p) as f:
        lines = [l for l in f if not l.startswith('#')]
    return {(r['subject'], int(r['index'])): r for r in csv.DictReader(lines)}

def strata(rows, keys, col=lambda r: int(r['correct'])):
    by, pop = defaultdict(list), {}
    for k in keys:
        by[k[0]].append(col(rows[k])); pop[k[0]] = int(rows[k]['population'])
    return by, pop

def micro(rows, keys, fpc):
    by, pop = strata(rows, keys)
    N = sum(pop.values())
    acc = sum(pop[s] * sum(v) / len(v) for s, v in by.items()) / N
    var = 0.0
    for s, v in by.items():
        n, p = len(v), sum(v) / len(v)
        if n <= 1: continue
        f = (1 - n / pop[s]) if fpc else 1.0
        var += (pop[s] / N) ** 2 * p * (1 - p) / (n - 1) * f
    return 100 * acc, 100 * math.sqrt(var)

def paired(a, b, keys, fpc):
    by, pop = defaultdict(list), {}
    for k in keys:
        by[k[0]].append(int(a[k]['correct']) - int(b[k]['correct'])); pop[k[0]] = int(a[k]['population'])
    N = sum(pop.values())
    d = sum(pop[s] * sum(v) / len(v) for s, v in by.items()) / N
    var = 0.0
    for s, v in by.items():
        n = len(v); mu = sum(v) / n
        if n <= 1: continue
        f = (1 - n / pop[s]) if fpc else 1.0
        var += (pop[s] / N) ** 2 * sum((x - mu) ** 2 for x in v) / (n - 1) / n * f
    disc = sum(1 for k in keys if a[k]['correct'] != b[k]['correct'])
    picks = sum(1 for k in keys if a[k]['pick'] != b[k]['pick'])
    return 100 * d, 100 * math.sqrt(var), disc, picks

plan4 = load(R + '/docs/data/mmlu-dumps/mmlu-4b-f16.csv')
f16_14 = load(R + '/docs/data/mmlu-dumps/mmlu-14b-f16.csv')
sealed = load(R + '/docs/mesures/sealed-8b-14b-2026-09-23-brut/mmlu-14b/mmlu-14b-sealed-FULL.csv')
awq = load(R + '/docs/mesures/census-14b-ref-2026-09-22-brut/mmlu-14b-awq-FULL.csv')
g16 = load(R + '/docs/data/mmlu-dumps/mmlu-4b-gguf-f16.csv')
keys = list(plan4.keys())
assert all(k in sealed and sealed[k]['qhash'] == plan4[k]['qhash'] for k in keys)
assert all(k in awq and awq[k]['qhash'] == plan4[k]['qhash'] for k in keys)

print('plan', len(keys), 'questions;', len({k[0] for k in keys}), 'subjects')
for fpc in (False, True):
    tag = 'FPC' if fpc else 'no FPC'
    print(f'--- {tag}')
    se = {}
    for nm, d in [('f16-14B', f16_14), ('awq-14B', awq), ('ours-14B', sealed)]:
        m, s = micro(d, keys, fpc); se[nm] = s
        print(f'  {nm:9s} micro {m:6.2f}  SE {s:4.2f}')
    d1 = paired(sealed, awq, keys, fpc)
    d2 = paired(f16_14, sealed, keys, fpc)
    d3 = paired(g16, plan4, keys, fpc)
    print(f'  ours - awq      {d1[0]:+.2f} (SE {d1[1]:.2f}) disc {d1[2]} pick-diff {d1[3]}')
    print(f'  f16 - ours      {d2[0]:+.2f} (SE {d2[1]:.2f}) disc {d2[2]} pick-diff {d2[3]}')
    print(f'  4B gguf-f16 - ours-f16  {d3[0]:+.2f} (SE {d3[1]:.3f}) disc {d3[2]} pick-diff {d3[3]}')
    # K32 planning: P32 - (AWQ14 + 1) = (R32 - AWQ14) + D_ours - 1
    # METHODE s3: across sizes, SEs in quadrature; within a model, paired.
    se_r32 = se['f16-14B']          # proxy: a model near 78-79 % on this plan
    for k_d, lab in [(1.0, 'D SE = sealed-AWQ SE'), (1.25, 'D SE x1.25')]:
        sd = d1[1] * k_d
        s_tot = math.sqrt(se_r32 ** 2 + se['awq-14B'] ** 2 + sd ** 2)
        s_fix = math.sqrt(se_r32 ** 2 + sd ** 2)
        print(f'  K32 [{lab}]: SE(P32-AWQ14) quadrature {s_tot:.2f} -> 95% half-width {1.96*s_tot:.2f};'
              f' threshold-fixed {s_fix:.2f} -> {1.96*s_fix:.2f}')
        print(f'      A needs P32 > {79.21 + 1.96*s_tot:.2f}, B needs P32 < {79.21 - 1.96*s_tot:.2f}')
    # D_ours alone: half-width
    print(f'  D_ours 95% half-width with SE {d1[1]:.2f}: {1.96*d1[1]:.2f};  x1.25: {1.96*1.25*d1[1]:.2f}')
