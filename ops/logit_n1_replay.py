# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Anomaly N1 replayed on the full split, with a paired interval on the gap. Reads dumps, spends nothing.

N1 (docs/CERVEAU.md, Q7): an LLVQ arm scores fewer MMLU questions than iid Gaussian noise of its own residual
amplitude on the four f16 option logits predicts. `ops/logit_snr.py` section 5 printed it on 2,280 questions, with a
Monte Carlo sd and no interval on the gap. This script forms that interval, per question and paired.

For an arm A against its f16 reference F on the key set K (questions present in both):

- beta and sd are fitted by `logit_snr.fidelity` on K: centred logits, pooled, y = beta x + e.
- Three references give each question k a predicted probability p_k of being answered right:
  gaussian iid: argmax of beta * x_k + N(0, (sd sqrt(4/3))^2) on each option, x_k the four raw f16 logits, over D
  draws (as `logit_snr.simulate`); free reshuffle: the arm's own centred residual vectors permuted across all
  questions, D permutations (as `logit_snr.replay`, bands = 1); banded reshuffle: the same inside 20 bands of f16
  top-two margin (bands = 20).
- c_k is the arm's own 0/1 correct. The gap to reference R is the stratified, population-weighted mean
      G_R = sum_s w_s * mean_{k in s} (c_k - p_k).
  Its normal interval uses SE = sqrt(sum_s w_s^2 var_s(c - p) / n_s), sample variance, no finite-population
  correction, 95 % = +-1.96 SE. A stratified bootstrap of the same statistic (questions resampled within subject, p_k
  held fixed) is printed beside it as a cross-check. Negative G: the arm does worse than the reference predicts.

Limit, stated once: beta and sd are fitted on the same questions the gap is scored on, and p_k is held fixed, so the
interval is conditional on that fit. It carries the sampling of questions and the Monte Carlo noise of p_k (both
inside var_s), not the uncertainty of beta and sd.

Determinism: one seed constant. Every (arm, statistic) pair draws from its own `random.Random`, seeded from SHA-256 of
the seed and the names, so adding or removing an arm moves no other arm's numbers. The three references of one arm
share one bootstrap stream, so their bootstrap intervals are paired across references.

Modes, from the repository root, single thread:
    uv run ops/logit_n1_replay.py selftest [--quick] [--derive-only]   synthetic arms only, no real quantized dump
    uv run ops/logit_n1_replay.py controls [--out f.tsv]               C1, C2, C3 and C4 on the real dumps
    uv run ops/logit_n1_replay.py run [--arms a,b] [--out f.tsv]       every arm of ARMS
Prereg: proofs/preregistration-recherche-qualite-n1-2026-11-13.md. Every number this prints is *computed*.
"""

import hashlib
import math
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logit_snr as ls  # noqa: E402

ROOT = ls.ROOT
SEED = 20261113
DRAWS = 200
BANDS = 20
BOOT = 10_000
Z95 = 1.96
REFS = ("gauss", "free", "banded")
REF_LABEL = {"gauss": "gaussian iid", "free": "free reshuffle", "banded": "banded reshuffle"}

D = "docs/data/mmlu-dumps/"
F16_4B = D + "mmlu-4b-f16-FULL.csv"
F16_8B = D + "mmlu-8b-f16-FULL.csv"
F16_14B = D + "mmlu-14b-f16-FULL.csv"

# The arms. Each is scored against its own f16 reference on the full split (14,042 questions, fingerprint
# a74a6d6213602979). The primary arm is the only one a decision rule reads.
ARMS = [
    ("sealed-4b", D + "mmlu-4b-abl-control-FULL.csv", F16_4B, "primary"),
    ("tetra-4b", D + "mmlu-4b-tetra-FULL.csv", F16_4B, "secondary"),
    ("awq-4b", "docs/mesures/paper-table-2026-09-25-brut/paper-awq4b-census-2026-09-25/mmlu-4b-awq-FULL.csv",
     F16_4B, "secondary"),
    ("abl-qkv", D + "mmlu-4b-abl-qkv-FULL.csv", F16_4B, "secondary"),
    ("abl-o", D + "mmlu-4b-abl-o-FULL.csv", F16_4B, "secondary"),
    ("abl-gateup", D + "mmlu-4b-abl-gateup-FULL.csv", F16_4B, "secondary"),
    ("abl-down", D + "mmlu-4b-abl-down-FULL.csv", F16_4B, "secondary"),
    ("abl-attn", D + "mmlu-4b-abl-attn-FULL.csv", F16_4B, "secondary"),
    ("abl-mlp", D + "mmlu-4b-abl-mlp-FULL.csv", F16_4B, "secondary"),
    ("abl-all", D + "mmlu-4b-abl-all-FULL.csv", F16_4B, "secondary"),
    ("sealed-8b", "docs/mesures/sealed-8b-27-2026-09-24-brut/mmlu-B/mmlu-8b-B-sealed-FULL.csv", F16_8B,
     "secondary"),
    ("sealed-14b", "docs/mesures/sealed-8b-14b-2026-09-23-brut/mmlu-14b/mmlu-14b-sealed-FULL.csv", F16_14B,
     "secondary"),
]
PRIMARY = "sealed-4b"
# Arms allowed to run on the intersection of their keys with the reference's. Empty: every arm must carry exactly
# the reference's questions, or the script refuses it.
INTERSECT = set()

# C2: the stratified micro each full-split dump must reproduce, to 0.01 pp, with the journal line that printed it.
EXPECTED = {
    D + "mmlu-4b-abl-control-FULL.csv": (63.3742, "ablation-qualite-4b-2026-11-12.txt line 84"),
    F16_4B: (70.14, "f16-full-2026-09-18.txt line 14"),
    D + "mmlu-4b-tetra-FULL.csv": (54.64, "tetranu-rowscales journal line 20"),
    "docs/mesures/paper-table-2026-09-25-brut/paper-awq4b-census-2026-09-25/mmlu-4b-awq-FULL.csv":
        (68.14, "paper-table-2026-09-25.txt line 48"),
    D + "mmlu-4b-abl-qkv-FULL.csv": (63.9225, "ablation-qualite-4b-2026-11-12.txt line 85"),
    D + "mmlu-4b-abl-o-FULL.csv": (63.2460, "ablation-qualite-4b-2026-11-12.txt line 86"),
    D + "mmlu-4b-abl-gateup-FULL.csv": (66.7569, "ablation-qualite-4b-2026-11-12.txt line 87"),
    D + "mmlu-4b-abl-down-FULL.csv": (65.3824, "ablation-qualite-4b-2026-11-12.txt line 88"),
    D + "mmlu-4b-abl-attn-FULL.csv": (64.1006, "ablation-qualite-4b-2026-11-12.txt line 89"),
    D + "mmlu-4b-abl-mlp-FULL.csv": (68.2310, "ablation-qualite-4b-2026-11-12.txt line 90"),
    D + "mmlu-4b-abl-all-FULL.csv": (70.0969, "ablation-qualite-4b-2026-11-12.txt line 91"),
    "docs/mesures/sealed-8b-27-2026-09-24-brut/mmlu-B/mmlu-8b-B-sealed-FULL.csv":
        (69.58, "sealed-8b-27-2026-09-24.txt line 25"),
    F16_8B: (75.05, "census-8b-2026-09-21.txt line 24"),
    "docs/mesures/sealed-8b-14b-2026-09-23-brut/mmlu-14b/mmlu-14b-sealed-FULL.csv":
        (75.66, "sealed-8b-14b-2026-09-23.txt line 24"),
    F16_14B: (78.88, "census-14b-ref-2026-09-22.txt line 1"),
}

# C3: section 5 of docs/mesures/logit-snr-4b-2026-09-12.txt (lines 97-103), 2,280 questions against mmlu-4b-f16.csv.
# (gaussian mean, sd, own residuals mean, sd, banded mean, sd)
F16_2280 = D + "mmlu-4b-f16.csv"
SECTION5 = {
    "awq4": (D + "mmlu-4b-awq.csv", (69.21, 0.83, 69.07, 0.78, 69.15, 0.83)),
    "planes14": (D + "mmlu-4b-llvq.csv", (59.58, 1.24, 60.04, 1.25, 59.00, 1.15)),
    "tetra": (D + "mmlu-4b-tetra.csv", (58.53, 1.05, 58.76, 1.06, 57.53, 1.03)),
    "tetra+q5": (D + "mmlu-4b-tetra-q5.csv", (60.03, 0.90, 60.47, 1.23, 60.06, 1.02)),
    "seed1": ("docs/data/bruit-mmlu-graines/mmlu-s1.csv", (60.66, 1.27, 60.81, 1.22, 60.49, 1.15)),
    "seed2": ("docs/data/bruit-mmlu-graines/mmlu-s2.csv", (58.74, 1.00, 58.76, 0.97, 57.92, 1.05)),
    "seed3": ("docs/data/bruit-mmlu-graines/mmlu-s3.csv", (60.34, 1.26, 60.61, 0.98, 60.07, 1.19)),
}
JOURNAL_SNR = "docs/mesures/logit-snr-4b-2026-09-12.txt"


def rng_for(arm, stat):
    """An independent stream per (arm, statistic), fixed by the seed and the two names."""
    h = hashlib.sha256(f"{SEED}|{arm}|{stat}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def gauss_sd(sd_fit):
    """Centring within a question removes a quarter of an iid variance; the simulation undoes it."""
    return sd_fit * math.sqrt(4 / 3)


def check_pair(name, ref, arm):
    """K, after refusing an arm whose questions, answers or populations differ from its reference's."""
    kr, ka = set(ref), set(arm)
    if kr != ka and name not in INTERSECT:
        raise SystemExit(f"{name}: {len(ka - kr)} keys only in the arm, {len(kr - ka)} only in the reference; "
                         "refused (add the arm to INTERSECT to score the intersection)")
    keys = sorted(kr & ka)
    for k in keys:
        if ref[k]["answer"] != arm[k]["answer"] or ref[k]["population"] != arm[k]["population"]:
            raise SystemExit(f"{name}: answer or population differs at {k}; refused")
    return keys


def _wins(v, a):
    """argmax(v) == a under max()'s rule, the first index wins a tie."""
    va = v[a]
    for i in range(4):
        if i != a and (v[i] > va or (v[i] == va and i < a)):
            return False
    return True


def p_gauss(ref, keys, beta, sdi, draws, rng):
    """Per question, the share of draws in which beta * f16 + iid N(0, sdi^2) picks the answer."""
    g = rng.gauss
    out = []
    for k in keys:
        L = ref[k]["logits"]
        a = ref[k]["answer"]
        m0, m1, m2, m3 = (beta * x for x in L)
        hits = 0
        for _ in range(draws):
            v = (m0 + g(0, sdi), m1 + g(0, sdi), m2 + g(0, sdi), m3 + g(0, sdi))
            if _wins(v, a):
                hits += 1
        out.append(hits / draws)
    return out


def residuals(ref, arm, keys, b):
    """The centred residual vector of each question, as logit_snr.replay forms it."""
    X, Y = [], []
    for k in keys:
        X += ls.centred(ref, k)
        Y += ls.centred(arm, k)
    mx, my = sum(X) / len(X), sum(Y) / len(Y)
    err = []
    for k in keys:
        x, y = ls.centred(ref, k), ls.centred(arm, k)
        err.append([y[i] - my - b * (x[i] - mx) for i in range(4)])
    return err


def p_reshuffle(ref, keys, err, beta, bands, draws, rng):
    """Per question, the share of permutations in which beta * centred f16 + a donor's residual picks the answer.
    Donors are drawn inside the question's band of f16 top-two margin; bands = 1 is a free permutation."""
    n = len(keys)
    base = [[beta * x for x in ls.centred(ref, k)] for k in keys]
    ans = [ref[k]["answer"] for k in keys]
    margin = []
    for k in keys:
        L = sorted(ref[k]["logits"], reverse=True)
        margin.append(L[0] - L[1])
    order = sorted(range(n), key=lambda i: margin[i])
    pools = {}
    for r, i in enumerate(order):
        pools.setdefault(r * bands // n, []).append(i)
    hits = [0] * n
    for _ in range(draws):
        for ks in pools.values():
            src = ks[:]
            rng.shuffle(src)
            for j, i in enumerate(ks):
                e = err[src[j]]
                bx = base[i]
                if _wins((bx[0] + e[0], bx[1] + e[1], bx[2] + e[2], bx[3] + e[3]), ans[i]):
                    hits[i] += 1
    return [h / draws for h in hits]


def gap(arm, keys, c, p):
    """G = sum_s w_s mean_s(c - p), SE stratified, sample variance, no finite-population correction."""
    w, g = ls.strata(arm, keys)
    pos = {k: i for i, k in enumerate(keys)}
    G, V = 0.0, 0.0
    for s, ks in g.items():
        d = [c[pos[k]] - p[pos[k]] for k in ks]
        n = len(d)
        m = sum(d) / n
        G += w[s] * m
        V += w[s] ** 2 * (sum((x - m) ** 2 for x in d) / (n - 1)) / n
    return 100 * G, 100 * math.sqrt(V)


def predicted(arm, keys, p, draws):
    """Predicted accuracy and its Monte Carlo SE, from the binomial error of each p_k over D draws."""
    w, g = ls.strata(arm, keys)
    pos = {k: i for i, k in enumerate(keys)}
    acc, v = 0.0, 0.0
    for s, ks in g.items():
        ps = [p[pos[k]] for k in ks]
        n = len(ps)
        acc += w[s] * sum(ps) / n
        v += w[s] ** 2 / n ** 2 * sum(q * (1 - q) / draws for q in ps)
    return 100 * acc, 100 * math.sqrt(v)


def bootstrap(arm, keys, c, ps, boot, rng):
    """Stratified bootstrap of G for each p in ps on one shared stream of resamples. Percentile 2.5 / 97.5."""
    w, g = ls.strata(arm, keys)
    pos = {k: i for i, k in enumerate(keys)}
    strata = []
    for s, ks in g.items():
        n = len(ks)
        ds = [[c[pos[k]] - p[pos[k]] for k in ks] for p in ps]
        strata.append((w[s] / n, list(range(n)), n, ds))
    out = [[] for _ in ps]
    choices = rng.choices
    for _ in range(boot):
        tot = [0.0] * len(ps)
        for wn, idx, n, ds in strata:
            pick = choices(idx, k=n)
            for r, d in enumerate(ds):
                tot[r] += wn * sum(map(d.__getitem__, pick))
        for r in range(len(ps)):
            out[r].append(100 * tot[r])
    res = []
    for v in out:
        v.sort()
        res.append((v[int(0.025 * len(v))], v[int(0.975 * len(v))]))
    return res


def score(name, ref, arm, keys, refs=REFS, draws=DRAWS, boot=BOOT):
    """Every quantity of one arm. Returns a dict; prints nothing."""
    b, sd, sx, snr = ls.fidelity(ref, arm, keys)
    c = [arm[k]["correct"] for k in keys]
    out = dict(name=name, n=len(keys), beta=b, sd=sd, snr=snr,
               measured=ls.accuracy(arm, keys), measured_se=ls.accuracy_se(arm, keys), rows={})
    ps = {}
    if "gauss" in refs:
        ps["gauss"] = p_gauss(ref, keys, b, gauss_sd(sd), draws, rng_for(name, "gauss"))
    if "free" in refs or "banded" in refs:
        err = residuals(ref, arm, keys, b)
        if "free" in refs:
            ps["free"] = p_reshuffle(ref, keys, err, b, 1, draws, rng_for(name, "free"))
        if "banded" in refs:
            ps["banded"] = p_reshuffle(ref, keys, err, b, BANDS, draws, rng_for(name, "banded"))
    names = [r for r in REFS if r in ps]
    bounds = (bootstrap(arm, keys, c, [ps[r] for r in names], boot, rng_for(name, "bootstrap"))
              if boot else [(float("nan"), float("nan"))] * len(names))
    for r, bb in zip(names, bounds):
        pr, mc = predicted(arm, keys, ps[r], draws)
        G, se = gap(arm, keys, c, ps[r])
        out["rows"][r] = dict(predicted=pr, mc_se=mc, G=G, se=se, lo=G - Z95 * se, hi=G + Z95 * se,
                              boot_lo=bb[0], boot_hi=bb[1])
    return out


def show(res, ref_name=""):
    print(f"arm {res['name']}{'  against ' + ref_name if ref_name else ''}   |K| = {res['n']}")
    print(f"  beta {res['beta']:.4f}   sd_res {res['sd']:.4f}   SNR {res['snr']:.4f}   "
          f"measured {res['measured']:.2f} +- {res['measured_se']:.2f} (SE)")
    print(f"  {'reference':<17} {'predicted':>9} {'MC SE':>6} {'G':>7} {'SE':>5} "
          f"{'95 % normal':>17} {'95 % bootstrap':>17}")
    for r, row in res["rows"].items():
        print(f"  {REF_LABEL[r]:<17} {row['predicted']:9.2f} {row['mc_se']:6.3f} {row['G']:+7.2f} "
              f"{row['se']:5.2f}   [{row['lo']:+6.2f}, {row['hi']:+6.2f}]   "
              f"[{row['boot_lo']:+6.2f}, {row['boot_hi']:+6.2f}]")


TSV_HEAD = ("arm\treference\tn\tbeta\tsd\tsnr\tmeasured\tmeasured_se\tpredicted\tmc_se\tG\tSE\tlo\thi"
            "\tboot_lo\tboot_hi")


def tsv_rows(res):
    for r, row in res["rows"].items():
        yield "\t".join([res["name"], r, str(res["n"])] + [
            f"{x:.4f}" for x in (res["beta"], res["sd"], res["snr"], res["measured"], res["measured_se"],
                                 row["predicted"], row["mc_se"], row["G"], row["se"], row["lo"], row["hi"],
                                 row["boot_lo"], row["boot_hi"])])


def write_tsv(path, results):
    if not path:
        return
    with open(path, "w") as fh:
        fh.write(TSV_HEAD + "\n")
        for res in results:
            for line in tsv_rows(res):
                fh.write(line + "\n")
    print(f"\nwrote {path}")


# ---------------------------------------------------------------------------
# Synthetic arms. The self-test reads one real file, the 2,280-question f16 dump, and nothing else.
# ---------------------------------------------------------------------------

def full_sized(f16):
    """A reference with the census shape: subject s gets its full-split population of questions, question j a copy
    of the (j mod n_s)-th of the 2,280-q sample. Weights and n_s equal the census's; no randomness."""
    by = {}
    for k in sorted(f16):
        by.setdefault(k[0], []).append(k)
    out = {}
    for s, ks in by.items():
        P = f16[ks[0]]["population"]
        for j in range(P):
            out[(s, j)] = dict(f16[ks[j % len(ks)]])
    return out


def planted_sds(f16, k, sigma, where, scale):
    """Per-option noise sd of a planted arm, same total variance 4 sigma^2 on every question except `margin`."""
    L = f16[k]["logits"]
    order = sorted(range(4), key=lambda i: -L[i])
    if where in ("all", "letter"):
        return [sigma] * 4
    if where in ("tail", "head"):
        w = [0.0] * 4
        for i in (order[2:] if where == "tail" else order[:2]):
            w[i] = sigma * math.sqrt(2.0)
        return w
    if where == "margin":
        top = sorted(L, reverse=True)
        return [sigma * (top[0] - top[1] + 1.0) / scale] * 4
    raise ValueError(where)


def margin_scale(f16):
    """rms of (f16 margin + 1), so the margin-scaled arm keeps the mean noise variance of sigma^2."""
    v = []
    for k in f16:
        top = sorted(f16[k]["logits"], reverse=True)
        v.append((top[0] - top[1] + 1.0) ** 2)
    return math.sqrt(sum(v) / len(v))


LETTER, OFFSET = 1, 1.0  # planted arm (c): +1.0 on option B, in arm units


def synth(f16, keys, beta, sigma, rng, where="all", scale=1.0):
    out = {}
    for k in keys:
        sds = planted_sds(f16, k, sigma, where, scale)
        v = [beta * f16[k]["logits"][i] + sds[i] * rng.gauss(0, 1) for i in range(4)]
        if where == "letter":
            v[LETTER] += OFFSET
        a = f16[k]["answer"]
        pick = max(range(4), key=lambda i: v[i])
        out[k] = dict(population=f16[k]["population"], answer=a, pick=pick, correct=int(pick == a), logits=v)
    return out


def _Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def exact_p(mu, sig, a):
    """P(argmax = a) for independent N(mu_i, sig_i^2); sig_i may be 0. Simpson on 201 nodes over +-8 sd."""
    if sig[a] == 0.0:
        p = 1.0
        for i in range(4):
            if i != a:
                p *= _Phi((mu[a] - mu[i]) / sig[i]) if sig[i] > 0 else (1.0 if mu[i] < mu[a] else 0.0)
        return p
    n, lo, hi = 200, -8.0, 8.0
    h = (hi - lo) / n
    tot = 0.0
    for j in range(n + 1):
        z = lo + j * h
        t = mu[a] + sig[a] * z
        f = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        for i in range(4):
            if i != a:
                f *= _Phi((t - mu[i]) / sig[i]) if sig[i] > 0 else (1.0 if mu[i] < t else 0.0)
        tot += f * (1 if j in (0, n) else (4 if j % 2 else 2))
    return tot * h / 3


def derive(f16, beta, sigma, scale):
    """The expected gap of each planted arm against the gaussian reference on the census-shaped reference, from the
    exact integral, before the instrument runs. The census-shaped set holds each 2,280-q question a known number of
    times, so the expectation is a weighted sum over the 2,280. The prediction is beta * f16 + iid noise at the sd the
    fit should read back: sigma for (a), (b) and (d), whose total variance is 4 sigma^2 a question on average;
    sqrt(sigma^2 + OFFSET^2 / 4) for (c), whose centred offset adds 3 OFFSET^2 / 16 to the per-logit residual
    variance. It neglects the error of the fitted beta and sd."""
    keys = sorted(f16)
    w, g = ls.strata(f16, keys)
    out = {}
    for where in ("tail", "margin", "letter", "head"):
        sd_pred = math.sqrt(sigma ** 2 + OFFSET ** 2 / 4) if where == "letter" else sigma
        em, ep = 0.0, 0.0
        for s, ks in g.items():
            P, n = f16[ks[0]]["population"], len(ks)
            m1, m2 = 0.0, 0.0
            for j, k in enumerate(ks):
                mult = P // n + (1 if j < P % n else 0)
                mu = [beta * x for x in f16[k]["logits"]]
                a = f16[k]["answer"]
                mp = mu[:]
                if where == "letter":
                    mp[LETTER] += OFFSET
                m1 += mult * exact_p(mp, planted_sds(f16, k, sigma, where, scale), a)
                m2 += mult * exact_p(mu, [sd_pred] * 4, a)
            em += w[s] * m1 / P
            ep += w[s] * m2 / P
        out[where] = (100 * em, 100 * ep, 100 * (em - ep))
    return out


def selftest(quick=False, derive_only=False, out=None):
    t0 = time.time()
    results = []
    f16 = ls.load(F16_2280)
    keys = sorted(f16)
    big = full_sized(f16)
    bkeys = sorted(big)
    ok = True
    beta, sigma = 0.6, 1.3
    print(f"self-test, synthetic arms built from {F16_2280} only ({len(keys)} q) and its census-shaped copy "
          f"({len(bkeys)} q). beta {beta}, noise sd {sigma} a logit before centring.")

    if not derive_only:
        print("\nU1. The fit reads the planted noise back, and the simulation sd undoes the centring.")
        a = synth(f16, keys, beta, sigma, rng_for("U1", "noise"))
        b, sd, _, _ = ls.fidelity(f16, a, keys)
        good = abs(b - beta) < 0.03 * beta and abs(gauss_sd(sd) - sigma) < 0.03 * sigma
        ok &= good
        print(f"   beta {b:.4f} (planted {beta}), simulation sd {gauss_sd(sd):.4f} (planted {sigma})   "
              f"{'passes' if good else 'FAILS'}")

        print("\nU2. Monte Carlo p_k against the exact integral, 300 questions, D = 2,000.")
        sub = keys[::7][:300]
        pm = p_gauss(f16, sub, beta, sigma, 2000, rng_for("U2", "gauss"))
        pe = [exact_p([beta * x for x in f16[k]["logits"]], [sigma] * 4, f16[k]["answer"]) for k in sub]
        z = [(pm[i] - pe[i]) / math.sqrt(max(pe[i] * (1 - pe[i]), 1e-4) / 2000) for i in range(len(sub))]
        rms = math.sqrt(sum(x * x for x in z) / len(z))
        mean_d = 100 * (sum(pm) - sum(pe)) / len(sub)
        good = 0.8 < rms < 1.2 and abs(mean_d) < 0.3
        ok &= good
        print(f"   rms z {rms:.3f} (expected 1), mean difference {mean_d:+.3f} pp   {'passes' if good else 'FAILS'}")

        print("\nU3. The normal SE against the bootstrap SE, and the key check.")
        res = score("U3", f16, a, keys, refs=("gauss",), boot=2000)
        row = res["rows"]["gauss"]
        bse = (row["boot_hi"] - row["boot_lo"]) / (2 * Z95)
        good = abs(bse - row["se"]) < 0.15 * row["se"]
        ok &= good
        print(f"   normal SE {row['se']:.3f}, bootstrap half-width / 1.96 {bse:.3f}   {'passes' if good else 'FAILS'}")
        short = {k: v for k, v in a.items() if k != keys[0]}
        try:
            check_pair("U3-short", f16, short)
            good = False
        except SystemExit as e:
            good = "refused" in str(e)
        ok &= good
        print(f"   an arm missing one key is refused   {'passes' if good else 'FAILS'}")
        r1 = rng_for("x", "gauss").random()
        good = r1 == rng_for("x", "gauss").random() and r1 != rng_for("y", "gauss").random()
        ok &= good
        print(f"   streams are fixed by (arm, statistic) and differ between arms   {'passes' if good else 'FAILS'}")

        print(f"\nC4. Null arms: beta * f16 + iid gaussian. 20 independent arms; G_gauss's 95 % normal interval")
        print("    must contain 0 in at least 17.")
        sizes = [("2,280 q", f16, keys, ("gauss", "free", "banded"))]
        if not quick:
            sizes.append(("census-shaped", big, bkeys, ("gauss",)))
        for label, ref, ks, refs in sizes:
            cover = {r: 0 for r in refs}
            Gs, SEs = [], []
            for i in range(20):
                arm = synth(ref, ks, beta, sigma, rng_for(f"C4:{label}:{i}", "noise"))
                res = score(f"C4:{label}:{i}", ref, arm, ks, refs=refs, boot=0)
                for r in refs:
                    row = res["rows"][r]
                    cover[r] += row["lo"] <= 0 <= row["hi"]
                Gs.append(res["rows"]["gauss"]["G"])
                SEs.append(res["rows"]["gauss"]["se"])
            mG = sum(Gs) / len(Gs)
            sG = math.sqrt(sum((x - mG) ** 2 for x in Gs) / (len(Gs) - 1))
            good = cover["gauss"] >= 17
            ok &= good
            extra = "".join(f", {REF_LABEL[r]} {cover[r]}/20" for r in refs if r != "gauss")
            print(f"   {label:<14} gaussian covers 0 in {cover['gauss']}/20{extra}; mean G {mG:+.3f}, sd of G "
                  f"{sG:.3f}, mean SE {sum(SEs) / len(SEs):.3f}   {'passes' if good else 'FAILS'}")

    if quick:
        print(f"\n{'self-test passes' if ok else 'SELF-TEST FAILS'} (quick: U1-U3 and C4 on 2,280 q), "
              f"{time.time() - t0:.0f} s")
        return 0 if ok else 1

    print("\nC5 derivation. Expected gap against the gaussian reference, exact integral, census-shaped reference.")
    print("   (a) tail: the same total variance on the options f16 ranks 3rd and 4th only")
    print("   (b) margin: sd of every option proportional to (f16 top-two margin + 1), mean variance sigma^2")
    print(f"   (c) letter: iid noise plus a constant +{OFFSET} on option {'ABCD'[LETTER]}")
    print("   (d) head: the same total variance on the options f16 ranks 1st and 2nd only")
    scale = margin_scale(big)
    der = derive(f16, beta, sigma, scale)
    for where, (em, ep, d) in der.items():
        print(f"   {where:<7} expected accuracy {em:6.2f}, expected gaussian prediction {ep:6.2f}, "
              f"expected G_gauss {d:+6.2f}")
    if derive_only:
        print(f"\nderivation only, {time.time() - t0:.0f} s")
        return 0

    print("\nC5. Planted couplings, census-shaped reference, every reference and the bootstrap.")
    for where in ("tail", "margin", "letter", "head"):
        arm = synth(big, bkeys, beta, sigma, rng_for(f"C5:{where}", "noise"), where, scale)
        res = score(f"C5:{where}", big, arm, bkeys)
        results.append(res)
        show(res)
        row = res["rows"]["gauss"]
        want = der[where][2]
        if where in ("tail", "margin", "head"):
            good = (row["lo"] > 0) if want > 0 else (row["hi"] < 0)
            ok &= good
            print(f"   resolved away from 0 with the derived sign ({want:+.2f})   {'passes' if good else 'FAILS'}")
        else:
            print(f"   reported, no pass rule (derived {want:+.2f})")
    write_tsv(out, results)
    print(f"\n{'self-test passes' if ok else 'SELF-TEST FAILS'}, {time.time() - t0:.0f} s")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Controls on the real dumps (C1-C4) and the run. Both run only after the prereg is stamped.
# ---------------------------------------------------------------------------

def controls(out=None):
    t0 = time.time()
    ok = True
    results = []
    print("C1. ops/logit_snr.py, unchanged, reproduces its journal.")
    got = subprocess.run([sys.executable, os.path.join(ROOT, "ops", "logit_snr.py")], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    with open(os.path.join(ROOT, JOURNAL_SNR)) as fh:
        want = fh.read()

    def sec5(t):
        return t[t.index("5. What the amplitude"):t.index("6. The loss")]

    s5 = sec5(got) == sec5(want)
    whole = got == want
    ok &= s5
    print(f"   section 5 character for character: {'passes' if s5 else 'FAILS'}; whole output: "
          f"{'identical' if whole else 'differs'}")

    print("\nC2. Measured accuracy of every full-split dump against its journal, to 0.01 pp.")
    paths = []
    for _, a, r, _ in ARMS:
        for p in (a, r):
            if p not in paths:
                paths.append(p)
    for p in paths:
        rows = ls.load(p)
        acc = ls.accuracy(rows, sorted(rows))
        fig, src = EXPECTED[p]
        good = abs(acc - fig) < 0.01
        ok &= good
        print(f"   {os.path.basename(p):<34} {acc:8.4f} against {fig:8.4f} ({src})   {'passes' if good else 'FAILS'}")

    print("\nC3. The seven section-5 arms, 2,280 q: each prediction within 3 journal sd of the journal's value.")
    f16 = ls.load(F16_2280)
    for nm, (p, j) in SECTION5.items():
        arm = ls.load(p)
        keys = check_pair(nm, f16, arm)
        res = score("C3:" + nm, f16, arm, keys, boot=0)
        results.append(res)
        line = f"   {nm:<9}"
        for i, r in enumerate(REFS):
            pr = res["rows"][r]["predicted"]
            z = (pr - j[2 * i]) / j[2 * i + 1]
            good = abs(z) <= 3
            ok &= good
            line += f"  {r} {pr:6.2f} vs {j[2 * i]:6.2f} ({z:+.2f} sd) {'passes' if good else 'FAILS'}"
        print(line)

    print("\nC4. Null arms on the real 4B f16 census dump, at the sealed arm's fitted beta and sd.")
    ref = ls.load(F16_4B)
    sealed = ls.load(next(a for n, a, _, _ in ARMS if n == PRIMARY))
    keys = check_pair(PRIMARY, ref, sealed)
    b, sd, _, _ = ls.fidelity(ref, sealed, keys)
    cover = 0
    for i in range(20):
        arm = synth(ref, keys, b, gauss_sd(sd), rng_for(f"C4post:{i}", "noise"))
        res = score(f"C4post:{i}", ref, arm, keys, refs=("gauss",), boot=0)
        cover += res["rows"]["gauss"]["lo"] <= 0 <= res["rows"]["gauss"]["hi"]
    good = cover >= 17
    ok &= good
    print(f"   beta {b:.4f}, simulation sd {gauss_sd(sd):.4f}: gaussian covers 0 in {cover}/20   "
          f"{'passes' if good else 'FAILS'}")
    write_tsv(out, results)
    print(f"\n{'controls pass' if ok else 'CONTROLS FAIL: rule D, the run describes and decides nothing'}, "
          f"{time.time() - t0:.0f} s")
    return 0 if ok else 1


def run(only=None, out=None):
    t0 = time.time()
    results = []
    cache = {}
    for name, a, r, role in ARMS:
        if only and name not in only:
            continue
        if r not in cache:
            cache[r] = ls.load(r)
        ref = cache[r]
        arm = ls.load(a)
        keys = check_pair(name, ref, arm)
        t1 = time.time()
        res = score(name, ref, arm, keys)
        results.append(res)
        print(f"\n[{role}] {a}")
        show(res, os.path.basename(r))
        print(f"  ({time.time() - t1:.0f} s)")
        if name == PRIMARY:
            row = res["rows"]["gauss"]
            verdict = "HOLDS" if row["hi"] < 0 else "FALLS"
            print(f"  P1: G_gauss 95 % normal interval [{row['lo']:+.2f}, {row['hi']:+.2f}] -> N1 {verdict} on the "
                  f"sealed 4B (valid only if `controls` passed, rule D)")
    write_tsv(out, results)
    print(f"\n{len(results)} arms, {time.time() - t0:.0f} s")
    return 0


def main(argv):
    if not argv or argv[0] not in ("selftest", "controls", "run"):
        print(__doc__)
        return 2
    out = None
    if "--out" in argv:
        out = argv[argv.index("--out") + 1]
    if argv[0] == "selftest":
        return selftest(quick="--quick" in argv, derive_only="--derive-only" in argv, out=out)
    if argv[0] == "controls":
        return controls(out)
    only = None
    if "--arms" in argv:
        only = set(argv[argv.index("--arms") + 1].split(","))
    return run(only, out)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
