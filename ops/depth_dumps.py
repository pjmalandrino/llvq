# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Dump readings of quality phase 3 (CERVEAU O15), at $0. Reads committed files, loads no model.

Implements, to the letter, proofs/preregistration-recherche-qualite3-dumps-2026-11-15.md:
G1 (flip hazard), M1 (slope and noise), M2 (flips of the late arm), G2 (chain length),
G3 (false arithmetic), G4 (shared repairs), the gates of its section 4, and its self-check.

Usage, from the repository root:
    uv run ops/depth_dumps.py selftest
    uv run ops/depth_dumps.py run <out_dir>

The GSM8K grader is a port of llvq-llm/src/gsm8k.rs (`regrade`, `extract`, `last_boxed`,
`numbers`, `canonical`, `frac_value`, `is_correct`), the rule `bin/gsm8kpair` applies.
The M1 fit is the one of docs/mesures/recherche-qualite2-r0-contraste-2026-11-14-brut/beta_table.py.
"""

import hashlib
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 20261115
DRAWS = 10_000
WEIGHT_SHARE_LATE = 0.306
TOL_REL = 0.005

MMLU = {
    "f16": "docs/data/mmlu-dumps/mmlu-4b-f16-FULL.csv",
    "control": "docs/data/mmlu-dumps/mmlu-4b-abl-control-FULL.csv",
    "late": "docs/data/mmlu-dumps/mmlu-4b-depth-late-FULL.csv",
    "early": "docs/data/mmlu-dumps/mmlu-4b-depth-early-FULL.csv",
}
SEPT = {
    "q5-shipped": "docs/data/mmlu-dumps/mmlu-q5-shipped-FULL.csv",
    "q5-down-0-11": "docs/data/mmlu-dumps/mmlu-q5-down-0-11.csv",
    "q5-down-12-23": "docs/data/mmlu-dumps/mmlu-q5-down-12-23.csv",
    "q5-down-24-35": "docs/data/mmlu-dumps/mmlu-q5-down-24-35.csv",
}
GSM = {
    "control": "docs/mesures/gsm8k-depth-restore-4b-2026-11-14-brut/gsm8k-4b-depth-control.jsonl",
    "late": "docs/mesures/gsm8k-depth-restore-4b-2026-11-14-brut/gsm8k-4b-depth-late.jsonl",
    "early": "docs/mesures/gsm8k-depth-restore-4b-2026-11-14-brut/gsm8k-4b-depth-early.jsonl",
    "f16": "docs/mesures/gsm8k-wave1-2026-09-26-brut/run5-vllm/gsm8k-4b-f16-vllm.jsonl",
}


# ---------------------------------------------------------------------------
# GSM8K grader, ported from llvq-llm/src/gsm8k.rs. Byte logic on ASCII only:
# every test below is on an ASCII character, so code points and bytes agree.
# ---------------------------------------------------------------------------

def _dig(c):
    return "0" <= c <= "9"


def last_boxed(s):
    key = "\\boxed{"
    p = s.rfind(key)
    if p < 0:
        return None
    start = p + len(key)
    depth = 1
    for i, ch in enumerate(s[start:]):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:start + i]
    return None


def numbers(s):
    out = []
    i, n = 0, len(s)
    while i < n:
        if not _dig(s[i]):
            i += 1
            continue
        start = i - 1 if i > 0 and s[i - 1] == "-" else i
        j = i
        while j < n and (_dig(s[j]) or (s[j] == "," and j + 1 < n and _dig(s[j + 1]))):
            j += 1
        if j + 1 < n and s[j] == "." and _dig(s[j + 1]):
            j += 1
            while j < n and _dig(s[j]):
                j += 1
        out.append(s[start:j])
        i = j
    return out


def canonical(raw):
    t = "".join(c for c in raw.strip() if c != ",")
    neg = t.startswith("-")
    body = t[1:] if neg else t
    if "." in body:
        intp, frac = body.split(".", 1)
    else:
        intp, frac = body, ""
    if intp == "" and frac == "":
        return None
    if not all(_dig(c) for c in intp) or not all(_dig(c) for c in frac):
        return None
    intp = intp.lstrip("0") or "0"
    frac = frac.rstrip("0")
    zero = intp == "0" and frac == ""
    out = "-" if neg and not zero else ""
    out += intp
    if frac:
        out += "." + frac
    return out


def frac_value(inner):
    """None: no fraction. ("frac", None): a fraction that is not an integer. ("frac", q)."""
    key = None
    for k in ("\\dfrac{", "\\tfrac{", "\\frac{"):
        p = inner.find(k)
        if p >= 0:
            key = p + len(k)
            break
    if key is None:
        return None
    rest = inner[key:]
    if "}" not in rest:
        return None
    num, rest = rest.split("}", 1)
    if not rest.startswith("{"):
        return None
    rest = rest[1:]
    if "}" not in rest:
        return None
    den = rest.split("}", 1)[0]

    def parse(s):
        c = canonical(s)
        if c is None or "." in c:
            return None
        return int(c)

    a, b = parse(num), parse(den)
    if a is None or b is None:
        return ("frac", None)
    if b == 0 or a % b != 0:
        return ("frac", None)
    q = abs(a) // abs(b)
    if (a < 0) != (b < 0) and q != 0:
        q = -q
    return ("frac", str(q))


def extract(completion):
    inner = last_boxed(completion)
    if inner is not None:
        v = frac_value(inner)
        if v is not None:
            return "boxed", v[1]
        ns = numbers(inner)
        if ns:
            return "boxed", canonical(ns[-1])
    ns = numbers(completion)
    if ns:
        return "last", canonical(ns[-1])
    return "none", None


def regrade(row):
    gold = canonical(row["gold"])
    if gold is None:
        raise ValueError(f"problem {row['index']}: gold {row['gold']!r} is not a number")
    source, extracted = extract(row["completion"])
    return gold, source, extracted, extracted == gold


def load_gsm(path):
    with open(os.path.join(ROOT, path)) as fh:
        lines = [l for l in fh if l.strip()]
    header = json.loads(lines[0])
    if header.get("llvq_gsm8k_dump") != 1:
        raise ValueError(f"{path} is not a GSM8K dump v1")
    rows, trailer = [], None
    for l in lines[1:]:
        v = json.loads(l)
        if "end" in v:
            trailer = v
            continue
        if trailer is not None:
            raise ValueError(f"{path}: a row after the trailer")
        rows.append(v)
    if trailer is None or trailer.get("questions") != len(rows):
        raise ValueError(f"{path}: no trailer or wrong count")
    for r in rows:
        r["gold_c"], r["source_c"], r["extracted_c"], r["correct_c"] = regrade(r)
    return header, rows, trailer


# ---------------------------------------------------------------------------
# G1. The flip hazard.
# ---------------------------------------------------------------------------

def common_prefix_len(a, b):
    """Leading characters shared by a and b, by bisection on slice equality."""
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def flip_row(ref, arm):
    """(d, censored) for one problem: d leading characters shared with the reference."""
    d = common_prefix_len(ref, arm)
    cens = (ref == arm) or ref.startswith(arm) or arm.startswith(ref)
    return d, cens


def hazard(d, cens):
    """Flips a character: uncensored problems over the sum of d over all problems."""
    d = np.asarray(d, dtype=np.int64)
    unc = ~np.asarray(cens, dtype=bool)
    n_events = int(unc.sum())
    exposure = int(d.sum())
    return n_events / exposure, n_events, exposure


def hazard_boot(d, cens, idx):
    """Hazard on each bootstrap draw of problems (rows of idx), same estimator."""
    d = np.asarray(d, dtype=np.float64)
    unc = (~np.asarray(cens, dtype=bool)).astype(np.float64)
    out = np.empty(idx.shape[0])
    for s in range(0, idx.shape[0], 1000):
        blk = idx[s:s + 1000]
        out[s:s + 1000] = unc[blk].sum(1) / d[blk].sum(1)
    return out


# ---------------------------------------------------------------------------
# M1. Slope and noise, as beta_table.py: centre the four logits, pool, y = beta x + e,
# sd_res = sqrt(sum e^2 / 3N).
# ---------------------------------------------------------------------------

def load_mmlu(path):
    out = {}
    with open(os.path.join(ROOT, path)) as fh:
        hdr = None
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            p = line.rstrip("\n").split(",")
            if hdr is None:
                hdr = p
                continue
            out[(p[0], p[1])] = (np.array([float(x) for x in p[7:11]]), int(p[4]), int(p[5]),
                                 int(p[6]), p[3])
    return out


def fit(x, y):
    """x, y: (N, 4) logits. Returns (beta, sd_res)."""
    x = x - x.mean(1, keepdims=True)
    y = y - y.mean(1, keepdims=True)
    b = (x * y).sum() / (x * x).sum()
    e = y - b * x
    sd = np.sqrt((e * e).sum() / (3 * x.shape[0]))
    return b, sd


def fit_stats(x, y):
    """Per-question sums Sxx, Sxy, Syy of the centred logits, for the bootstrap."""
    x = x - x.mean(1, keepdims=True)
    y = y - y.mean(1, keepdims=True)
    return np.stack([(x * x).sum(1), (x * y).sum(1), (y * y).sum(1)], 1)


def fit_from_sums(sxx, sxy, syy, n):
    b = sxy / sxx
    sd = np.sqrt((syy - 2 * b * sxy + b * b * sxx) / (3 * n))
    return b, sd


# ---------------------------------------------------------------------------
# G3. False arithmetic. A step is "a op b = c", plain unsigned numbers, op in
# {+, -, ×, ÷, *, /}; ASCII '-' and U+2212 are both minus, LaTeX \times and \div are
# spellings of × and ÷. Commas between digits and currency signs are stripped.
# A step inside a chain (an operator right before a, or right after c) is not of the form.
# ---------------------------------------------------------------------------

OPS = "+-\u2212\u00d7\u00f7*/"
STEP_RE = re.compile(r"(\d+(?:\.\d+)?)[ \t]*([+\-\u2212\u00d7\u00f7*/])[ \t]*(\d+(?:\.\d+)?)"
                     r"[ \t]*=[ \t]*(\d+(?:\.\d+)?)")
BEFORE_BAD = set("0123456789." + OPS + "^")
AFTER_BAD = set(OPS + "^")


def prep(s):
    s = s.replace("**", " ")
    s = s.replace("\\times", "\u00d7").replace("\\div", "\u00f7")
    for cur in ("\\$", "$", "\u20ac", "\u00a3"):
        s = s.replace(cur, "")
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    return s


def apply_op(a, op, b):
    if op == "+":
        return a + b
    if op in "-\u2212":
        return a - b
    if op in "\u00d7*":
        return a * b
    if b == 0:
        return float("inf")
    return a / b


def steps(text):
    """Every step of the form in text: list of (a, op, b, c, false?)."""
    s = prep(text)
    out = []
    pos = 0
    while True:
        m = STEP_RE.search(s, pos)
        if m is None:
            break
        pos = m.start() + 1
        j = m.start() - 1
        while j >= 0 and s[j] in " \t":
            j -= 1
        if j >= 0 and s[j] in BEFORE_BAD:
            continue
        k = m.end()
        while k < len(s) and s[k] in " \t":
            k += 1
        if k < len(s) and s[k] in AFTER_BAD:
            continue
        a, op, b, c = float(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4))
        v = apply_op(a, op, b)
        false = abs(v - c) > TOL_REL * max(1.0, abs(c))
        out.append((m.group(1), op, m.group(3), m.group(4), false))
        pos = m.end()
    return out


# ---------------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------------

def ci(v):
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


def sha256(path):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, path), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def boot_idx(n):
    """The bootstrap draws over problems, 10,000 x n, seed of the prereg."""
    return np.random.default_rng(SEED).integers(0, n, size=(DRAWS, n))


# ---------------------------------------------------------------------------
# Self-check, prereg section 4.
# ---------------------------------------------------------------------------

def _rand_text(rng, n):
    return bytes(rng.integers(97, 123, size=n, dtype=np.uint8)).decode()


def _planted_arm(rng, refs, h):
    """An arm leaving each reference with per-character hazard h. Some arms stop early,
    some run on past the reference, so censoring by prefix is planted as well as equality."""
    out = []
    for f in refs:
        L = len(f)
        u = rng.random()
        if u < 0.3:
            T = int(rng.integers(0, L + 1))     # stops early: a prefix if no flip before T
        elif u < 0.4:
            T = L + int(rng.integers(1, 50))    # runs on past the reference
        else:
            T = L                               # same end
        d = int(rng.geometric(h)) - 1           # characters matched before the flip
        if d < min(T, L):
            c = "A"                             # never in the reference alphabet
            out.append(f[:d] + c + _rand_text(rng, int(rng.integers(0, 200))))
        elif T <= L:
            out.append(f[:T])
        else:
            out.append(f + _rand_text(rng, T - L))
    return out


def selftest():
    ok = True
    rng = np.random.default_rng(7)

    print("S1. GSM8K grader, the cases of llvq-llm/src/gsm8k.rs tests.")
    cases = [
        (canonical("18"), "18"), (canonical("18.00"), "18"), (canonical("018"), "18"),
        (canonical("1,000.50"), "1000.5"), (canonical("-0.0"), "0"), (canonical("0.5"), "0.5"),
        (canonical(".5"), "0.5"), (canonical("eighteen"), None), (canonical(""), None),
        (canonical("1.2.3"), None),
        (numbers("a 1,000 b"), ["1,000"]), (numbers("3, 4 and 5."), ["3", "4", "5"]),
        (numbers("10-2=8"), ["10", "-2", "8"]), (numbers("$18.50 each."), ["18.50"]),
        (numbers("no digits"), []),
        (last_boxed("x \\boxed{3} then \\boxed{18}"), "18"),
        (last_boxed("\\boxed{\\frac{1}{2}}"), "\\frac{1}{2}"),
        (last_boxed("\\boxed{12"), None), (last_boxed("no box"), None),
        (extract("So she makes 9 * 2 = 18 dollars.\n\n\\boxed{18}"), ("boxed", "18")),
        (extract("\\boxed{\\$1,080}"), ("boxed", "1080")),
        (extract("\\boxed{18 \\text{ dollars}}"), ("boxed", "18")),
        (extract("\\boxed{2 \\times 9 = 18}"), ("boxed", "18")),
        (extract("The answer is 42."), ("last", "42")),
        (extract("I cannot tell."), ("none", None)),
        (extract("\\boxed{\\frac{36}{2}}")[1], "18"),
        (extract("\\boxed{\\dfrac{3}{4}}"), ("boxed", None)),
        (extract("\\boxed{\\frac{3}{0}}"), ("boxed", None)),
        (extract("9 * 2 = 18, so \\boxed{1"), ("last", "1")),
        (extract("\\boxed{18.0}")[1] == "18", True),
        (extract("\\boxed{18.5}")[1] == "18", False),
    ]
    bad = [i for i, (got, want) in enumerate(cases) if got != want]
    ok &= not bad
    print(f"   {len(cases) - len(bad)} of {len(cases)} cases agree"
          + ("" if not bad else f"; FAILS on cases {bad}"))

    print("\nS2. Planted flip rate. Three arms against one reference set, hazards 0.0030,")
    print("    0.0020, 0.0012 a character, 20,000 problems, 30 % of arms stopped early and 10 %")
    print("    run past the reference. Pass: each hazard within 5 %, each cut within 0.03.")
    refs = [_rand_text(rng, int(rng.integers(200, 1200))) for _ in range(20000)]
    planted = {"control": 0.0030, "late": 0.0020, "early": 0.0012}
    got = {}
    for name, h in planted.items():
        arm = _planted_arm(rng, refs, h)
        rows = [flip_row(f, a) for f, a in zip(refs, arm)]
        dd, cc = [r[0] for r in rows], [r[1] for r in rows]
        hh, ne, ex = hazard(dd, cc)
        hb = hazard_boot(dd, cc, np.arange(len(dd))[None, :])[0]
        got[name] = hh
        good = abs(hh - h) < 0.05 * h and abs(hb - hh) < 1e-12
        ok &= good
        print(f"   {name:<8} planted {h:.4f} read {hh:.5f} ({ne} flips over {ex} characters, "
              f"{sum(r[1] for r in rows)} censored)  {'passes' if good else 'FAILS'}")
    for name, want in (("late", 1 - 0.0020 / 0.0030), ("early", 1 - 0.0012 / 0.0030)):
        r = 1 - got[name] / got["control"]
        good = abs(r - want) < 0.03
        ok &= good
        print(f"   r_{name:<6} planted {want:.4f} read {r:.4f}  {'passes' if good else 'FAILS'}")

    print("\nS3. Planted slope and noise: x four logits a question with a question offset,")
    print("    y = beta x + offset + N(0, sigma). Pass: beta within 4 of its sampling sd,")
    print("    sd_res within 2 %, and the bootstrap sums give the same fit.")
    for beta, sigma in ((0.5817, 1.4205), (0.95, 0.40), (0.30, 2.00)):
        n = 14042
        x = rng.normal(0, 1.6, size=(n, 4)) + rng.normal(22, 2, size=(n, 1))
        y = beta * x + rng.normal(0, 3, size=(n, 1)) + rng.normal(0, sigma, size=(n, 4))
        b, sd = fit(x, y)
        s = fit_stats(x, y).sum(0)
        b2, sd2 = fit_from_sums(s[0], s[1], s[2], n)
        se_b = sigma / np.sqrt(s[0])            # sampling sd of the slope at this sigma
        good = (abs(b - beta) < 4 * se_b and abs(sd - sigma) < 0.02 * sigma
                and abs(b2 - b) < 1e-9 and abs(sd2 - sd) < 1e-9)
        ok &= good
        print(f"   beta {beta:.4f} sigma {sigma:.4f}: read {b:.4f} (sd {se_b:.4f}) {sd:.4f}, from sums "
              f"{b2:.4f} {sd2:.4f}  {'passes' if good else 'FAILS'}")

    print("\nS4. Three planted false steps among true ones, chains and decoys.")
    text = ("Janet has 16 - 3 - 4 = 9 eggs. She sells them: $9 \\times \\$2 = \\$18.\n"
            "Then 1,200 + 300 = 1,500 and 10 / 3 = 3.33, also 7 \u00d7 6 = 42, 48 \u00f7 4 = 12.\n"
            "**5 * 5 = 25** and 20 \u2212 5 = 15 and 3.5 + 2 = 5.5.\n"
            "Wrong: 12 \u00d7 12 = 140. Wrong: 1,000 + 250 = 1,260. Wrong: 45 / 9 = 5.04.\n"
            "Not steps: 2 \u00d7 3 + 4 = 10, (2 + 3) \u00d7 4 = 20, 3 + 4 = 7 \u00d7 2, 5 - 8 = -3, "
            "1/2 \u00d7 10 = 5, 50 \u00d7 20% = 10.")
    st = steps(text)
    false = [(a, op, b, c) for a, op, b, c, f in st if f]
    want_false = [("12", "\u00d7", "12", "140"), ("1000", "+", "250", "1260"), ("45", "/", "9", "5.04")]
    good = len(st) == 11 and false == want_false
    ok &= good
    print(f"   parsed {len(st)} steps (want 11), false {false}  {'passes' if good else 'FAILS'}")

    print("\n" + ("SELF-CHECK PASSES" if ok else "SELF-CHECK FAILS"))
    return ok


# ---------------------------------------------------------------------------
# The run.
# ---------------------------------------------------------------------------

def run(out):
    os.makedirs(out, exist_ok=True)
    print("DUMP READINGS, QUALITY PHASE 3 (O15). prereg "
          "proofs/preregistration-recherche-qualite3-dumps-2026-11-15.md")
    print(f"seed {SEED}, {DRAWS} bootstrap draws, percentile 95 % intervals\n")
    print("SELF-CHECK (run inside run mode)")
    if not selftest():
        print("self-check fails: no reading is computed")
        return 1

    inputs = list(MMLU.values()) + list(SEPT.values()) + list(GSM.values())
    with open(os.path.join(out, "inputs.sha256"), "w") as fh:
        for p in inputs:
            fh.write(f"{sha256(p)}  {p}\n")

    # ---------------- GSM8K: load, grade, gates ----------------
    print("\n==================== GSM8K ====================")
    gsm = {k: load_gsm(p) for k, p in GSM.items()}
    rows = {k: v[1] for k, v in gsm.items()}
    N = len(rows["f16"])
    same = all(len(rows[k]) == N for k in rows) and all(
        [r["index"] for r in rows[k]] == [r["index"] for r in rows["f16"]]
        and [r["qhash"] for r in rows[k]] == [r["qhash"] for r in rows["f16"]]
        and [r["gold_c"] for r in rows[k]] == [r["gold_c"] for r in rows["f16"]]
        for k in rows)
    print(f"GATE qhash: four dumps, {N} problems each, same index, qhash and gold in the same order: "
          f"{'passes' if same else 'FAILS'}")
    gate_qhash = same

    right = {k: np.array([r["correct_c"] for r in rows[k]], dtype=bool) for k in rows}
    want = {"control": 1082, "late": 1139, "early": 1177, "f16": 1215}
    cnt = {k: int(right[k].sum()) for k in right}
    disc = {
        "late": (int((right["late"] & ~right["control"]).sum()), int((~right["late"] & right["control"]).sum())),
        "early": (int((right["early"] & ~right["control"]).sum()), int((~right["early"] & right["control"]).sum())),
    }
    gate_grade = (cnt == want and disc["late"] == (108, 51) and disc["early"] == (143, 48))
    for k in ("control", "late", "early", "f16"):
        print(f"   {k:<8} right {cnt[k]} of {N} (gate {want[k]})")
    print(f"   late vs control: right in late only {disc['late'][0]}, in control only {disc['late'][1]} (gate 108, 51)")
    print(f"   early vs control: right in early only {disc['early'][0]}, in control only {disc['early'][1]} (gate 143, 48)")
    print(f"GATE grading: {'passes' if gate_grade else 'FAILS, G2, G3, G4 void'}")
    with open(os.path.join(out, "gsm8k_grades.tsv"), "w") as fh:
        fh.write("index\tqhash\tgold\t" + "\t".join(f"{k}_source\t{k}_extracted\t{k}_correct" for k in GSM) + "\n")
        for i in range(N):
            fh.write(f"{rows['f16'][i]['index']}\t{rows['f16'][i]['qhash']}\t{rows['f16'][i]['gold_c']}\t"
                     + "\t".join(f"{rows[k][i]['source_c']}\t{rows[k][i]['extracted_c']}\t{int(rows[k][i]['correct_c'])}"
                                 for k in GSM) + "\n")

    idx = boot_idx(N)

    # ---------------- G1 ----------------
    print("\n---- G1, the flip hazard ----")
    ref = [r["completion"] for r in rows["f16"]]
    g1 = {}
    for k in ("control", "late", "early"):
        rr = [flip_row(f, r["completion"]) for f, r in zip(ref, rows[k])]
        d = np.array([x[0] for x in rr], dtype=np.int64)
        c = np.array([x[1] for x in rr], dtype=bool)
        h, ne, ex = hazard(d, c)
        hb = hazard_boot(d, c, idx)
        g1[k] = dict(d=d, c=c, h=h, ne=ne, ex=ex, hb=hb)
    with open(os.path.join(out, "g1_flips.tsv"), "w") as fh:
        fh.write("index\tqhash\tlen_f16\t" + "\t".join(f"len_{k}\td_{k}\tcensored_{k}" for k in g1) + "\n")
        for i in range(N):
            fh.write(f"{rows['f16'][i]['index']}\t{rows['f16'][i]['qhash']}\t{len(ref[i])}\t"
                     + "\t".join(f"{len(rows[k][i]['completion'])}\t{g1[k]['d'][i]}\t{int(g1[k]['c'][i])}" for k in g1)
                     + "\n")
    g1_void = {}
    for k in g1:
        v = g1[k]
        g1_void[k] = v["ne"] < 200
        lo, hi = ci(v["hb"])
        print(f"   {k:<8} uncensored {v['ne']:4d} of {N}, sum d {v['ex']}, h = {1000 * v['h']:.4f} per 1000 "
              f"characters [{1000 * lo:.4f}, {1000 * hi:.4f}]"
              + ("   VOID, under 200 uncensored" if g1_void[k] else ""))
    for k in g1:
        d = g1[k]["d"]
        print(f"   {k:<8} median d {np.median(d):.1f}; share d >= 20: {100 * np.mean(d >= 20):.2f} %, "
              f">= 100: {100 * np.mean(d >= 100):.2f} %, >= 400: {100 * np.mean(d >= 400):.2f} %")
    rres = {}
    for k in ("late", "early"):
        r = 1 - g1[k]["h"] / g1["control"]["h"]
        rb = 1 - g1[k]["hb"] / g1["control"]["hb"]
        lo, hi = ci(rb)
        rres[k] = (r, lo, hi)
        void = g1_void[k] or g1_void["control"]
        print(f"   r_{k:<6} = {r:.4f} [{lo:.4f}, {hi:.4f}]" + ("   VOID" if void else ""))
    r, lo, hi = rres["late"]
    if lo > WEIGHT_SHARE_LATE:
        row = "the interval of r_late lies above 0.306"
    elif hi < WEIGHT_SHARE_LATE:
        row = "the interval of r_late lies below 0.306"
    else:
        row = "the interval of r_late contains 0.306"
    print(f"   G1 rule row: {row}")
    with open(os.path.join(out, "g1_summary.tsv"), "w") as fh:
        fh.write("arm\tuncensored\tsum_d\th\th_lo\th_hi\tmedian_d\tshare_d20\tshare_d100\tshare_d400\tr\tr_lo\tr_hi\n")
        for k in g1:
            v = g1[k]
            lo_, hi_ = ci(v["hb"])
            rr = rres.get(k, (float("nan"),) * 3)
            fh.write(f"{k}\t{v['ne']}\t{v['ex']}\t{v['h']:.8f}\t{lo_:.8f}\t{hi_:.8f}\t{np.median(v['d']):.1f}\t"
                     f"{np.mean(v['d'] >= 20):.6f}\t{np.mean(v['d'] >= 100):.6f}\t{np.mean(v['d'] >= 400):.6f}\t"
                     f"{rr[0]:.6f}\t{rr[1]:.6f}\t{rr[2]:.6f}\n")

    # ---------------- G2 ----------------
    print("\n---- G2, the chain's length (H7-B)" + ("" if gate_grade else ", VOID by the grading gate") + " ----")
    ngen = np.array([r["n_gen"] for r in rows["f16"]])
    order = np.lexsort((np.arange(N), ngen))          # by FP16 n_gen, ties by problem order
    group = np.empty(N, dtype=np.int64)
    group[order] = np.arange(N) * 3 // N
    dlc = right["late"].astype(np.float64) - right["control"].astype(np.float64)
    gains = []
    with open(os.path.join(out, "g2_groups.tsv"), "w") as fh:
        fh.write("group\tn\tngen_min\tngen_max\tcontrol_right\tlate_right\tgain_points\n")
        for g, name in enumerate(("short", "middle", "long")):
            m = group == g
            gain = 100 * dlc[m].mean()
            gains.append(gain)
            fh.write(f"{name}\t{m.sum()}\t{ngen[m].min()}\t{ngen[m].max()}\t{right['control'][m].sum()}\t"
                     f"{right['late'][m].sum()}\t{gain:.4f}\n")
            print(f"   {name:<6} n {m.sum()}, FP16 n_gen {ngen[m].min()}..{ngen[m].max()}, control "
                  f"{right['control'][m].sum()}, late {right['late'][m].sum()}, gain {gain:+.2f} points")
    S = gains[2] - gains[0]
    ms, ml = (group == 0).astype(np.float64), (group == 2).astype(np.float64)
    Sb = np.empty(DRAWS)
    for s in range(0, DRAWS, 1000):
        blk = idx[s:s + 1000]
        gl = (dlc[blk] * ml[blk]).sum(1) / ml[blk].sum(1)
        gs = (dlc[blk] * ms[blk]).sum(1) / ms[blk].sum(1)
        Sb[s:s + 1000] = 100 * (gl - gs)
    slo, shi = ci(Sb)
    print(f"   S = gain(long) - gain(short) = {S:+.2f} points [{slo:+.2f}, {shi:+.2f}]")
    rows_g2 = []
    if slo > 0:
        rows_g2.append("the interval of S lies above 0")
    if shi < 3:
        rows_g2.append("the interval of S lies below +3 points")
    if not rows_g2:
        rows_g2.append("otherwise")
    print(f"   G2 rule row(s): {' ; '.join(rows_g2)}")

    # ---------------- G3 ----------------
    print("\n---- G3, false arithmetic (H7-D)" + ("" if gate_grade else ", VOID by the grading gate") + " ----")
    st_ctrl = [steps(r["completion"]) for r in rows["control"]]
    nst = np.array([len(s) for s in st_ctrl])
    nfalse = np.array([sum(1 for x in s if x[4]) for s in st_ctrl])
    coverage = float(np.mean(nst > 0))
    falsec = nfalse > 0
    rep_l = right["late"] & ~right["control"]
    rep_e = right["early"] & ~right["control"]
    with open(os.path.join(out, "g3_control_steps.tsv"), "w") as fh:
        fh.write("index\tqhash\tn_steps\tn_false\trepaired_late\trepaired_early\n")
        for i in range(N):
            fh.write(f"{rows['control'][i]['index']}\t{rows['control'][i]['qhash']}\t{nst[i]}\t{nfalse[i]}\t"
                     f"{int(rep_l[i])}\t{int(rep_e[i])}\n")
    with open(os.path.join(out, "g3_false_steps.tsv"), "w") as fh:
        fh.write("index\ta\top\tb\tc\n")
        for i, s in enumerate(st_ctrl):
            for a, op, b, c, f in s:
                if f:
                    fh.write(f"{rows['control'][i]['index']}\t{a}\t{op}\t{b}\t{c}\n")
    g3_void = coverage < 0.5
    print(f"   coverage: {100 * coverage:.2f} % of the control's completions hold at least one parsed step "
          f"({int((nst > 0).sum())} of {N}); {int(nst.sum())} steps, {int(nfalse.sum())} false; "
          f"{int(falsec.sum())} false completions")
    print(f"GATE G3 coverage >= 50 %: {'passes' if not g3_void else 'FAILS, G3 void'}")
    sl = falsec[rep_l].mean()
    se = falsec[rep_e].mean()
    diff = 100 * (sl - se)
    fl, fe, rl, re_ = falsec.astype(float), falsec.astype(float), rep_l.astype(float), rep_e.astype(float)
    Db = np.empty(DRAWS)
    for s in range(0, DRAWS, 1000):
        blk = idx[s:s + 1000]
        a = (fl[blk] * rl[blk]).sum(1) / rl[blk].sum(1)
        b = (fe[blk] * re_[blk]).sum(1) / re_[blk].sum(1)
        Db[s:s + 1000] = 100 * (a - b)
    dlo, dhi = ci(Db)
    print(f"   false share among the control's completions the late arm repairs: {int(falsec[rep_l].sum())} of "
          f"{int(rep_l.sum())} = {100 * sl:.2f} %")
    print(f"   false share among the control's completions the early arm repairs: {int(falsec[rep_e].sum())} of "
          f"{int(rep_e.sum())} = {100 * se:.2f} %")
    print(f"   difference late-repaired minus early-repaired = {diff:+.2f} points [{dlo:+.2f}, {dhi:+.2f}]")
    print(f"   beside: false share over all control completions {100 * falsec.mean():.2f} %, over the "
          f"control's wrong ones {100 * falsec[~right['control']].mean():.2f} %, right ones "
          f"{100 * falsec[right['control']].mean():.2f} %")
    rows_g3 = []
    if dlo > 0:
        rows_g3.append("the interval of the difference of shares, late-repaired minus early-repaired, lies above 0")
    if dhi < 10:
        rows_g3.append("it lies below +10 points")
    if not rows_g3:
        rows_g3.append("otherwise")
    print(f"   G3 rule row(s): {' ; '.join(rows_g3)}")

    # ---------------- G4 ----------------
    print("\n---- G4" + ("" if gate_grade else ", VOID by the grading gate") + " ----")
    both = int((rep_l & rep_e).sum())
    print(f"   both arms repair {both} problems, of {int(rep_l.sum())} (late) and {int(rep_e.sum())} (early)")

    # ---------------- MMLU ----------------
    print("\n==================== MMLU ====================")
    mm = {k: load_mmlu(p) for k, p in MMLU.items()}
    sp = {k: load_mmlu(p) for k, p in SEPT.items()}
    keys = list(mm["f16"].keys())
    acc = {k: 100 * np.mean([mm[k][q][3] for q in keys]) for k in mm}
    gl = 100 * (sum(mm["late"][q][3] for q in keys) - sum(mm["control"][q][3] for q in keys)) / len(keys)
    ge = 100 * (sum(mm["early"][q][3] for q in keys) - sum(mm["control"][q][3] for q in keys)) / len(keys)
    xf = np.array([mm["f16"][q][0] for q in keys])
    bc, sc = fit(xf, np.array([mm["control"][q][0] for q in keys]))
    gate_m = (all(q in mm[k] for k in mm for q in keys) and len(keys) == 14042
              and round(acc["control"], 2) == 63.37 and round(acc["f16"], 2) == 70.14
              and round(gl, 2) == 0.43 and round(ge, 2) == 6.12
              and round(bc, 4) == 0.5817 and round(sc, 4) == 1.4205)
    print(f"   {len(keys)} questions; micro accuracy f16 {acc['f16']:.2f} (gate 70.14), control "
          f"{acc['control']:.2f} (gate 63.37), late {acc['late']:.2f}, early {acc['early']:.2f}")
    print(f"   paired gains late - control {gl:+.2f} (gate +0.43), early - control {ge:+.2f} (gate +6.12)")
    print(f"   control fit beta {bc:.4f} (gate 0.5817), sd_res {sc:.4f} (gate 1.4205)")
    print(f"GATE MMLU: {'passes' if gate_m else 'FAILS, M1 and M2 void'}")

    print("\n---- M1, slope and noise ----")
    arms = [("control", mm["control"]), ("late", mm["late"]), ("early", mm["early"]), ("f16 (self)", mm["f16"])]
    skeys = [q for q in keys if all(q in sp[k] for k in sp)]
    sx = np.array([mm["f16"][q][0] for q in skeys])
    fits = []
    for name, d in arms:
        y = np.array([d[q][0] for q in keys])
        fits.append((name, "4b-depth", keys, xf, y, 100 * np.mean([d[q][3] for q in keys])))
    for name, d in sp.items():
        y = np.array([d[q][0] for q in skeys])
        fits.append((name, "september", skeys, sx, y, 100 * np.mean([d[q][3] for q in skeys])))
    stats = {}
    for name, grp, ks, x, y, a in fits:
        stats[name] = fit_stats(x, y)
    # Bootstrap over questions: counts of each draw, same seed. The two families share n.
    nq = len(keys)
    assert len(skeys) == nq
    rng = np.random.default_rng(SEED)
    names = [f[0] for f in fits]
    S3 = np.concatenate([stats[n] for n in names], 1)       # (nq, 3 * len(names))
    boot = np.empty((DRAWS, S3.shape[1]))
    for s in range(0, DRAWS, 250):
        m = min(250, DRAWS - s)
        ii = rng.integers(0, nq, size=(m, nq))
        C = np.stack([np.bincount(r, minlength=nq) for r in ii]).astype(np.float64)
        boot[s:s + m] = C @ S3
    res = {}
    with open(os.path.join(out, "m1_fits.tsv"), "w") as fh:
        fh.write("dump\tfamily\tn\tmmlu\tbeta\tbeta_lo\tbeta_hi\tsd_res\tsd_lo\tsd_hi\tratio\tratio_lo\tratio_hi\n")
        for j, (name, grp, ks, x, y, a) in enumerate(fits):
            b, sd = fit(x, y)
            bb, sdb = fit_from_sums(boot[:, 3 * j], boot[:, 3 * j + 1], boot[:, 3 * j + 2], len(ks))
            with np.errstate(divide="ignore", invalid="ignore"):
                rb = bb / sdb
                ratio = b / sd if sd > 0 else float("inf")
            res[name] = (b, sd, ratio, bb, sdb, rb)
            if sd > 1e-12:
                blo, bhi = ci(bb); slo_, shi_ = ci(sdb); rlo, rhi = ci(rb)
            else:
                blo = bhi = b; slo_ = shi_ = sd; rlo = rhi = float("inf")
            fh.write(f"{name}\t{grp}\t{len(ks)}\t{a:.4f}\t{b:.6f}\t{blo:.6f}\t{bhi:.6f}\t{sd:.6f}\t{slo_:.6f}\t"
                     f"{shi_:.6f}\t{ratio:.6f}\t{rlo:.6f}\t{rhi:.6f}\n")
            print(f"   {name:<14} {grp:<9} n {len(ks)} MMLU {a:6.2f}  beta {b:.4f} [{blo:.4f}, {bhi:.4f}]  "
                  f"sd_res {sd:.4f} [{slo_:.4f}, {shi_:.4f}]  beta/sd_res {ratio:.4f} [{rlo:.4f}, {rhi:.4f}]")
    print("   beside, no rule:")
    for a_, b_ in (("late", "control"), ("early", "control")):
        d = res[a_][5] - res[b_][5]
        lo_, hi_ = ci(d)
        print(f"   beta/sd_res {a_} - {b_}: {res[a_][2] - res[b_][2]:+.4f} [{lo_:+.4f}, {hi_:+.4f}]; "
              f"relative {100 * (res[a_][2] / res[b_][2] - 1):+.2f} %")
    for a_ in ("q5-down-0-11", "q5-down-12-23", "q5-down-24-35"):
        d = res[a_][3] - res["q5-shipped"][3]
        lo_, hi_ = ci(d)
        print(f"   beta {a_} - q5-shipped: {res[a_][0] - res['q5-shipped'][0]:+.4f} [{lo_:+.4f}, {hi_:+.4f}]")
    d = res["q5-down-24-35"][3] - res["q5-down-0-11"][3]
    lo_, hi_ = ci(d)
    print(f"   beta q5-down-24-35 - q5-down-0-11: {res['q5-down-24-35'][0] - res['q5-down-0-11'][0]:+.4f} "
          f"[{lo_:+.4f}, {hi_:+.4f}]")

    print("\n---- M2, the flips of the late arm ----")
    ch = [q for q in keys if mm["late"][q][2] != mm["control"][q][2]]
    won = [q for q in ch if mm["late"][q][3] and not mm["control"][q][3]]
    lost = [q for q in ch if mm["control"][q][3] and not mm["late"][q][3]]
    neither = len(ch) - len(won) - len(lost)

    def margin(q):
        L = np.sort(mm["f16"][q][0])[::-1]
        return L[0] - L[1]
    mall = np.array([margin(q) for q in keys])
    mch = np.array([margin(q) for q in ch])
    print(f"   the late arm changes the pick on {len(ch)} of {len(keys)} questions ({100 * len(ch) / len(keys):.2f} %)")
    print(f"   won {len(won)} ({100 * len(won) / len(ch):.2f} %), lost {len(lost)} ({100 * len(lost) / len(ch):.2f} %), "
          f"both wrong {neither} ({100 * neither / len(ch):.2f} %)")
    print(f"   FP16 margin (top logit minus second): on changed questions mean {mch.mean():.4f}, median "
          f"{np.median(mch):.4f}; on all questions mean {mall.mean():.4f}, median {np.median(mall):.4f}")
    with open(os.path.join(out, "m2_changed.tsv"), "w") as fh:
        fh.write("subject\tindex\tanswer\tpick_control\tpick_late\toutcome\tf16_margin\n")
        for q in ch:
            o = "won" if q in set(won) else ("lost" if q in set(lost) else "both_wrong")
            fh.write(f"{q[0]}\t{q[1]}\t{mm['f16'][q][1]}\t{mm['control'][q][2]}\t{mm['late'][q][2]}\t{o}\t"
                     f"{margin(q):.6f}\n")

    print("\n==================== GATES ====================")
    print(f"   grading {'passes' if gate_grade else 'FAILS'}; MMLU {'passes' if gate_m else 'FAILS'}; qhash "
          f"{'passes' if gate_qhash else 'FAILS'}; G1 uncensored >= 200: "
          + ", ".join(f"{k} {'passes' if not g1_void[k] else 'FAILS'}" for k in g1)
          + f"; G3 coverage {'passes' if not g3_void else 'FAILS'}; self-check passes")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "selftest":
        raise SystemExit(0 if selftest() else 1)
    if len(sys.argv) == 3 and sys.argv[1] == "run":
        raise SystemExit(run(sys.argv[2]))
    print(__doc__)
    raise SystemExit(2)
