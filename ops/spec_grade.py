# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Grade the math prompts of a `specbench --dump` file offline.

    uv run ops/spec_grade.py <dump.jsonl> [--prompts <prompts.jsonl>] [--pair 4e pass]
    uv run ops/spec_grade.py --selftest

The dump holds one JSON line a prompt and decoder (`4e`, `pass`, then each
drafter): id, kind, decoder, tokens, text. The gold comes from a `gold` field
of the dump line if it has one, else from the prompt file's line of the same
id (`ops/data/spec-gsm8k-200.jsonl` carries it); a prompt with no gold is
counted for its boxes only.

The rules are the repository's (`llvq-llm/src/gsm8k.rs`), ported line for
line: the answer is the last number inside the last closed `\\boxed{...}`
(`\\frac{a}{b}` read as a / b when b divides a, as no number otherwise);
with no closed box, the last number of the completion; both sides reduced to
one decimal spelling and compared as strings, no tolerance.

Printed: per decoder, the math prompts, the closed boxes, the correct answers;
the paired table of two decoders (default `4e` against `pass`): both right,
both wrong, only the first right, only the second right; the exact McNemar
p-value (two-sided binomial on the discordant pairs); and every prompt where
the two decoders' answers differ.
"""

import argparse
import json
import math
import sys

def numbers(s):
    """Every number of `s` in order: `numbers` of gsm8k.rs (a minus right
    before the first digit, commas only between digits, a decimal part)."""
    out = []
    i, b = 0, s
    while i < len(b):
        if not b[i].isdigit() or not b[i].isascii():
            i += 1
            continue
        start = i - 1 if i > 0 and b[i - 1] == "-" else i
        j = i
        while j < len(b) and ((b[j].isascii() and b[j].isdigit()) or (b[j] == "," and j + 1 < len(b) and b[j + 1].isascii() and b[j + 1].isdigit())):
            j += 1
        if j + 1 < len(b) and b[j] == "." and b[j + 1].isascii() and b[j + 1].isdigit():
            j += 1
            while j < len(b) and b[j].isascii() and b[j].isdigit():
                j += 1
        out.append(b[start:j])
        i = j
    return out


def canonical(raw):
    t = raw.strip().replace(",", "")
    neg = t.startswith("-")
    body = t[1:] if neg else t
    i, _, f = body.partition(".")
    if not i and not f:
        return None
    if not all(c in "0123456789" for c in i) or not all(c in "0123456789" for c in f):
        return None
    i = i.lstrip("0") or "0"
    f = f.rstrip("0")
    zero = i == "0" and not f
    return ("-" if neg and not zero else "") + i + ("." + f if f else "")


def last_boxed(s):
    key = "\\boxed{"
    p = s.rfind(key)
    if p < 0:
        return None
    start = p + len(key)
    depth = 1
    for k in range(start, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[start:k]
    return None


def frac_value(inner):
    """(True, value or None) for a fraction, (False, None) when there is none."""
    best = None
    for key in ("\\dfrac{", "\\tfrac{", "\\frac{"):
        p = inner.find(key)
        if p >= 0:
            best = p + len(key)
            break
    if best is None:
        return False, None
    rest = inner[best:]
    if "}" not in rest:
        return False, None
    num, rest = rest.split("}", 1)
    if not rest.startswith("{") or "}" not in rest[1:]:
        return False, None
    den = rest[1:].split("}", 1)[0]

    def parse(x):
        c = canonical(x)
        return None if c is None or "." in c else int(c)

    a, b = parse(num), parse(den)
    if a is None or b is None or b == 0 or a % b != 0:
        return True, None
    # i128 division toward zero, as in Rust
    q = abs(a) // abs(b)
    return True, str(-q if (a < 0) != (b < 0) else q)


def extract(text):
    """(source, answer): source is boxed, last or none."""
    inner = last_boxed(text)
    if inner is not None:
        is_frac, v = frac_value(inner)
        if is_frac:
            return "boxed", v
        ns = numbers(inner)
        if ns:
            return "boxed", canonical(ns[-1])
    ns = numbers(text)
    if ns:
        return "last", canonical(ns[-1])
    return "none", None


def mcnemar_exact(b, c):
    """Two-sided exact McNemar: 2 * P(X <= min(b, c)), X ~ Bin(b + c, 1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def grade(dump, prompts=None, pair=("4e", "pass"), out=sys.stdout):
    gold = {}
    for r in prompts or []:
        if "id" in r and r.get("gold") is not None:
            gold[r["id"]] = canonical(str(r["gold"]))
    answers = {}  # decoder -> id -> (source, answer)
    order, ids = [], []
    for r in dump:
        if r.get("kind") != "math":
            continue
        if r.get("gold") is not None:
            gold[r["id"]] = canonical(str(r["gold"]))
        d = r["decoder"]
        if d not in answers:
            answers[d] = {}
            order.append(d)
        if r["id"] not in ids:
            ids.append(r["id"])
        answers[d][r["id"]] = extract(r["text"])
    summary = {}
    print(f"{len(ids)} math prompts, {sum(1 for i in ids if i in gold)} with a gold", file=out)
    for d in order:
        a = answers[d]
        boxed = sum(1 for s, _ in a.values() if s == "boxed")
        graded = [i for i in a if i in gold]
        right = sum(1 for i in graded if a[i][1] is not None and a[i][1] == gold[i])
        summary[d] = (len(a), boxed, right, len(graded))
        print(f"  {d:>14}: {len(a)} prompts, {boxed} closed boxes, {right} of {len(graded)} correct", file=out)
    x, y = pair
    if x not in answers or y not in answers:
        print(f"pair {x} / {y}: not both in the dump (decoders: {', '.join(order)})", file=out)
        return summary, None
    both = [i for i in ids if i in answers[x] and i in answers[y]]
    graded = [i for i in both if i in gold]
    ok = lambda d, i: answers[d][i][1] is not None and answers[d][i][1] == gold[i]
    rr = sum(1 for i in graded if ok(x, i) and ok(y, i))
    ww = sum(1 for i in graded if not ok(x, i) and not ok(y, i))
    rw = sum(1 for i in graded if ok(x, i) and not ok(y, i))
    wr = sum(1 for i in graded if not ok(x, i) and ok(y, i))
    p = mcnemar_exact(rw, wr)
    print(f"paired {x} / {y} on {len(graded)} graded prompts: both right {rr}, both wrong {ww}, only {x} right {rw}, only {y} right {wr}; exact McNemar p = {p:.4g}", file=out)
    diff = [i for i in both if answers[x][i][1] != answers[y][i][1]]
    print(f"answers differ on {len(diff)} prompts:", file=out)
    for i in diff:
        g = gold.get(i, "-")
        print(f"  {i}: {x} {answers[x][i][1]} ({answers[x][i][0]}), {y} {answers[y][i][1]} ({answers[y][i][0]}), gold {g}", file=out)
    return summary, (rr, ww, rw, wr, p, diff)


def selftest():
    assert numbers("1,000 and 3, 4 and -5 and 2.50.") == ["1,000", "3", "4", "-5", "2.50"]
    assert canonical("1,000.50") == "1000.5" and canonical("-0.0") == "0" and canonical("007") == "7" and canonical("x") is None
    assert last_boxed("a \\boxed{1} b \\boxed{\\frac{6}{3}}") == "\\frac{6}{3}"
    assert last_boxed("\\boxed{12") is None
    assert extract("so \\boxed{\\frac{6}{3}}") == ("boxed", "2")
    assert extract("so \\boxed{\\frac{7}{3}}") == ("boxed", None)
    assert extract("so \\boxed{\\$1,200.00}") == ("boxed", "1200")
    assert extract("cut \\boxed{12") == ("last", "12")
    assert extract("nothing") == ("none", None)
    assert abs(mcnemar_exact(0, 0) - 1.0) < 1e-12
    assert abs(mcnemar_exact(0, 5) - 2 / 32) < 1e-12
    assert abs(mcnemar_exact(2, 8) - 2 * (1 + 10 + 45) / 1024) < 1e-12
    prompts = [{"_comment": "x"}, {"id": "a", "kind": "math", "prompt": "?", "gold": "18"}, {"id": "b", "kind": "math", "prompt": "?", "gold": "3"},
               {"id": "c", "kind": "math", "prompt": "?", "gold": "70000"}, {"id": "d", "kind": "chat", "prompt": "?"}]
    dump = []
    texts = {
        "4e": {"a": "\\boxed{18}", "b": "\\boxed{3}", "c": "is 7 then", "d": "hi"},
        "pass": {"a": "\\boxed{18}", "b": "\\boxed{4}", "c": "\\boxed{70,000}", "d": "hi"},
        "dflash:tree": {"a": "\\boxed{18}", "b": "\\boxed{4}", "c": "\\boxed{70,000}", "d": "hi"},
    }
    for i in "abcd":
        for d, t in texts.items():
            dump.append({"id": i, "kind": "chat" if i == "d" else "math", "decoder": d, "tokens": [1], "text": t[i]})
    import io

    buf = io.StringIO()
    summary, paired = grade(dump, prompts, ("4e", "pass"), buf)
    assert summary["4e"] == (3, 2, 2, 3), summary
    assert summary["pass"] == (3, 3, 2, 3), summary
    rr, ww, rw, wr, p, diff = paired
    assert (rr, ww, rw, wr) == (1, 0, 1, 1) and p == 1.0 and diff == ["b", "c"], paired
    # a gold carried by the dump line wins without a prompt file
    dump2 = [{"id": "z", "kind": "math", "decoder": "4e", "tokens": [], "text": "\\boxed{5}", "gold": "5"}]
    s2, _ = grade(dump2, None, ("4e", "pass"), io.StringIO())
    assert s2["4e"] == (1, 1, 1, 1)
    print("selftest: ok")


def main():
    ap = argparse.ArgumentParser(description="grade the math prompts of a specbench --dump file")
    ap.add_argument("dump", nargs="?")
    ap.add_argument("--prompts", help="the prompt file, for its gold fields")
    ap.add_argument("--pair", nargs=2, default=["4e", "pass"], metavar=("A", "B"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.dump:
        ap.error("a dump file, or --selftest")
    grade(read_jsonl(a.dump), read_jsonl(a.prompts) if a.prompts else None, tuple(a.pair))


if __name__ == "__main__":
    main()
