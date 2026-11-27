"""The readings of lot H from the raw files of all.sh. Standard library only.  python3 -I readings.py <raw dir>
Tokens: each line of an /api/chat stream carries one generated token's text (thinking, content or a call), so two
answers are the same token for token when their sequences of per-line deltas are equal; the first differing line is
the first differing position."""
import json, os, statistics, sys

D = sys.argv[1]


def deltas(label, req):
    out = []
    for line in open(os.path.join(D, f"{label}.{req}.raw.txt")):
        t, _, text = line.rstrip("\n").partition("\t")
        if not text.strip():
            continue
        c = json.loads(text)
        if c.get("done"):
            continue
        m = c.get("message") or {}
        out.append((m.get("thinking") or "", m.get("content") or "", json.dumps(m.get("tool_calls"), sort_keys=True)))
    return out


def stream(label, req):
    return json.load(open(os.path.join(D, f"{label}.{req}.json")))


def compare(a, b, req):
    x, y = deltas(a, req), deltas(b, req)
    first = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), None)
    if first is None and len(x) != len(y):
        first = min(len(x), len(y))
    sa, sb = stream(a, req), stream(b, req)
    same_text = (sa["thinking"], sa["content"], sa["tool_calls"]) == (sb["thinking"], sb["content"], sb["tool_calls"])
    verdict = "same tokens" if first is None else f"first differing token at position {first}"
    return f"{req:5s} {a:11s} vs {b:11s}: {len(x)} / {len(y)} tokens, {verdict}, text equal {same_text}"


def load_lines(label):
    return [l.rstrip("\n") for l in open(os.path.join(D, f"{label}.server.log")) if l.startswith("llvq: inkling2:mini: ")]


def main():
    print("== H-i: the default against the committed build ==")
    for req in ("u1", "chat"):
        print(compare("hi-before", "hi-after", req))
        for r in range(4):
            print(compare("hi-before", f"p{r}-full", req))
    base = load_lines("hi-before")
    for lab in ["hi-after"] + [f"p{r}-full" for r in range(4)]:
        print(f"load lines {lab} vs hi-before: {'identical' if load_lines(lab) == base else 'DIFFER'} ({len(load_lines(lab))} lines)")
    print("== H-ii: narrow against full ==")
    for req in ("u1", "chat"):
        for r in range(4):
            print(compare(f"p{r}-full", f"p{r}-narrow", req))
        for r in range(1, 4):
            print(compare("p0-narrow", f"p{r}-narrow", req))
    for r in range(4):
        n = load_lines(f"p{r}-narrow")
        extra = [l for l in n if l not in base]
        print(f"load lines p{r}-narrow: {len(n)}, all of hi-before's present {all(l in n for l in base)}, added {extra}")
    print("== H-iii: first token of u1, cold, a fresh server each, full then narrow ==")
    rows = []
    for r in range(4):
        f, n = stream(f"p{r}-full", "u1"), stream(f"p{r}-narrow", "u1")
        rf = (f["final"] or {}).get("prompt_eval_duration", 0) / 1e9
        rn = (n["final"] or {}).get("prompt_eval_duration", 0) / 1e9
        tag = "discarded" if r == 0 else "timed"
        print(f"pair {r} ({tag}): full {f['first_token_s']:.2f} s (read {rf:.2f}), narrow {n['first_token_s']:.2f} s "
              f"(read {rn:.2f}), ratio {n['first_token_s'] / f['first_token_s']:.4f}, prompt {(f['final'] or {}).get('prompt_eval_count')}")
        if r:
            rows.append((f["first_token_s"], n["first_token_s"], rf, rn))
    for i, name in ((0, "first token full"), (1, "first token narrow"), (2, "read full"), (3, "read narrow")):
        v = [x[i] for x in rows]
        print(f"{name}: median {statistics.median(v):.2f} s [{min(v):.2f}, {max(v):.2f}]")
    rat = [x[1] / x[0] for x in rows]
    print(f"pair ratio narrow / full: median {statistics.median(rat):.4f} [{min(rat):.4f}, {max(rat):.4f}], pairs {[round(x, 4) for x in rat]}")
    rr = [x[3] / x[2] for x in rows]
    print(f"pair ratio of the server's read: median {statistics.median(rr):.4f} [{min(rr):.4f}, {max(rr):.4f}]")
    for lab in ["hi-before", "hi-after"]:
        s = stream(lab, "u1")
        print(f"{lab}: u1 first token {s['first_token_s']:.2f} s")
    print("== H-iv: the cache does not cross ==")
    for lab in ("iv1-full", "iv2-narrow", "iv3-full"):
        s = stream(lab, "u1")
        f = s["final"] or {}
        pl = [l.rstrip() for l in open(os.path.join(D, f"{lab}.server.log")) if "prefix" in l]
        print(f"{lab}: cached {f.get('prompt_eval_cached_count')} of {f.get('prompt_eval_count')}, first token {s['first_token_s']:.2f} s")
        for l in pl:
            print(f"  {l}")
    print(compare("iv1-full", "p1-full", "u1"))
    print(compare("iv2-narrow", "p1-narrow", "u1"))
    print(compare("iv3-full", "p1-full", "u1"))


main()
