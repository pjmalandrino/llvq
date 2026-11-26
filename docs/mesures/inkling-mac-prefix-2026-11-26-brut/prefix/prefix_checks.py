"""The requests and readings of ops/inkling/serve-prefix-mac.sh: the prefix cache on disk of `llvq serve`.

The static block is the long prompt of serve_checks.py (a system prompt of numbered sections and 15 tools, built the
same way on every run, about 8,770 tokens once rendered); each request adds one short user message, like a coding
agent's first turn. Standard library only.

  prefix_checks.py req <name>                                  prints the request body (/api/chat, streamed)
  prefix_checks.py ndjson <t0> <raw out> <json out>            reads an /api/chat stream on stdin, timestamps lines
  prefix_checks.py check <name> <files...>                     one line PASS or FAIL into $O/summary.txt
  prefix_checks.py final                                       exit 1 when a correctness check failed
"""
import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("serve_checks", os.path.join(HERE, "serve_checks.py"))
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)

MODEL = os.environ.get("MODEL", "inkling2-mini:local")
O = os.environ.get("O", ".")
# The save point the static block above gives: the last multiple of 512 tokens inside it.
B = int(os.environ.get("PREFIX_B", "8704"))

USERS = {
    "u1": sc.LONG_USER,
    "u2": "Do not call any tool. In one short sentence: what does section 12 of the policy ask for?",
    "u3": "Do not call any tool. In one short sentence: what does section 21 of the policy ask for?",
}


def system(changed=False):
    s = sc.long_system()
    if changed:
        # one word of the first sentence
        assert "small research team" in s
        s = s.replace("small research team", "large research team", 1)
    return s


def req(name):
    """`<user>` or `<user>-changed`: the static block (changed by one word) and that user message."""
    user, _, changed = name.partition("-")
    return {"model": MODEL, "stream": True, "keep_alive": -1, "tools": sc.long_tools(),
            "messages": [{"role": "system", "content": system(changed == "changed")},
                         {"role": "user", "content": USERS[user]}],
            "options": {"num_predict": 48, "temperature": 0}}


def ndjson(t0, raw_path, out_path):
    """Read an /api/chat stream: every line kept raw with its arrival time; the content joined; the final counts."""
    t0 = float(t0)
    raw = open(raw_path, "w")
    s = {"lines": 0, "first_token_s": None, "last_s": None, "content": "", "thinking": "", "tool_calls": [],
         "final": None, "errors": []}
    for line in iter(sys.stdin.buffer.readline, b""):
        t = time.time() - t0
        text = line.decode("utf-8", "replace").rstrip("\n")
        raw.write(f"{t:.4f}\t{text}\n")
        if not text.strip():
            continue
        s["lines"] += 1
        s["last_s"] = t
        try:
            c = json.loads(text)
        except ValueError:
            s["errors"].append(f"not JSON: {text[:200]}")
            continue
        if "error" in c:
            s["errors"].append(c["error"])
            continue
        m = c.get("message") or {}
        if (m.get("content") or m.get("thinking") or m.get("tool_calls")) and s["first_token_s"] is None:
            s["first_token_s"] = t
        s["content"] += m.get("content") or ""
        s["thinking"] += m.get("thinking") or ""
        s["tool_calls"] += m.get("tool_calls") or []
        if c.get("done"):
            s["final"] = {k: v for k, v in c.items() if k != "message"}
    raw.close()
    json.dump(s, open(out_path, "w"), indent=1)


def record(name, kind, ok, detail, seconds=None):
    line = f"{'PASS' if ok else 'FAIL'} {name} [{kind}] {detail}"
    if seconds is not None:
        line += f" | {seconds:.2f} s"
    print(line)
    with open(f"{O}/summary.txt", "a") as f:
        f.write(line + "\n")
    with open(f"{O}/checks.jsonl", "a") as f:
        f.write(json.dumps({"check": name, "kind": kind, "pass": bool(ok), "detail": detail, "seconds": seconds}) + "\n")


def read(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError) as e:
        return {"_unreadable": str(e)}


def answer(s):
    """What a client reads of the turn: the text and the calls."""
    return json.dumps({"content": s.get("content"), "tool_calls": s.get("tool_calls")}, sort_keys=True)


def counts(s):
    f = s.get("final") or {}
    return f.get("prompt_eval_count") or 0, f.get("prompt_eval_cached_count"), (f.get("prompt_eval_duration") or 0) / 1e9


def answered(s):
    return "_unreadable" not in s and not s.get("errors") and s.get("final") is not None and \
        bool(s.get("content") or s.get("tool_calls"))


def prefix_files(home):
    d = os.path.join(home, "prefix")
    return sorted(f for f in os.listdir(d) if f.endswith(".llvqprefix")) if os.path.isdir(d) else []


def check(name, *a):
    if name == "cold-save":
        # P-a: <stream json> <home> <server log>
        s, home, log = read(a[0]), a[1], open(a[2], errors="replace").read()
        n, cached, pe = counts(s)
        files = prefix_files(home)
        saved = [x for x in log.splitlines() if "prefix cache: saved" in x]
        ok = answered(s) and cached == 0 and len(files) == 1 and f"-{B:08d}-" in files[0] and len(saved) == 1 \
            and f"saved {B} tokens" in saved[0]
        record("P-a-cold-save", "correctness", ok, f"prompt {n}, cached {cached}, read {pe:.2f} s, files {files}, "
               f"log {saved}", s.get("first_token_s"))
    elif name == "loaded":
        # P-b / P-c: <label> <stream json> <reference stream json> <server log> <loaded lines before>
        label, s, ref, log, before = a[0], read(a[1]), read(a[2]), open(a[3], errors="replace").read(), int(a[4])
        n, cached, pe = counts(s)
        loaded = [x for x in log.splitlines() if "prefix cache: loaded" in x]
        ok = answered(s) and cached is not None and cached >= B and len(loaded) == before + 1
        record(f"{label}-loaded", "correctness", ok, f"prompt {n}, cached {cached} (at least {B}), read {pe:.2f} s, "
               f"log {loaded[before:]}", s.get("first_token_s"))
        same = answered(ref) and answer(s) == answer(ref)
        record(f"{label}-same-answer", "correctness", same,
               f"answer {answer(s)[:120]!r} against the server without the cache {answer(ref)[:120]!r}")
    elif name == "same":
        # <label> <stream json> <reference stream json>
        label, s, ref = a[0], read(a[1]), read(a[2])
        same = answered(s) and answered(ref) and answer(s) == answer(ref)
        record(f"{label}-same-answer", "correctness", same, f"answer {answer(s)[:120]!r} against {answer(ref)[:120]!r}")
    elif name == "damaged":
        # P-d: <stream json> <reference stream json> <server log> <file name>
        s, ref, log, fname = read(a[0]), read(a[1]), open(a[2], errors="replace").read(), a[3]
        n, cached, pe = counts(s)
        refused = [x for x in log.splitlines() if "prefix cache:" in x and "refused" in x]
        ok = answered(s) and cached == 0 and len(refused) == 1 and fname in refused[0] and "damaged" in refused[0]
        record("P-d-damaged-refused", "correctness", ok, f"prompt {n}, cached {cached}, read {pe:.2f} s, log {refused}",
               s.get("first_token_s"))
        same = answered(ref) and answer(s) == answer(ref)
        record("P-d-same-answer", "correctness", same, f"answer {answer(s)[:120]!r} against {answer(ref)[:120]!r}")
    elif name == "changed":
        # P-e: <stream json> <server log>
        s, log = read(a[0]), open(a[1], errors="replace").read()
        n, cached, pe = counts(s)
        loaded = [x for x in log.splitlines() if "prefix cache: loaded" in x]
        ok = answered(s) and cached == 0 and not loaded
        record("P-e-changed-no-load", "correctness", ok, f"prompt {n}, cached {cached}, read {pe:.2f} s, loads {loaded}",
               s.get("first_token_s"))
    elif name == "off":
        # P-f: <home> <server log> <stream json>...
        home, log, streams = a[0], open(a[1], errors="replace").read(), [read(x) for x in a[2:]]
        cs = [counts(s)[1] for s in streams]
        lines = [x for x in log.splitlines() if "prefix cache" in x]
        ok = all(answered(s) for s in streams) and all(c == 0 for c in cs) and not os.path.exists(
            os.path.join(home, "prefix")) and lines == [x for x in lines if "prefix cache: off" in x] and len(lines) == 1
        record("P-f-off", "correctness", ok, f"cached {cs}, prefix dir exists {os.path.exists(os.path.join(home, 'prefix'))}, "
               f"log {lines}")
    else:
        sys.exit(f"unknown check {name}")


def final():
    rows = [json.loads(line) for line in open(f"{O}/checks.jsonl") if line.strip()]
    bad = [r["check"] for r in rows if r["kind"] == "correctness" and not r["pass"]]
    line = f"correctness: {sum(r['kind'] == 'correctness' for r in rows) - len(bad)} PASS, {len(bad)} FAIL {bad}"
    print(line)
    with open(f"{O}/summary.txt", "a") as f:
        f.write(line + "\n")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "req":
        print(json.dumps(req(rest[0])))
    elif cmd == "ndjson":
        ndjson(*rest)
    elif cmd == "check":
        check(*rest)
    elif cmd == "show":
        s = read(rest[1])
        f = s.get("final") or {}
        print(f"{rest[0]}: first token {s.get('first_token_s')} s, prompt {f.get('prompt_eval_count')}, "
              f"cached {f.get('prompt_eval_cached_count')}, read {(f.get('prompt_eval_duration') or 0) / 1e9:.2f} s, "
              f"content {(s.get('content') or '')[:60]!r}")
    elif cmd == "final":
        final()
    else:
        sys.exit(f"unknown command {cmd}")
