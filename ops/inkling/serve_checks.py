"""The requests of ops/jobs/inkling-serve-card.sh and the reading of their answers. Standard library only.

The job script copies this file into the job's command, so the image does not need to carry it.
Correctness checks decide the job's exit code. Reading checks say PASS or FAIL and change nothing.

  serve_checks.py req <name> [answer of the previous step]   prints the request body
  serve_checks.py sse <t0> <raw out> <json out>               reads an SSE stream on stdin, timestamps each line
  serve_checks.py check <name> <files...>                     one line PASS or FAIL into $O/summary.txt
  serve_checks.py final                                       exit 1 when a correctness check failed
"""
import json
import os
import sys
import time

MODEL = os.environ.get("MODEL", "inkling2-mini:local")
O = os.environ.get("O", ".")

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "The city name, for example Paris."}},
            "required": ["city"],
        },
    },
}
SYS_TOOLS = "You are a helpful assistant. When a tool can answer the question, call it."
USER_TOOLS = "What is the weather in Paris right now?"
RESULT = json.dumps({"city": "Paris", "temperature_c": 18, "sky": "clear"})

# The long prompt: a long system prompt in numbered sections and many tool definitions, built the same way on every
# run. About 9,000 tokens once rendered; the server returns the exact count.
TOPICS = [
    ("logging", "every request is written to the audit log with its time, its author and its outcome"),
    ("backups", "the nightly backup is checked by a restore on a spare machine once a week"),
    ("access", "an account that has not been used for ninety days is disabled until its owner asks again"),
    ("secrets", "a key is never written in a file that is tracked, and a leaked key is revoked within the hour"),
    ("reviews", "every change to the build is read by a second person before it is merged"),
    ("releases", "a release names its commit, its build machine and the checksum of every file it ships"),
    ("incidents", "an incident gets a short written account within two days, with what failed and what changed"),
    ("costs", "a paid job states its price before it starts and its running total after it ends"),
    ("data", "personal data stays in the region where it was collected and is deleted on request"),
    ("tests", "a test that skips for want of an input fails and names the input it was missing"),
]
SECTIONS = 38
N_TOOLS = 14
LONG_USER = "Do not call any tool. In one short sentence: what does section 7 of the policy ask for?"
LONG_FOLLOW = "Thank you. In one word: is that section about security?"


def long_system():
    out = ["You are the operations assistant of a small research team. Read the policy below before you answer."]
    for k in range(SECTIONS):
        name, rule = TOPICS[k % len(TOPICS)]
        out.append(
            f"Section {k + 1}, {name}, part {k // len(TOPICS) + 1}. The rule is that {rule}. "
            f"This section applies to every machine the team rents, to every laptop that holds a copy of the "
            f"repository, and to every job that runs on a card. When a case is not covered, the operator decides, "
            f"and the decision is written down with its date. A rule that changes is replaced, and the old rule is "
            f"kept in the history file with the date of the change. Questions about section {k + 1} go to the "
            f"operator first, and to the rest of the team only when the operator is away for more than a day."
        )
    return "\n\n".join(out)


def long_tools():
    tools = [WEATHER]
    for i in range(N_TOOLS):
        name, rule = TOPICS[i % len(TOPICS)]
        tools.append({
            "type": "function",
            "function": {
                "name": f"{name}_lookup_{i:02d}",
                "description": f"Look up the records about {name} for one machine or one job. The policy says that "
                             f"{rule}. Returns a list of records, newest first.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "machine": {"type": "string", "description": "The machine name, as the inventory writes it."},
                        "job": {"type": "string", "description": "The job identifier, when the question is about a job."},
                        "since": {"type": "string", "description": "The first date to include, in ISO 8601."},
                        "limit": {"type": "integer", "description": "How many records at most, default 20."},
                    },
                    "required": ["machine"],
                },
            },
        })
    return tools


def oai_stream(body):
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    return body


def req(name, prev=None):
    if name == "b":
        return {"model": MODEL, "max_tokens": 256,
                "messages": [{"role": "user", "content": "What is the capital of France? Answer with one word."}]}
    if name == "c":
        return oai_stream({"model": MODEL, "max_tokens": 48,
                           "messages": [{"role": "user", "content": "Count from one to five in words, separated by commas."}]})
    if name in ("d1", "n1"):
        msgs = [{"role": "system", "content": SYS_TOOLS}, {"role": "user", "content": USER_TOOLS}]
        if name == "d1":
            return {"model": MODEL, "messages": msgs, "tools": [WEATHER], "max_tokens": 256}
        return {"model": MODEL, "messages": msgs, "tools": [WEATHER], "stream": False, "keep_alive": -1,
                "options": {"num_predict": 256}}
    if name == "d2":
        r = json.load(open(prev))
        m = r["choices"][0]["message"]
        calls = m.get("tool_calls") or []
        if not calls:
            sys.exit("d1 carried no tool call: no follow-up to send")
        msgs = req("d1")["messages"] + [{"role": "assistant", "content": m.get("content"), "tool_calls": calls}]
        msgs += [{"role": "tool", "tool_call_id": c["id"], "content": RESULT} for c in calls]
        return oai_stream({"model": MODEL, "messages": msgs, "tools": [WEATHER], "max_tokens": 256})
    if name == "n2":
        r = json.load(open(prev))
        m = r["message"]
        calls = m.get("tool_calls") or []
        if not calls:
            sys.exit("n1 carried no tool call: no follow-up to send")
        msgs = req("n1")["messages"] + [{"role": "assistant", "content": m.get("content", ""), "tool_calls": calls}]
        msgs += [{"role": "tool", "content": RESULT, "tool_name": c["function"]["name"]} for c in calls]
        return {"model": MODEL, "messages": msgs, "tools": [WEATHER], "stream": False, "keep_alive": -1,
                "options": {"num_predict": 256}}
    if name == "e1":
        return oai_stream({"model": MODEL, "max_tokens": 64, "tools": long_tools(),
                           "messages": [{"role": "system", "content": long_system()},
                                        {"role": "user", "content": LONG_USER}]})
    if name == "e3":
        s = json.load(open(prev))
        msgs = req("e1")["messages"]
        a = {"role": "assistant", "content": s["content"]}
        if s["tool_calls"]:
            a["tool_calls"] = [{"function": {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}}
                               for c in s["tool_calls"]]
        msgs.append(a)
        msgs += [{"role": "tool", "content": "[]", "tool_name": c["function"]["name"]} for c in s["tool_calls"]]
        msgs.append({"role": "user", "content": LONG_FOLLOW})
        return {"model": MODEL, "messages": msgs, "tools": long_tools(), "stream": False, "keep_alive": -1,
                "options": {"num_predict": 256}}
    sys.exit(f"unknown request {name}")


def sse(t0, raw_path, out_path):
    """Read an OpenAI stream: every line kept raw with its arrival time, the deltas joined."""
    t0 = float(t0)
    raw = open(raw_path, "w")
    s = {"lines": 0, "chunks": 0, "done": False, "first_chunk_s": None, "first_token_s": None, "last_s": None,
         "content": "", "reasoning": "", "tool_calls": [], "finish_reason": None, "usage": None, "errors": []}
    for line in iter(sys.stdin.buffer.readline, b""):
        t = time.time() - t0
        text = line.decode("utf-8", "replace").rstrip("\n")
        raw.write(f"{t:.4f}\t{text}\n")
        s["lines"] += 1
        if not text.startswith("data: "):
            continue
        payload = text[6:]
        s["last_s"] = t
        if payload == "[DONE]":
            s["done"] = True
            continue
        try:
            c = json.loads(payload)
        except ValueError:
            s["errors"].append(f"not JSON: {payload[:200]}")
            continue
        if "error" in c:
            s["errors"].append(c["error"])
            continue
        s["chunks"] += 1
        if s["first_chunk_s"] is None:
            s["first_chunk_s"] = t
        if c.get("usage"):
            s["usage"] = c["usage"]
        for ch in c.get("choices", []):
            d = ch.get("delta", {})
            if d.get("content") or d.get("reasoning") or d.get("tool_calls"):
                if s["first_token_s"] is None:
                    s["first_token_s"] = t
            s["content"] += d.get("content") or ""
            s["reasoning"] += d.get("reasoning") or ""
            s["tool_calls"] += d.get("tool_calls") or []
            if ch.get("finish_reason"):
                s["finish_reason"] = ch["finish_reason"]
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


def seconds(path):
    """curl's -w '%{http_code} %{time_starttransfer} %{time_total}'."""
    try:
        code, _, total = open(path).read().split()
        return int(code), float(total)
    except (OSError, ValueError):
        return 0, None


def args_object(c):
    a = c.get("function", {}).get("arguments")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except ValueError:
            return None
    return a if isinstance(a, dict) else None


def check(name, *files):
    if name == "a":
        r = read(files[0])
        ids = [m.get("id") for m in r.get("data", [])]
        record("a-models", "correctness", MODEL in ids, f"GET /v1/models lists {ids}")
    elif name == "load":
        r, (code, t) = read(files[0]), seconds(files[1])
        record("load", "correctness", code == 200 and r.get("done_reason") == "load",
               f"HTTP {code}, done_reason {r.get('done_reason')!r}, load_duration {r.get('load_duration', 0) / 1e9:.1f} s", t)
    elif name == "b":
        r, (code, t) = read(files[0]), seconds(files[1])
        ch = (r.get("choices") or [{}])[0]
        content = (ch.get("message") or {}).get("content") or ""
        ok = code == 200 and ch.get("finish_reason") == "stop" and "paris" in content.lower()
        record("b-chat", "correctness", ok, f"HTTP {code}, finish {ch.get('finish_reason')!r}, content {content[:80]!r}, "
               f"usage {r.get('usage')}", t)
    elif name in ("c", "d2", "e1", "e2"):
        s = read(files[0])
        code, t = seconds(files[1]) if len(files) > 1 else (0, None)
        if "_unreadable" in s:
            record(f"{name}-stream", "correctness", False, f"no stream read: {s['_unreadable']}")
            return
        answered = s["done"] and not s["errors"] and s["finish_reason"] in ("stop", "length", "tool_calls") \
            and (s["content"] or s["tool_calls"] or s["reasoning"])
        usage = s["usage"] or {}
        base = (f"[DONE] {s['done']}, chunks {s['chunks']}, finish {s['finish_reason']!r}, errors {s['errors'][:2]}, "
                f"prompt_tokens {usage.get('prompt_tokens')}, completion_tokens {usage.get('completion_tokens')}, "
                f"first chunk {s['first_chunk_s']} s, first token {s['first_token_s']} s, content {s['content'][:80]!r}")
        if name == "c":
            record("c-stream", "correctness", answered and usage.get("prompt_tokens", 0) > 0 and bool(s["content"]),
                   base, s["first_token_s"])
        elif name == "d2":
            ok = answered and s["finish_reason"] == "stop" and bool(s["content"].strip()) and not s["tool_calls"]
            record("d2-tool-followup", "correctness", ok, base, s["first_token_s"])
            record("d2-uses-result", "reading", "18" in s["content"], "the answer names the 18 degrees of the result")
        else:
            record(f"{name}-long", "correctness", bool(answered) and usage.get("prompt_tokens", 0) > 0, base,
                   s["first_token_s"])
            p = usage.get("prompt_tokens") or 0
            record(f"{name}-prompt-size", "reading", 8000 <= p <= 10000, f"prompt_tokens {p}, about 9,000 asked")
            if name == "e2":
                s1 = read(files[2])
                t1, t2 = s1.get("first_token_s"), s["first_token_s"]
                ratio = (t2 / t1) if t1 and t2 else None
                record("e2-repeat-reuse", "reading", ratio is not None and ratio < 0.5,
                       f"first token of the repeat over the first: {ratio}; inkling.rs reads a repeated prompt of one "
                       f"segment again in full, so FAIL here is what the code predicts, not a fault")
    elif name == "d1":
        r, (code, t) = read(files[0]), seconds(files[1])
        ch = (r.get("choices") or [{}])[0]
        calls = (ch.get("message") or {}).get("tool_calls") or []
        good = [c for c in calls if c.get("function", {}).get("name") == "get_weather" and args_object(c) is not None
                and "city" in args_object(c)]
        ok = code == 200 and ch.get("finish_reason") == "tool_calls" and len(good) >= 1 and bool(calls[0].get("id"))
        record("d1-tool-call", "correctness", ok, f"HTTP {code}, finish {ch.get('finish_reason')!r}, calls {calls}, "
               f"usage {r.get('usage')}", t)
        city = (args_object(good[0]) or {}).get("city", "") if good else ""
        record("d1-city", "reading", "paris" in str(city).lower(), f"city {city!r}")
    elif name == "n1":
        r, (code, t) = read(files[0]), seconds(files[1])
        calls = (r.get("message") or {}).get("tool_calls") or []
        record("n1-native-tool-call", "correctness", code == 200 and len(calls) >= 1,
               f"HTTP {code}, calls {calls}, prompt_eval_count {r.get('prompt_eval_count')}, "
               f"cached {r.get('prompt_eval_cached_count')}, eval_count {r.get('eval_count')}", t)
    elif name in ("n2", "e3"):
        r, (code, t) = read(files[0]), seconds(files[1])
        prev = read(files[2])
        content = (r.get("message") or {}).get("content") or ""
        n, c = r.get("prompt_eval_count") or 0, r.get("prompt_eval_cached_count") or 0
        if name == "n2":
            first = prev.get("prompt_eval_count") or 0
        else:
            first = (prev.get("usage") or {}).get("prompt_tokens") or 0
        pe = (r.get("prompt_eval_duration") or 0) / 1e9
        record(f"{name}-followup", "correctness", code == 200 and bool(content.strip()),
               f"HTTP {code}, content {content[:80]!r}, done_reason {r.get('done_reason')!r}", t)
        record(f"{name}-reuse-prompt", "reading", c >= first > 0,
               f"cached {c} of {n}; the first request's prompt was {first} tokens; read {n - c} tokens in {pe:.2f} s")
        record(f"{name}-reuse-answer", "reading", c > first > 0,
               f"the answer the cache held was reused too when cached > {first}")
    elif name in ("ps-loaded", "ps-end"):
        r = read(files[0])
        log = open(files[1], errors="replace").read() if os.path.exists(files[1]) else ""
        m = [x for x in r.get("models", []) if x.get("name") == MODEL]
        st = (m[0].get("llvq") or {}) if m else {}
        ok = st.get("state") == "ready" and st.get("backend") == "cuda" and "panicked" not in log
        record(name, "correctness", ok, f"state {st.get('state')!r}, backend {st.get('backend')!r}, "
               f"error {st.get('error')!r}, size_vram {m[0].get('size_vram') if m else None}, "
               f"'panicked' in the server log: {'panicked' in log}")
    elif name == "gpu":
        for f in files:
            line = f"INFO {os.path.basename(f)}: {open(f).read().strip()}" if os.path.exists(f) else f"INFO {f}: absent"
            print(line)
            with open(f"{O}/summary.txt", "a") as out:
                out.write(line + "\n")
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
        print(json.dumps(req(rest[0], rest[1] if len(rest) > 1 else None)))
    elif cmd == "sse":
        sse(*rest)
    elif cmd == "check":
        check(*rest)
    elif cmd == "final":
        final()
    else:
        sys.exit(f"unknown command {cmd}")
