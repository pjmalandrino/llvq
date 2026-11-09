#!/usr/bin/env python3
# Standard library only: this file runs inside the CUDA image beside `llvq serve`.
"""The batched decode in the client's units: B concurrent requests to `llvq serve`.

B threads post `/api/generate` at once, raw prompt "The capital of France is", greedy, 256 tokens each
(`num_predict`, the tiny vocabulary of a stop never comes on this prompt, and a short answer is reported, not
hidden). Per round and B:

* aggregate tok/s = the tokens every request generated / the wall time from the first post to the last answer;
* tok/s a request = its `eval_count / eval_duration` (the server's own decode clock, prefill excluded);
* time to the first token is not measured (non-streamed answers).

One warm-up round at each B, then `--rounds` rounds; medians and ranges per B. Every answer's text is compared with
the B = 1 answer of the same round set: under the batched GEMM a sequence does not owe batch 1's text, so the count
is reported, not gated.

Exit codes: 0 measured; 2 refused; 4 a request failed or came back short.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request

PROMPT = "The capital of France is"


def post(host: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(f"http://{host}/api/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def one_round(host: str, model: str, b: int, tokens: int) -> tuple[float, list[dict]]:
    body = {"model": model, "prompt": PROMPT, "raw": True, "stream": False,
            "options": {"temperature": 0, "num_predict": tokens}}
    out: list[dict | Exception] = [None] * b  # type: ignore[list-item]
    barrier = threading.Barrier(b)

    def worker(i: int) -> None:
        try:
            barrier.wait()
            out[i] = post(host, body, 1800.0)
        except Exception as e:  # noqa: BLE001 - reported below
            out[i] = e

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(b)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    bad = [o for o in out if isinstance(o, Exception)]
    if bad:
        raise RuntimeError(f"B={b}: {len(bad)} requests failed, first: {bad[0]}")
    return wall, out  # type: ignore[return-value]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1:11435")
    ap.add_argument("--model", required=True)
    ap.add_argument("--batches", default="1,8,16,32,64,128")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    try:
        batches = [int(x) for x in a.batches.split(",")]
        if any(b < 1 or b > 256 for b in batches):
            raise ValueError("B in 1..=256")
    except ValueError as e:
        print(f"refused: --batches {a.batches!r}: {e}", file=sys.stderr)
        return 2
    rates: dict[int, list[float]] = {b: [] for b in batches}
    per: dict[int, list[float]] = {b: [] for b in batches}
    texts: dict[int, list[str]] = {}
    try:
        for r in range(a.rounds + 1):
            for b in batches:
                wall, outs = one_round(a.host, a.model, b, a.tokens)
                counts = [o["eval_count"] for o in outs]
                if min(counts) < a.tokens:
                    print(f"short answers at B={b}: {sorted(counts)[:4]}", file=sys.stderr)
                    return 4
                agg = sum(counts) / wall
                dec = [o["eval_count"] / (o["eval_duration"] / 1e9) for o in outs]
                if r == 0:
                    texts[b] = [o["response"] for o in outs]
                    print(f"  B={b:>3} warm-up {agg:9.1f} tok/s aggregate ({wall:.2f} s)")
                    continue
                rates[b].append(agg)
                per[b].append(statistics.median(dec))
                print(f"  round {r} B={b:>3} {agg:9.1f} tok/s aggregate, {statistics.median(dec):7.1f} a request ({wall:.2f} s)")
    except RuntimeError as e:
        print(f"VIOLATION: {e}", file=sys.stderr)
        return 4
    ref = texts[batches[0]][0]
    out = {"prompt": PROMPT, "tokens": a.tokens, "rounds": a.rounds, "per_batch": {}}
    print(f"\nllvq serve, {a.model}, {a.tokens} tokens a request, raw prompt, {a.rounds} rounds")
    print(f"{'B':>4} {'tok/s aggregate':>16} {'range':>22} {'tok/s a request':>16}  answers equal to B={batches[0]}")
    for b in batches:
        xs = rates[b]
        same = sum(t == ref for t in texts[b])
        out["per_batch"][str(b)] = {"aggregate": xs, "median": statistics.median(xs), "per_request": per[b],
                                    "answers_equal": same}
        print(f"{b:>4} {statistics.median(xs):>16.1f} [{min(xs):>9.1f}-{max(xs):>9.1f}] "
              f"{statistics.median(per[b]):>16.1f}  {same} of {b}")
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
