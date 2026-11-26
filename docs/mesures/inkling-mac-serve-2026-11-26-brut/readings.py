"""Reads every <arm>.json beside this file: one line an arm, then arm A against arm B (tokens, logits).
usage: python3 -I readings.py  (from this directory)"""
import json, os, statistics, struct

here = os.path.dirname(os.path.abspath(__file__))
def load(a):
    p = os.path.join(here, a + ".json")
    return json.load(open(p)) if os.path.exists(p) else None
def gb(r, k):
    return (r["memory"].get(k) or {}).get("used", 0) / 1e9
arms = ["gate-default", "gate-all-bf16", "decode32-r1", "decode32-r2", "decode32-r3",
        "a2000", "b2000", "a2000-peak", "b2000-peak", "b9000", "b9000-peak"]
for a in arms:
    r = load(a)
    if r is None:
        print(f"{a}: not run"); continue
    m = open(os.path.join(here, a + ".machine")).read().strip()
    pk = r.get("peak_probe") or {}
    print(f"{a}: prompt {r['prompt_tokens']} in {r['seconds_prefill']:.2f} s ({r['prompt_tokens'] / r['seconds_prefill']:.1f} tok/s); "
          f"step median {r['step_seconds_median']:.4f} s = {1 / r['step_seconds_median']:.2f} tok/s; GB after load "
          f"{gb(r, 'after_resident'):.2f}, after the prompt {gb(r, 'after_prefill'):.2f}, end {gb(r, 'end'):.2f}; "
          f"peak prompt {pk.get('prefill', 0) / 1e9:.2f} whole {pk.get('whole_run', 0) / 1e9:.2f} GB; "
          f"equal to reference {r.get('all_equal')}; kv positions {r.get('kv_positions_allocated')} | {m}")
d = [load(f"decode32-r{i}") for i in (1, 2, 3)]
if all(d):
    s = [x["step_seconds_median"] for x in d]
    print(f"decode32: median of 3 rounds {statistics.median(s):.4f} s [{min(s):.4f}, {max(s):.4f}] = "
          f"{1 / statistics.median(s):.2f} tok/s; same tokens across rounds {len({tuple(x['tokens']) for x in d}) == 1}")
def floats(a):
    b = open(os.path.join(here, a + ".logits.f32"), "rb").read()
    return struct.unpack(f"<{len(b) // 4}f", b)
for x, y in [("a2000", "b2000"), ("a2000", "a2000-peak"), ("b2000", "b2000-peak"), ("b9000", "b9000-peak")]:
    rx, ry = load(x), load(y)
    if not (rx and ry):
        continue
    fx, fy = floats(x), floats(y)
    same_tok = rx["tokens"] == ry["tokens"]
    first = next((i for i, (u, v) in enumerate(zip(rx["tokens"], ry["tokens"])) if u != v), None)
    gap = max(abs(u - v) for u, v in zip(fx, fy)) if len(fx) == len(fy) else None
    print(f"{x} against {y}: same 32 tokens {same_tok} (first difference at {first}); logits bytes equal {fx == fy}; "
          f"largest logit gap {gap}")
