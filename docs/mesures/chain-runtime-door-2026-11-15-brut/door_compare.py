import json, statistics, sys
a = [json.loads(l) for l in open(sys.argv[1])]; b = [json.loads(l) for l in open(sys.argv[2])]
for rid in dict.fromkeys(r["id"] for r in a):
    ra = [r for r in a if r["id"] == rid]; rb = [r for r in b if r["id"] == rid]
    for name, rs in (("chain", ra), ("eager", rb)):
        texts = {r["text"] for r in rs}
        t = [r["ttft_client_s"] for r in rs if r["round"] > 0]
        print(f"{rid} {name}: first content median {statistics.median(t):.3f} s [{min(t):.3f}-{max(t):.3f}] over {len(t)} timed rounds; "
              f"prompt {rs[0]['prompt_eval_count']} tokens; texts identical across rounds: {len(texts) == 1}")
    x, y = ra[0]["text"], rb[0]["text"]
    if x == y:
        print(f"{rid}: the chain's text equals the eager prefill's, {ra[0]['eval_count']} tokens")
    else:
        i = next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), min(len(x), len(y)))
        print(f"{rid}: the texts part at character {i} of {len(x)} and {len(y)}: chain {x[max(0, i - 40):i + 40]!r} / eager {y[max(0, i - 40):i + 40]!r}")
