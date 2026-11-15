import json, sys, time, urllib.request
host, model, prompts, long_file, ntok, rounds, out = sys.argv[1:8]
def first(path):
    for l in open(path):
        r = json.loads(l)
        if "_comment" not in r:
            return r["id"], r["prompt"]
reqs = [first(long_file), first(prompts)]
w = open(out, "w")
for rnd in range(int(rounds) + 1):
    for rid, text in reqs:
        body = {"model": model, "messages": [{"role": "user", "content": text}], "stream": True, "think": False,
                "options": {"temperature": 0, "num_predict": int(ntok)}, "keep_alive": -1}
        req = urllib.request.Request(f"http://{host}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time(); t_first = None; content = ""; last = {}
        with urllib.request.urlopen(req, timeout=600) as r:
            for line in r:
                v = json.loads(line)
                c = v.get("message", {}).get("content", "")
                if c and t_first is None:
                    t_first = time.time() - t0
                content += c
                if v.get("done"):
                    last = v
        rec = {"round": rnd, "id": rid, "ttft_client_s": t_first, "wall_s": time.time() - t0, "text": content,
               "prompt_eval_count": last.get("prompt_eval_count"), "prompt_eval_cached_count": last.get("prompt_eval_cached_count"),
               "prompt_eval_s": (last.get("prompt_eval_duration") or 0) / 1e9, "eval_count": last.get("eval_count"),
               "eval_s": (last.get("eval_duration") or 0) / 1e9, "done_reason": last.get("done_reason")}
        w.write(json.dumps(rec) + "\n"); w.flush()
        print(f"  round {rnd} {rid}: first content {t_first:.3f} s, prompt {rec['prompt_eval_count']} ({rec['prompt_eval_cached_count']} cached) "
              f"in {rec['prompt_eval_s']:.3f} s, {rec['eval_count']} tokens in {rec['eval_s']:.3f} s, {rec['done_reason']}", flush=True)
