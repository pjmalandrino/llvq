# Summary of alternated decode rounds: python3 -I summarize.py <dir> <prefix> <n>
import json, sys, hashlib, statistics
d, p, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
def load(l):
    r = json.load(open(f"{d}/{l}.json"))
    h = hashlib.sha256(open(f"{d}/{l}.logits.f32", "rb").read()).hexdigest()
    return r, h
gains = []
for i in range(1, n + 1):
    a, ha = load(f"{p}-ref{i}"); b, hb = load(f"{p}-var{i}")
    ma, mb = a["step_seconds_median"], b["step_seconds_median"]
    g = ma / mb - 1
    gains.append(g)
    print(f"round {i}: ref {1e3*ma:.2f} ms ({1/ma:.2f} tok/s), var {1e3*mb:.2f} ms ({1/mb:.2f} tok/s), "
          f"gain {100*g:+.2f} %, {1e3*(ma-mb):+.2f} ms; tokens equal {a['tokens']==b['tokens']}, "
          f"sha equal {ha==hb} ({ha[:8]}), after load {a['memory']['after_resident']['used']/1e9:.3f} / {b['memory']['after_resident']['used']/1e9:.3f} GB")
print(f"median gain {100*statistics.median(gains):+.2f} % [{100*min(gains):+.2f}, {100*max(gains):+.2f}]; "
      f"rounds faster {sum(g>0 for g in gains)} of {n}")
