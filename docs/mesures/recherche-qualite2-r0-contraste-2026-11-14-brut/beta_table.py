# Free: slope (contrast kept) and residual sd of the answer logits of each full-split 4B dump against FP16.
# Same fit as ops/logit_snr.fidelity: centre the four logits of each question, pool, y = beta x + e.
import csv, sys, glob, os, numpy as np
def load(p):
    rows = [r for r in csv.reader(l for l in open(p) if not l.startswith("#"))][1:]
    return {(r[0], r[1]): (np.array([float(x) for x in r[7:11]]), int(r[4]), int(r[6])) for r in rows}
ref = load(sys.argv[1])
print("%-44s %6s %7s %7s %7s %6s" % ("dump", "MMLU", "beta", "sd_res", "beta/sd", "n"))
for p in sys.argv[2:]:
    try: d = load(p)
    except Exception as e: print(os.path.basename(p), "unreadable", e); continue
    keys = [k for k in d if k in ref]
    if len(keys) < 14000: print("%-44s skipped, %d common questions" % (os.path.basename(p), len(keys))); continue
    x = np.array([ref[k][0] for k in keys]); y = np.array([d[k][0] for k in keys])
    x = x - x.mean(1, keepdims=True); y = y - y.mean(1, keepdims=True)
    b = (x * y).sum() / (x * x).sum(); e = y - b * x; sd = np.sqrt((e * e).sum() / (3 * len(keys)))
    print("%-44s %6.2f %7.4f %7.4f %7.3f %6d" % (os.path.basename(p)[:44], 100 * np.mean([d[k][2] for k in keys]), b, sd, b / sd, len(keys)))
