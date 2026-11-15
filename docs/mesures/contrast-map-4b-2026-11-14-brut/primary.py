# The prereg's primary numbers, read from the job's TSV files. Light, no model.
import csv, sys, statistics as st, math
R = sys.argv[1]; H = sys.argv[2]
def tsv(p):
    rows = [l for l in open(p) if not l.startswith("#")]
    return list(csv.DictReader(rows, delimiter="\t"))
m = tsv(R + "/matrices.tsv"); print("matrices.tsv columns:", list(m[0].keys()))
med = [float(r["row_med_fp16"]) for r in m] if "row_med_fp16" in m[0] else None
key_a = [k for k in m[0] if k.startswith("a_") and "fp16" in k][0]; key_rel = [k for k in m[0] if k.startswith("rel") and "fp16" in k][0]
key_med = [k for k in m[0] if "row" in k and ("med" in k or "p50" in k) and "fp16" in k][0]
a = [float(r[key_a]) for r in m]; rel = [float(r[key_rel]) for r in m]; rm = [float(r[key_med]) for r in m]
print("a_med (median over 252 matrices of the row-median gain, FP16 x): %.5f ; of the pooled gain: %.5f ; matrices with row-median < 0.98: %d of %d ; < 0.99: %d" % (st.median(rm), st.median(a), sum(x < 0.98 for x in rm), len(rm), sum(x < 0.99 for x in rm)))
x = rel; y = [1 - v for v in a]; mx, my = st.mean(x), st.mean(y)
c = sum((u - mx) * (v - my) for u, v in zip(x, y)) / sum((u - mx) ** 2 for u in x); c0 = sum(u * v for u, v in zip(x, y)) / sum(u * u for u in x)
print("c, OLS slope of (1 - a) on rel error over 252 matrices: %.3f ; through the origin: %.3f" % (c, c0))
for t in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
    s = [(1 - float(r[key_a])) / float(r[key_rel]) for r in m if r["proj"] == t] if "proj" in m[0] else []
    if s: print("  %-10s median of (1-a)/rel over 36 layers: %.2f" % (t, st.median(s)))
at = tsv(R + "/attention.tsv"); d = [float(r["entropy_sealed"]) - float(r["entropy_fp16"]) for r in at]
print("dH mean over %d heads: %+.4f nats ; share with higher entropy in sealed: %.3f" % (len(d), st.mean(d), sum(v > 0 for v in d) / len(d)))
big = sorted(((float(r["entropy_sealed"]) - float(r["entropy_fp16"])), r["layer"], r["head"], r["entropy_fp16"], r["top_fp16"], r["top_sealed"]) for r in at)
print("largest entropy rises:", [(l, h, round(v, 2), ef[:5], tf[:5], ts[:5]) for v, l, h, ef, tf, ts in big[-6:]])
print("largest entropy falls:", [(l, h, round(v, 2)) for v, l, h, *_ in big[:4]])
li = tsv(R + "/linear.tsv"); print([ (r["depth"], r["band"], r["e_lin"][:6], r["diag"][:6], r["ridge_over_lambda_min"][:9]) for r in li if r["depth"] == "36"])
# lens: FP16 contrast sd by depth, and SNR in comparable units
ll = tsv(H + "/lens_logits.tsv")
by = {}
for r in ll:
    f = [float(r["fp16_" + k]) for k in "abcd"]; s = [float(r["sealed_" + k]) for k in "abcd"]
    fm = sum(f) / 4; sm = sum(s) / 4
    by.setdefault(int(r["depth"]), []).append(([v - fm for v in f], [v - sm for v in s]))
print("depth  sd(FP16 centred lens)  slope  noise  slope*sd/noise")
for dep in (0, 8, 16, 24, 25, 26, 28, 30, 32, 34, 35, 36):
    xs = [v for f, s in by[dep] for v in f]; ys = [v for f, s in by[dep] for v in s]
    b = sum(u * v for u, v in zip(xs, ys)) / sum(u * u for u in xs); n = len(by[dep])
    noise = math.sqrt(sum((v - b * u) ** 2 for u, v in zip(xs, ys)) / (3 * n)); sd = math.sqrt(sum(u * u for u in xs) / (3 * n))
    print("%5d  %8.3f  %.4f  %.4f  %.3f" % (dep, sd, b, noise, b * sd / noise))
