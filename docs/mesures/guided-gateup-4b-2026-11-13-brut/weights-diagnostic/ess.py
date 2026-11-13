# Effective sample size of the guided per-token weights, per (layer, type), at G=4 (16 columns summed 4 by 4).
import sys, glob, os, numpy as np
d = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(d, "*.f32"))):
    w = np.fromfile(f, dtype="<f4").reshape(131072, 16).astype(np.float64)
    g = w.reshape(131072, 4, 4).sum(axis=2)          # G = 4
    ess = (g.sum(0) ** 2) / (g ** 2).sum(0) / 131072  # share of tokens that count
    top1 = np.sort(g, axis=0)[-1311:, :].sum(0) / g.sum(0)  # mass carried by the top 1 % tokens
    rows.append((os.path.basename(f), ess.min(), ess.mean(), top1.mean()))
for r in rows[:3] + rows[-3:]:
    print("%s ess_min %.4f ess_mean %.4f top1pct_mass %.3f" % r)
e = np.array([r[2] for r in rows]); t = np.array([r[3] for r in rows]); m = np.array([r[1] for r in rows])
print("files", len(rows), "ESS share mean %.4f min %.4f max %.4f ; worst group %.4f ; top-1%% mass mean %.3f max %.3f" % (e.mean(), e.min(), e.max(), m.min(), t.mean(), t.max()))
