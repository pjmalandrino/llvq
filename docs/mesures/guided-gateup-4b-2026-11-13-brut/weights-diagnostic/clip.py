# Effective sample share of the guided weights at G = 4 after clipping each group's weights at c x its mean.
import sys, glob, os, numpy as np
d = sys.argv[1]; cs = [3, 5, 10, 20, 50]
out = {c: [] for c in cs}; kept = {c: [] for c in cs}
for f in sorted(glob.glob(os.path.join(d, "*.f32"))):
    g = np.fromfile(f, dtype="<f4").reshape(131072, 4, 4).astype(np.float64).sum(2)
    for c in cs:
        w = np.minimum(g, c * g.mean(0))
        out[c].append((w.sum(0) ** 2) / (w ** 2).sum(0) / 131072)
        kept[c].append(w.sum(0) / g.sum(0))
for c in cs:
    e = np.array(out[c]); k = np.array(kept[c])
    print("clip %2dx: ESS share mean %.3f, worst matrix mean %.3f, worst group %.3f (%d tokens) ; mass kept mean %.2f min %.2f" %
          (c, e.mean(), e.mean(1).min(), e.min(), e.min() * 131072, k.mean(), k.min()))
