# Where the guided weight mass sits inside a window (64 windows of 2,048 tokens), G collapsed to 1.
import sys, glob, os, numpy as np
d = sys.argv[1]
acc0 = []; acc16 = []; clip = []
for f in sorted(glob.glob(os.path.join(d, "*.f32"))):
    w = np.fromfile(f, dtype="<f4").reshape(64, 2048, 16).astype(np.float64).sum(2)
    tot = w.sum()
    acc0.append(w[:, 0].sum() / tot); acc16.append(w[:, :16].sum() / tot)
    v = w.reshape(-1); v1 = v.copy(); v1.reshape(64, 2048)[:, 0] = 0   # drop position 0
    ess = lambda x: x.sum() ** 2 / (x ** 2).sum() / x.size
    c = np.minimum(v, 10 * v.mean())                                   # clip at 10 x the mean
    clip.append((ess(v), ess(v1), ess(c)))
a0 = np.array(acc0); a16 = np.array(acc16); c = np.array(clip)
print("mass at position 0: mean %.3f min %.3f max %.3f" % (a0.mean(), a0.min(), a0.max()))
print("mass in positions 0..15: mean %.3f max %.3f" % (a16.mean(), a16.max()))
print("ESS share (G=1): as is %.4f ; without position 0 %.4f ; clipped at 10x mean %.4f" % tuple(c.mean(0)))
for i in (0, 2, 20, 40, 70): print(i, "pos0 %.3f first16 %.3f ess %.4f ess_no0 %.4f" % (a0[i], a16[i], c[i, 0], c[i, 1]))
