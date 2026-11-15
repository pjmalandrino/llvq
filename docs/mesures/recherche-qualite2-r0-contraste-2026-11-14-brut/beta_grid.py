# Free computation on the FP16 4B full-split dump: what the Gaussian replay model predicts for MMLU
# at other (beta, sd) than the sealed 4B's fitted pair. Model of ops/logit_n1_replay.py: pick = argmax of
# beta * f16_logit + N(0, sd^2) on each of the four options. Exact per question by Monte Carlo, 4000 draws.
import csv, sys, numpy as np
rows = [r for r in csv.reader(l for l in open(sys.argv[1]) if not l.startswith("#"))][1:]
L = np.array([[float(x) for x in r[7:11]] for r in rows]); ans = np.array([int(r[4]) for r in rows])
print("questions", len(L), "FP16 measured %.2f" % (100 * (L.argmax(1) == ans).mean()))
rng = np.random.default_rng(0)
def acc(beta, sd, draws=600):
    hit = 0.0
    for _ in range(draws):
        hit += ((beta * L + sd * rng.standard_normal(L.shape)).argmax(1) == ans).mean()
    return 100 * hit / draws
b0, s0 = 0.5817, 1.4205
print("sealed 4B fit  beta %.4f sd %.4f -> %.2f (journal 64.65)" % (b0, s0, acc(b0, s0)))
for b, s, label in [(1.0, s0, "full contrast, same noise"), (0.80, s0, "contrast 0.80, same noise"), (0.70, s0, "contrast 0.70, same noise"),
                    (b0, s0 * 0.8, "same contrast, noise -20 %"), (b0, s0 * 0.5, "same contrast, noise halved"),
                    (0.9604, 0.8741 * np.sqrt(4/3), "AWQ 4B fit (journal pred 69.02)"), (0.4809, 1.5054 * np.sqrt(4/3), "bare Tetra fit (journal pred 59.09)")]:
    print("%-38s beta %.4f sd %.4f -> %.2f" % (label, b, s, acc(b, s)))
