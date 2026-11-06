# ARITHMETIC HELPER, not a code sketch and not repository code. Run once with `python3 -I`
# (a tiny script, as the ground rules allow) to produce the paired interval quoted in
# wave5-deep/prefill.md section 8. It reads no file: the 24 NLLs are copied by hand from
# docs/mesures/ppl-scelles-2026-10-04-brut/ppl-4b.txt and docs/mesures/gqa-ppl-2026-11-01-brut/gqa-ppl.txt.
# Paired prefill-path NLL: dense reconstruction (ppl-scelles-2026-10-04, 4B sealed, f16, L40S)
# against the fused kernel's published arm (gqa-ppl-2026-11-01, cat+expanded, prefill path),
# same 12 windows of 4096 (fingerprint 3f1baca9033bf251). Numbers copied from the raw files.
# CROSS-JOB: the two arms ran a month apart, in two jobs on two images (97a2b62a and 5e5b1fd5).
# METHODE section 2: "A cross-job reading is reported and not published". The gqa-ppl journal
# itself reads the two ppl figures "side by side, not divided". So the interval below is
# *computed*, reported, NOT published, and it INDICATES; it bounds nothing. It also mixes the
# projection arithmetic with the head arithmetic (dense head against the fused emb_q4 head).
# Only the fresh-window control of prereg-prefill-2 step 5 may feed a T8 decision.
import math, statistics as st
dense = [2.528929144, 2.902055432, 2.263911306, 2.032169328, 2.639317193, 2.601170921,
         2.603101152, 2.596139061, 2.733776423, 2.241730578, 2.668958429, 2.577238820]
fused = [2.529226763, 2.902096688, 2.263824977, 2.032402797, 2.639750744, 2.599391646,
         2.603219198, 2.596115690, 2.733835089, 2.241309905, 2.668974645, 2.577210203]
d = [a - b for a, b in zip(dense, fused)]
n = len(d); m = st.mean(d); s = st.stdev(d); t = 2.201  # t(0.975, 11)
lo, hi = m - t * s / math.sqrt(n), m + t * s / math.sqrt(n)
print("per-window dNLL (dense - fused):", ["%+.6f" % x for x in d])
print("mean %+.6f nats, sd %.6f, CI95 [%+.6f, %+.6f]" % (m, s, lo, hi))
print("ppl ratio dense/fused - 1: %+.4f %% [%+.4f %%, %+.4f %%]" % (
    100 * (math.exp(m) - 1), 100 * (math.exp(lo) - 1), 100 * (math.exp(hi) - 1)))
print("ppl dense %.4f fused %.4f" % (math.exp(st.mean(dense)), math.exp(st.mean(fused))))
print("windows dense worse:", sum(1 for x in d if x > 0), "of", n)
print("max |dNLL| %.6f nats = %.4f %% of ppl" % (max(abs(x) for x in d), 100 * (math.exp(max(abs(x) for x in d)) - 1)))
