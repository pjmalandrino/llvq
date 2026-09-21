# Preregistration — the full sensitivity map of Qwen3-0.6B

**Written, committed and TIMESTAMPED on 2026-09-15, BEFORE the full run.**
Operator go given 2026-09-15.

🚨 **Not edited again**: the stamp attests these bytes at this date. A fact it
gets wrong is written beside it, in a `-ECARTS.md`.

**What is measured**: `errmap` at `286605d`, 196 matrices, 392 evaluations.
**Cost: $0, about 1 h 45 of Mac.**

## 1. What the run produces

For each of the 7 projections × 28 blocks, the gradient and curvature of the
model's NLL with respect to a scale error on that matrix, by central differences
at ε = 0.01. From them, a quadratic surrogate over all 196 scales.

The surrogate is exact for one matrix at a time by construction. Cross terms are
not measured. So the run's one real test is the held-out combination of §3.

## 2. Signed predictions

1. **A majority of directions come back concave or flat** (curvature ≤ 0). The
   two-block pilot read 9 of 14. Named against me: a convex majority means the
   quantized run sits near a minimum of its own loss, which would make the trust
   region unnecessary and the map far easier than this document assumes.
2. **`q_proj` and `k_proj` are scale invariant**, gradient and curvature both
   exactly zero, on all 28 blocks. The mechanism is Qwen3's per-head RMS norm on
   q and k. A single non-zero gradient on either refutes the explanation.
3. **The held-out combination agrees in sign** with the prediction, and its
   error is **under 50 % of the predicted move**. The two-block pilot read 20 %
   and conservative.
4. **The pooled single scale the map implies differs from 1.02**, the sweep's
   re-encoded minimum. These measure different things — post-hoc scaling against
   re-encoding — and the gap is the compensation's reaction, not an error in
   either. Named against me: agreement to better than 0.005 would mean the
   sequential loop does not react to the centroids at all, which contradicts
   how it is written.

## 3. The held-out test

The 8 matrices the map ranks highest are moved together to their trust-region
optima, and the NLL is measured. Prediction against measurement is reported as
an absolute error, as a fraction of the predicted move, and as a sign agreement.
The model is then restored and its NLL re-checked against the baseline; a
mismatch there voids the run.

## 4. What this licenses

Nothing about the served encoder, and no MMLU claim. A map is a measurement of
where error costs, not a change to anything. Acting on it needs its own prereg,
and its first honest test would be a re-encoding arm, since post-hoc scaling and
re-encoding are the two different things §2.4 is about.
