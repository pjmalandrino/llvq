"""Route P and 4E arithmetic for the persistent lead (wave5-deep/persistent.md).

Pure arithmetic on stated inputs, run with python3 -I. Every input is labelled:
M = measured (journal named), C = computed, E = estimated. Every output carries
the weakest label of its inputs, so every tok/s here is *estimated*.
"""

# ---- bytes (C, from row formulas; persistent.md section 8 item 12 and section 4) ----
W_T_SERVED = 0.7487  # GB, the 168 served Tetra records with binary16 tails (C)
W_Q_SERVED = 0.4094  # GB, the 84 int4 g128 records (C)
W_H_SERVED = 0.2188  # GB, the tied int4 g64 head (C, gqa-ab-256.txt prints 218.8 MB)

# Bare 252-Tetra file, one bench pass (C from row formulas, checked against
# banc-252 sweep-64: "0.98 GB read, 2.150 b/weight").
LAYERS = 36
rows_2560 = 4096 + 1024 + 1024 + 9728 + 9728  # q k v gate up, d_in 2560: 106 blocks, tail 16
rows_4096 = 2560                              # o, d_in 4096: 170 blocks, tail 16
rows_9728 = 2560                              # down, d_in 9728: 405 blocks, tail 8
stride_b = {2560: 640, 4096: 1024, 9728: 2432}  # bytes a row, 6 nblocks rounded to 8
words = LAYERS * (rows_2560 * stride_b[2560] + rows_4096 * stride_b[4096] + rows_9728 * stride_b[9728])
tail_w = LAYERS * (rows_2560 * 16 + rows_4096 * 16 + rows_9728 * 8)
rscale = LAYERS * (rows_2560 + rows_4096 + rows_9728) * 4
W_WORDS = words / 1e9
W_F32 = (words + 4 * tail_w + rscale) / 1e9
W_F16 = (words + 2 * tail_w + rscale) / 1e9

PEAK = 864.0     # GB/s, L40S spec
BEST = 671.0     # GB/s, best read rate measured on this card: cuBLAS FP16 GEMV (M, banc-252-2026-10-06)

print(f"bare file: words {W_WORDS:.4f} GB, f32 family {W_F32:.4f} GB, f16 family {W_F16:.4f} GB")
print(f"tails: f32 {4*tail_w/1e6:.1f} MB, f16 {2*tail_w/1e6:.1f} MB; rscale {rscale/1e6:.1f} MB")
for name, w in [("pword (words only)", W_WORDS), ("f16-family decode arm", W_F16), ("f32 bench arm", W_F32)]:
    print(f"  {name:24s} floor {w/PEAK*1e3:.3f} ms at {PEAK:.0f} GB/s, {w/BEST*1e3:.3f} ms at {BEST:.0f} GB/s")


# ---- route P, step B complete, 4B short context ----
def route_p(bt, bq, bh, a_us, c_us, h_us, prolog_ms, drain_us, fixed_ms=0.09,
            n_a=360, n_c=36, n_h=96):
    t = (W_T_SERVED / bt + W_Q_SERVED / bq + W_H_SERVED / bh) * 1e3  # ms
    t += (n_a * a_us + n_c * c_us + n_h * h_us + n_c * drain_us) / 1e3
    t += prolog_ms + fixed_ms
    return t


cases = {
    # B_T capped at the best measured read rate (verifier, finding 8)
    "fast (B_T 670)": dict(bt=670, bq=650, bh=650, a_us=2.2, c_us=3.0, h_us=1.5, prolog_ms=0.10, drain_us=1.0),
    "mid  (B_T 590)": dict(bt=590, bq=590, bh=620, a_us=2.4, c_us=3.75, h_us=2.5, prolog_ms=0.15, drain_us=1.5),
    "slow (B_T 490)": dict(bt=490, bq=490, bh=600, a_us=2.6, c_us=4.5, h_us=3.5, prolog_ms=0.20, drain_us=2.0),
    # the old top end, for the record: above every rate measured on this card
    "old top (B_T 820)": dict(bt=820, bq=650, bh=650, a_us=2.2, c_us=3.0, h_us=1.5, prolog_ms=0.10, drain_us=1.0),
}
print("\nroute P, 4B short context (E):")
for k, v in cases.items():
    t = route_p(**v)
    print(f"  {k:18s} t = {t:.3f} ms -> {1e3/t:.0f} tok/s")

# ---- 4E, the judge's model (wave2/judge.md line 261): t = 1.397 GB/B + 0.456 ms + 2.5 us ----
print("\n4E, judge's short-context model, B capped (C from E):")
for b in (560, 600, 671, 740):
    t = 1.397 / b * 1e3 + 0.456 + 0.0025
    print(f"  B = {b} GB/s: {t:.3f} ms -> {1e3/t:.0f} tok/s")

# ---- bench thresholds for rule C, mapped through the f16-family bytes ----
print("\nbench pass time -> B_T (f16 family, C):")
for t_ms in (1.42, 1.50, 1.60, 1.65, 1.80, 2.00, 2.30):
    print(f"  {t_ms:.2f} ms -> {W_F16 / t_ms * 1e3:.0f} GB/s")

# h estimator: grid-wide syncs in pchain_w (C): 95 barrier ordinals in the site
# list + 48 counter waits whose target is the whole grid + 2 trailing barriers.
N_SYNC = 95 + 48 + 2
print(f"\npchain_w grid-wide syncs: {N_SYNC}; old divisor 97 overstated h by {N_SYNC/97:.2f}x")

# C1/C2 coherence: pchain_wp = base + N_SYNC * h + prologues (C from E)
for base, h, pro in ((1.65, 2.5, 0.15), (2.00, 4.0, 0.20), (1.50, 1.5, 0.10)):
    print(f"  base {base:.2f} ms, h {h} us, prologues {pro} ms -> pchain_wp {base + N_SYNC*h/1e3 + pro:.3f} ms")

# Route P at the C1 and C2 edges, mid inputs otherwise (E)
print("\nroute P at the rule-C edges (E):")
for bt, h in ((W_F16 / 1.65e-3, 2.5), (W_F16 / 2.00e-3, 4.0)):
    t = route_p(bt=bt, bq=min(bt, 620), bh=620, a_us=2.4, c_us=3.75, h_us=h, prolog_ms=0.15, drain_us=1.5)
    print(f"  B_T {bt:.0f} GB/s, h {h} us -> {1e3/t:.0f} tok/s")

# Shared-memory pipe demand of the x reads in a Tetra stream (C): wavefronts a
# warp-block (32 blocks of 6 B) under each load shape, at a stream rate B.
CLK = 2.52e9  # Hz, SM max clock (M, a3-gpu.txt); a lower sustained clock tightens every cap
SMS = 142
def cap(wavefronts, b=PEAK):
    wb_per_cycle = b * 1e9 / SMS / (32 * 6) / CLK
    demand = wb_per_cycle * wavefronts
    return demand, (b / demand if demand > 1 else None)
print("\nx-read pipe demand at 864 GB/s (C, assumes 1 wavefront/cycle/SM for shared loads):")
for label, wf in (("24 LDS.32, stride 24 (8-way)", 192), ("24 LDS.32, stride 28 (4-way)", 96),
                  ("24 LDS.32, stride 25 (none)", 24), ("6 LDS.128, stride 24 (2-way/quarter)", 48),
                  ("6 LDS.128, stride 28 (none)", 24)):
    d, c = cap(wf)
    print(f"  {label:38s} {wf:3d} wavefronts -> demand {d:.2f}/cycle" + (f", cap {c:.0f} GB/s" if c else ", no cap"))

# The 2026-09-20 seg failure (M, llvq-cuda/tests/tetra48_segment_matches_unfused.rs:10-12)
fused, separate = -0.3080684, -0.57255346
print(f"\nseg failure, row 0: relative difference {(fused - separate) / separate:+.1%} (C)")

# 4E at 8k, the judge's model (wave2/judge.md line 262): t = 1.377 GB/B + 1.227 GB/(eta x 864) + 0.405 ms
print("\n4E at 8k, judge's model, B capped (C from E):")
for b, eta in ((560, 0.40), (671, 0.60), (740, 0.60)):
    t = 1.377 / b * 1e3 + 1.227 / (eta * 864) * 1e3 + 0.405
    print(f"  B = {b}, eta = {eta}: {t:.3f} ms -> {1e3/t:.0f} tok/s")

# persall-Tetra prediction (E): additive model on measured inputs.
#   Planes14 (a3): fused eager 4.565 ms, A1 floor at 144 launches 1.794 ms, persall 3.366 ms
#   -> intrinsic warp-per-row geometry left in persall = 3.366 - (4.565 - 1.794)
geom = 3.366 - (4.565 - 1.794)
#   Tetra bare file, tile 64 (banc-252 sweep-64): tetra48 3.701 ms, nullk 2.291 ms
beyond = 3.701 - 2.291
pa = geom + beyond
print(f"\npersall geometry residue {geom:.3f} ms (C); Tetra beyond its floor {beyond:.3f} ms (C)")
print(f"persall-Tetra, additive model: {pa:.2f} ms -> {W_F32/pa*1e3:.0f} GB/s on the f32 family (E)")
for t_ms in (1.60, 1.80, 2.00, 2.60):
    print(f"  {t_ms:.2f} ms -> {W_F32/t_ms*1e3:.0f} GB/s (f32 family)")
