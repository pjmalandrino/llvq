# Arithmetic behind wave5-deep/mac.md sections 1, 3.2 and 5, and the predictions of prereg-mac-1/-2.
# Not experiment code. It reads no file, writes nothing and touches no device. Its expressions were
# evaluated with `python3 -I` through stdin during the deep dive (a tiny computation the ground rules
# allow); the figures quoted in the documents come from that evaluation.
# Labels: inputs are measured (journal or commit named) or computed from code; outputs are computed
# when every input is measured or computed, estimated otherwise.
GB = 1e9
B_MEAS = 340.0  # GB/s, measured: commit 5ec72b0 (512 MB streamed, best of 7)

# Weight bytes a token = projections + the head table (computed, wave-1 U1; head = vocab*d*(4/8+4/64), q4 g64).
proj = {"4B": 1.158, "8B": 2.064, "14B": 4.168}
head = {"4B": 0.219, "8B": 0.350, "14B": 0.438}
W = {k: proj[k] + head[k] for k in proj}
# Device bytes: the loader's HOST count printed on the L40S (paper-table-2026-09-25). Memory figure only.
DEV = {"4B": 1.38, "8B": 2.76, "14B": 5.04}
PARAMS = {"4B": 4_022_468_096, "8B": 8_190_735_360, "14B": 14_768_307_200}  # fiche-4b:66, rtbits-planes-8b, rtbits-14b


def int4_bytes(records):  # (count, d_out, d_in); 4.25 b/weight = int4 + f16 scale + f16 bias per 128
    return sum(c * o * i for c, o, i in records) * 4.25 / 8 / GB


INT4 = {
    "4B": int4_bytes([(36, 1024, 2560), (36, 2560, 4096), (12, 2560, 9728)]),
    "8B": int4_bytes([(36, 1024, 4096), (17, 4096, 12288)]),
    "14B": int4_bytes([(40, 1024, 5120), (40, 5120, 5120), (19, 5120, 17408)]),
}
TILED = {
    "4B": int4_bytes([(12, 2560, 9728)]),
    "8B": int4_bytes([(17, 4096, 12288)]),
    "14B": int4_bytes([(19, 5120, 17408)]),
}
TETRA = {k: proj[k] - INT4[k] for k in proj}

print("== bytes, computed")
for k in W:
    print(f"{k}: weights a token {W[k]:.3f} GB (Tetra {TETRA[k]:.3f}, int4 {INT4[k]:.3f}, head {head[k]:.3f}),"
          f" tiled int4 {TILED[k]:.3f} ({TILED[k] / W[k]:.1%}); device {DEV[k]} = x{DEV[k] / W[k]:.3f} of it;"
          f" {DEV[k] * 1e9 * 8 / PARAMS[k]:.3f} b/param")
# -> 4B 1.377, x1.002, 2.745 b/param; 8B 2.414, x1.143, 2.696; 14B 4.606, x1.094, 2.730

print("\n== roofline at 340 GB/s (computed bound, not a forecast)")
for k in W:
    print(f"{k}: {1e3 / (W[k] / B_MEAS * 1e3):.0f} tok/s")  # 247 / 141 / 74

IQ2 = 114.02 * 1.246620832      # measured tok/s x file bytes, m3-iq2-metal-2026-08-30 -> 142.1 GB/s
F16_LLAMA = 43.08 * 8.051284928  # -> 346.8 GB/s
TORCH_MPS = (35 * 8.045, 37 * 8.045)  # hf-chat-4b-2026-10-04:56, "not a benchmark" -> 282-298 GB/s
print(f"\nllama.cpp IQ2_XXS {IQ2:.1f} GB/s, llama.cpp f16 {F16_LLAMA:.1f}, torch mps f16 {TORCH_MPS[0]:.0f}-{TORCH_MPS[1]:.0f}"
      " (computed, other stacks: GB/s only, rule 5)")

# The q5 anchors. 5a75338: 48.5 tok/s, matvec 12.66 ms (252 launches, projections only),
# rotation 2.72 ms (144 launches), rest ~5.2 ms; 1000/48.5 = 20.62 ms. 5ec72b0 read 49.8 (20.08 ms)
# later with the same kernels: pairing 5.2 with 20.1 would leave 4.7, so the pairing is 48.5 / 20.6.
q5_head = 151936 * 2560 * (8 + 32 / 64) / 8 / GB  # q8 g64 head, 0.413 GB, inside the 5.2 ms
print(f"\nq5 step at 48.5 tok/s: {1000 / 48.5:.2f} ms = 12.66 + 2.72 + 5.2")
for rh in (150, 250):
    print(f"  q8 head {q5_head:.3f} GB at {rh} GB/s: {q5_head / rh * 1e3:.2f} ms -> non-weight rest {5.2 - q5_head / rh * 1e3:.2f} ms")
# -> 2.44-3.55 ms: U3 is 2.4-3.6 ms after the head (estimated)

# Rotation cost model t(n) = a + b n, fitted on 12 us at n=2560 (31a5acb) and the 144-launch total 2.72 ms (5a75338).
t2560 = 12.0
s = (2720 - 72 * t2560) / 36            # t(4096) + t(9728)
b = (s - 2 * t2560) / (13824 - 2 * 2560)
a = t2560 - 2560 * b
rot = lambda n: a + b * n               # noqa: E731
# Launch counts measured on the card: 96 (4B), 127 (8B), 101 (14B) rot_launches a token
# (paper-table-2026-09-25-brut/paper-served-*-r2/fused-q4-256.txt:3).
T_ROT = {
    "4B": (72 * rot(2560) + 24 * rot(9728)) / 1e3,
    "8B": (108 * rot(4096) + 19 * rot(12288)) / 1e3,
    "14B": (80 * rot(5120) + 21 * rot(17408)) / 1e3,
}
print(f"\nrotation fit (estimated): t(n) = {a:.2f} + {b * 1e3:.4f}e-3 n us;"
      f" per token 4B {T_ROT['4B']:.2f} ms, 8B {T_ROT['8B']:.2f}, 14B {T_ROT['14B']:.2f}")

REST = {"4B": (2.45, 3.55), "8B": (2.45, 3.55), "14B": (2.45 * 40 / 36, 3.55 * 40 / 36)}  # scaled by layers
R_TETRA = {"4B": (74, 77), "8B": (75, 120), "14B": (75, 120)}  # GB/s; 4B computed from q5, 8B/14B estimated
F_MS = {}
print("\n== F, the fused step after the wall (estimated; int4 and q4 head at 75-250 and 150-250 GB/s)")
for k in W:
    lo = TETRA[k] / R_TETRA[k][1] * 1e3 + INT4[k] / 250 * 1e3 + head[k] / 250 * 1e3 + T_ROT[k] + REST[k][0]
    hi = TETRA[k] / R_TETRA[k][0] * 1e3 + INT4[k] / 75 * 1e3 + head[k] / 150 * 1e3 + T_ROT[k] + REST[k][1]
    F_MS[k] = (lo, hi)
    print(f"{k}: {lo:.1f}-{hi:.1f} ms -> {1e3 / hi:.1f}-{1e3 / lo:.1f} tok/s -> {W[k] * 1e3 / hi:.0f}-{W[k] * 1e3 / lo:.0f} GB/s of weights;"
          f" tiled records {TILED[k] / 250 * 1e3:.2f}-{TILED[k] / 75 * 1e3:.2f} ms")
# -> 4B 45-61 tok/s, 8B 28-47, 14B 15-28

# The f16 head (arms Fh and D). model.rs:1395 calls broadcast_matmul, which copies the transposed table
# every token (candle-core 0.9.2 tensor.rs:1545-1556): read + write 0.778 GB, then the gemm reads it.
h16 = 151936 * 2560 * 2 / GB
traffic = 3 * h16                                # 2.334 GB a token at 4B
l40s_extra = 1000 / 55.0 - 1000 / 113.8          # paper-table: f16-table arm against q4-table arm, 4B, 9.39 ms
frac = traffic / (l40s_extra / 1e3) / 864        # 29 % of the L40S's 864 GB/s
H16 = {f: traffic / (f * B_MEAS) * 1e3 for f in (0.40, frac, 0.15)}
print(f"\nf16 head: {traffic:.3f} GB of traffic; L40S {l40s_extra:.2f} ms ({frac:.0%} of 864);"
      f" on the M3 Max at 40 / {frac:.0%} / 15 % of 340: {H16[0.40]:.1f} / {H16[frac]:.1f} / {H16[0.15]:.1f} ms (estimated)")

f_mid = 1000 / 52                                 # F's signed point, 52 tok/s
fh_lo = F_MS["4B"][0] - head["4B"] / 250 * 1e3 + H16[0.40]
fh_hi = F_MS["4B"][1] - head["4B"] / 150 * 1e3 + H16[0.15]
fh_mid = f_mid - head["4B"] / 200 * 1e3 + H16[frac]
print(f"Fh 4B: {fh_lo:.1f}-{fh_hi:.1f} ms -> {1e3 / fh_hi:.1f}-{1e3 / fh_lo:.1f} tok/s, point {1e3 / fh_mid:.1f}")
# -> 15.0-30.6 tok/s, point 23.8; head phase Fh - F about 24 - 1.1 = 23 ms (estimated)

D_MS = (1000 / 9.6, 1000 / 7.5)                   # measured low-grade: 976549b 7.5, a10c92c 7.9, kvq8-4b 9.6
d_mid = 1000 / 8.5
print(f"D: {D_MS[0]:.0f}-{D_MS[1]:.0f} ms; {8.045 * 7.5:.0f}-{8.045 * 9.6:.0f} GB/s of weights,"
      f" {(8.045 + 2 * h16) * 7.5:.0f}-{(8.045 + 2 * h16) * 9.6:.0f} GB/s counting the head copy")
print(f"raw F/D x{D_MS[0] / F_MS['4B'][1]:.2f}-x{D_MS[1] / F_MS['4B'][0]:.2f}, point x{d_mid / f_mid:.2f}")
print(f"same-head Fh/D x{D_MS[0] / fh_hi:.2f}-x{D_MS[1] / fh_lo:.2f}, point x{d_mid / fh_mid:.2f}")
# Signed intervals widen F to [40, 64] and Fh to [14, 32]: raw [x4.2, x8.5], same head [x1.4, x4.3].
print(f"with the widened F [40, 64]: raw x{40 / 9.6:.2f}-x{64 / 7.5:.2f};"
      f" widened Fh [14, 32]: same head x{14 / 9.6:.2f}-x{32 / 7.5:.2f}")

print("\n== dense arms at 8B and 14B, at the 4B dense arm's 60-77 GB/s of weights (estimated)")
for k, dd in (("8B", 16.38), ("14B", 29.54)):
    print(f"{k}: {dd} GB -> {60 / dd:.2f}-{77 / dd:.2f} tok/s")
for r in (2.0, 2.6):
    print(f"14B dense arm, 9 rounds of 128 tokens at {r} tok/s: {9 * 128 / r:.0f} s")
# -> 443-576 s, plus a load of about 4-6 min: 12-16 min, not most of an hour

print("\n== L7 on the Mac, the expanded form up to 4,096 positions at 16-25 tok/s (estimated)")
print(f"cache traffic at the 2,048 mean: {16.9 * 2048 / 8192:.2f} GB + {W['4B']:.3f} of weights a step")
for rate in (16, 25):
    print(f"  {rate} tok/s: 12 windows (49,140 steps) {49140 / rate / 60:.0f} min an arm;"
          f" 4 windows (16,380) {16380 / rate / 60:.0f} min an arm")
for ms_row in (15, 17):
    print(f"  chunked prefill of 4,096 rows at {ms_row} ms a row: {4096 * ms_row / 1e3:.0f} s a window")
# -> decode 33-51 min an arm at 12 windows, 11-17 at 4; prefill 61-70 s a window
