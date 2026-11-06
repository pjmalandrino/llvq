# ARITHMETIC HELPER, not a code sketch and not repository code. Run with `python3 -I`
# (a tiny script, as the ground rules allow) to produce the computed and estimated numbers
# of wave5-deep/prefill.md, prereg-prefill-1.md and prereg-prefill-2.md. Inputs are named
# with their source; every range is an estimate unless the source says measured.
# Revision 2 (2026-11-02): A is tied to L4 (the old ranges combined anti-correlated extremes);
# L4 is constrained by the 8B and 14B windows of the same ppl-scelles job; exact in-place
# variants added; request, job-cost and U7 arithmetic added.
GB = 1e9
models = {  # q heads, layers, hidden, intermediate (fiche/rtbits/HF config)
    "4B": (32, 36, 2560, 9728), "8B": (32, 36, 4096, 12288),
    "14B": (40, 40, 5120, 17408), "32B": (64, 64, 5120, 25600)}
L = 8192

print("== 1. score tensor per layer, f16, second 4096-chunk of an 8k prompt (L = 8192), GB")
print("   S = heads*4096*L*2; S_b(r) = heads*r*L*2")
for name, (h, nl, d, f) in models.items():
    S = h * 4096 * L * 2
    Sb = {r: h * r * L * 2 for r in (1024, 512, 256)}
    print(f"  {name}: today 3S {3*S/GB:.2f} | Lifetimes 2S {2*S/GB:.2f} | InPlaceAll S {S/GB:.2f} | "
          f"InPlace(1024) {(S+2*Sb[1024])/GB:.2f} (512) {(S+2*Sb[512])/GB:.2f} (256) {(S+2*Sb[256])/GB:.2f} | "
          f"Rows+lean 2S_b(1024) {2*Sb[1024]/GB:.2f} (512) {2*Sb[512]/GB:.2f} | Rows 3S_b(1024) {3*Sb[1024]/GB:.2f}")

print("\n== 2. attention FLOP at 8k (4B: 32 heads, 36 layers, hd 128)")
h, nl, hd = 32, 36, 128
tri = 4 * (L * L / 2) * hd * h * nl
rect = 4 * (4096 * 4096 + 4096 * 8192) * hd * h * nl
print(f"  causal triangle {tri:.3e}  candle rectangles (2 chunks of 4096) {rect:.3e}  ratio {rect/tri:.2f}")
for tf in (362e12, 181e12, 91.6e12):
    print(f"  triangle at {tf/1e12:.1f} TFLOPS: {tri/tf:.3f} s")
for r in (1024, 512):
    s1 = sum(r * (r * (i + 1)) for i in range(4096 // r))
    s2 = sum(r * (4096 + r * (i + 1)) for i in range(4096 // r))
    print(f"  RowsTrim r={r}: area {(s1+s2)/(4096*4096+4096*8192):.3f} of both rectangles; "
          f"second chunk alone {s2/(4096*8192):.3f}; first chunk alone {s1/(4096*4096):.3f}")

print("\n== 3. dense windows of ppl-scelles-2026-10-04 (measured timestamps, 12 windows each)")
W4 = (12.5 - 1.1) / 11
W8 = (14.3 - 1.2) / 11
W14 = (21.2 - 1.8) / 11
res = 0.1 / 11            # one 0.1 s timestamp step spread over 11 windows
print(f"  W4 {W4:.4f} s, W8 {W8:.4f} s, W14 {W14:.4f} s (each +-{res:.4f} from the 0.1 s print)")
# 8B: same attention geometry as the 4B (32 q heads, 8 kv, hd 128, 36 layers), so A cancels.
# W8 - W4 = (k8 - 1) * L4_4B + (h8 - h4); k8 = L4_8B / L4_4B in [1.6, 1.9] (projections x1.91,
# elementwise x1.26-1.6); h8 - h4 in [0.005, 0.015] (head 5.1 vs 3.2 TFLOP plus log-softmax).
d84 = W8 - W4
lo8 = (d84 - 0.015) / (1.9 - 1)
hi8 = (d84 - 0.005) / (1.6 - 1)
lo8r = (d84 - 2 * res - 0.015) / (1.9 - 1)
hi8r = (d84 + 2 * res - 0.005) / (1.6 - 1)
print(f"  8B constraint: L4_4B in [{lo8:.3f}, {hi8:.3f}] s; with the print resolution [{lo8r:.3f}, {hi8r:.3f}]")
# 14B: attention x(40*40)/(32*36) = 1.389; L4_14B = k14 * L4_4B, k14 in [2.9, 3.4]; h4 in [0.03, 0.05],
# h14 in [0.035, 0.06]. W14 = 1.389 * (W4 - L4 - h4) + k14 * L4 + h14.
att14 = (40 * 40) / (32 * 36)
cands = []
for k14 in (2.9, 3.4):
    for h4 in (0.03, 0.05):
        for h14 in (0.035, 0.06):
            cands.append((W14 - att14 * (W4 - h4) - h14) / (k14 - att14))
print(f"  14B consistency: L4_4B in [{min(cands):.3f}, {max(cands):.3f}] s (attention scale {att14:.3f})")
L4lo, L4hi = 0.155, 0.25     # the range carried forward (8B constraint, rounded)
h4lo, h4hi = 0.03, 0.05
Alo, Ahi = W4 - L4hi - h4hi, W4 - L4lo - h4lo
print(f"  carried: L4_4B in [{L4lo}, {L4hi}]; A(4096^2) in [{Alo:.3f}, {Ahi:.3f}]; "
      f"A(8k) = 3A in [{3*Alo:.2f}, {3*Ahi:.2f}] s")
print(f"  per layer: (4096,4096) {Alo/36*1e3:.1f}-{Ahi/36*1e3:.1f} ms; (4096,8192) {2*Alo/36*1e3:.1f}-{2*Ahi/36*1e3:.1f} ms; "
      f"(2048,2048) {Alo/4/36*1e3:.1f}-{Ahi/4/36*1e3:.1f} ms")
# Physical floor of L4: 4B projections in a 4096-row window at the 362 TFLOPS spec, plus elementwise.
proj4 = 2560 * (4096 + 1024 + 1024 + 4096 + 3 * 9728) * 36
fl4 = 2 * proj4 * 4096
print(f"  physical floor: projections {fl4/1e12:.1f} TFLOP -> {fl4/362e12:.3f} s at 362; L4 >= ~0.12-0.13 s; "
      f"so A(8k) <= {3*(W4-0.115-0.02):.2f} s (the R4 bound)")

print("\n== 4. U7 from the same window difference (estimated)")
proj8 = 4096 * (4096 + 1024 + 1024 + 4096 + 3 * 12288) * 36
extra = 2 * (proj8 - proj4) * 4096
for t in (0.100, 0.113, 0.122, 0.132, 0.145):
    print(f"  extra projections {extra/1e12:.2f} TFLOP in {t:.3f} s -> {extra/t/1e12:.0f} TFLOPS")

print("\n== 5. GEMM route TTFT, 4B, chunks of 4096, today's attention: TTFT(8k) = 3W - L4 - 3h + 0.04 + 2 dq")
def route(dq, L4r=(L4lo, L4hi), hr=(h4lo, h4hi)):
    hi = 3 * W4 - L4r[0] - 3 * hr[0] + 0.04 + 2 * dq[1]
    lo = 3 * W4 - L4r[1] - 3 * hr[1] + 0.04 + 2 * dq[0]
    return lo, hi
for label, dq in (("dense arm (no dequant)", (0.0, 0.0)), ("C-dense f64", (0.15, 0.25)), ("C-rot16", (0.05, 0.10))):
    a, b = route(dq)
    c, e = route(dq, (0.20, 0.37))
    print(f"  {label}: {a:.2f}-{b:.2f} s (L4 {L4lo}-{L4hi}); old L4 0.20-0.37 gives {c:.2f}-{e:.2f} s")
for label, dq in (("C-dense", (0.15, 0.25)), ("C-rot16", (0.05, 0.10))):
    # Tied: TTFT(4096) = L4 + (W - L4 - h) + dq + mask = W - h + dq + mask, free of L4.
    # TTFT(2048) = L4/2 + (W - L4 - h)/4 + dq + mask = W/4 + L4/4 - h/4 + dq + mask.
    t4 = (W4 - h4hi + dq[0] + 0.01, W4 - h4lo + dq[1] + 0.02)
    t2 = (W4 / 4 + L4lo / 4 - h4hi / 4 + dq[0] + 0.005, W4 / 4 + L4hi / 4 - h4lo / 4 + dq[1] + 0.01)
    print(f"  {label}: TTFT(2048) {t2[0]:.2f}-{t2[1]:.2f} s; TTFT(4096) {t4[0]:.2f}-{t4[1]:.2f} s (tied)")
print("  grouped form on the rows path: +0.18 s at 8k over expanded (measured, served-grouped 35.02 vs 34.84)")
print("  14B from its own window: TTFT = 3 W14 - L4_14 - 3 h14 + 0.05 + 2 dq14, L4_14 = (2.9-3.4) L4_4B,")
print("  h14 0.035-0.06, dq14 = (3.3-3.9) dq_4B (estimated):")
L4_14 = (2.9 * L4lo, 3.4 * L4hi)
for label, dq in (("C-rot16", (0.05, 0.10)), ("C-dense", (0.15, 0.25))):
    lo = 3 * W14 - L4_14[1] - 3 * 0.06 + 0.05 + 2 * 3.3 * dq[0]
    hi = 3 * W14 - L4_14[0] - 3 * 0.035 + 0.05 + 2 * 3.9 * dq[1]
    print(f"    {label}: {lo:.1f}-{hi:.1f} s")

print("\n== 6. with a flash prefill attention (0.08-0.22 s at 8k), mask build gone")
for label, dq in (("C-dense", (0.15, 0.25)), ("C-rot16", (0.05, 0.10))):
    two = (2 * L4lo + 2 * dq[0] + 0.08, 2 * L4hi + 2 * dq[1] + 0.22)
    one = (2 * L4lo + dq[0] + 0.08, 2 * L4hi + dq[1] + 0.22)
    print(f"  {label}: two chunks {two[0]:.2f}-{two[1]:.2f} s; one chunk {one[0]:.2f}-{one[1]:.2f} s")

print("\n== 7. an 8,192 + 512 request, grouped served decode (measured: 12.90 ms at 8,192, +0.67 ms per 1k positions)")
dec512 = sum(12.90 + 0.67 * (i / 1000) for i in range(512)) / 1000
print(f"  512 decode tokens from 8,192: {dec512:.2f} s; today {35.02 + dec512:.1f} s")
for label, dq in (("C-dense", (0.15, 0.25)), ("C-rot16", (0.05, 0.10))):
    a, b = route(dq)
    print(f"  route {label}: {a + dec512:.1f}-{b + dec512:.1f} s")
print(f"  route + flash: {0.44 + dec512:.1f}-{1.22 + dec512:.1f} s")

print("\n== 8. 1a rows path floor, 4B, f32 activation read per weight per row from shared")
tetra, int4 = 2_862_612_480, 770_703_360
by = (tetra + int4) * 4
bw = 142 * 128 * 2.52e9
print(f"  shared bytes per prompt token {by/GB:.2f} GB; at {bw/1e12:.1f} TB/s: {by/bw*1e3:.3f} ms; Tetra 2-way: {(tetra*8+int4*4)/bw*1e3:.3f} ms")
for per in (0.45, 0.80):
    for att in (3 * Alo, 3 * Ahi):
        print(f"  {per} ms/token + {att:.2f} s attention: TTFT(8k) {8192*per/1e3+att:.1f} s")

print("\n== 9. FP64 instructions of an exact dense dequant, 4B (no FMA: one instr per mul or add)")
q, k, g, u, dn = 4096*2560, 1024*2560, 9728*2560, 9728*2560, 2560*9728
n5 = (q + k + g + u) * 36; n19 = dn * 24
per5 = 2.5 + 2*5 + 9 + 1; per19 = 2.5 + 2*19 + 9 + 1
ins = n5*per5 + n19*per19
peak64 = 142 * 2 * 2.52e9
print(f"  {ins/1e9:.1f} G instr; AD10x FP64 peak {peak64/1e12:.3f} T instr/s -> {ins/peak64:.3f} s at peak, x1.3-2 realistic")
for rate in (0.70e12, 0.50e12, 0.30e12):
    print(f"  probe reads {rate/1e12:.2f} T/s: one pass {ins/rate:.3f} s at that peak, {1.3*ins/rate:.2f}-{2*ins/rate:.2f} s realistic")

print("\n== 10. C-dense side tables, 4B: f64 row scales and f32 tail of the Tetra matrices")
rows = (4096+1024+9728+9728)*36 + 2560*24
tailv = (4096+1024+9728+9728)*36*16 + 2560*24*8
side = rows*8 + tailv*4
print(f"  rows {rows}, tail values {tailv}, bytes {side/1e6:.1f} MB = {side*8/4.02e9:.3f} b/param over 4.02 G params")

print("\n== 11. transient VRAM of the GEMM route, f16 scratch of the largest matrix")
for name, (h, nl, d, f) in models.items():
    print(f"  {name}: {f*d*2/1e6:.1f} MB")

print("\n== 12. int4 rows kernel staging, f32, T multiple of 256 columns")
for R in (4, 8, 16):
    for cap in (49152, 101376):
        T = (cap // (R*4)) // 256 * 256
        print(f"  R={R} cap={cap}: T={T}  tiles at 9728: {-(-9728//T)}  at 17408: {-(-17408//T)}")

print("\n== 13. job prices, l40sx1 at $1.80/h (ops/run.py)")
rate = 1.80 / 60
# prereg-1: container 3, oracle 2, bench door 3, served door 1.5, 3a 2, 3b, 3c-3e 1
# 3b: round 0, 7 arms x (2,048 + 4,096 + 8,192 prefill + 3 decodes of 256 tokens); rounds 1-3, 3 arms at 8,192
b3 = 7 * (8.20 + 17.03 + 35.02 + 3 * 3.0) / 60 + 3 * 3 * 35.02 / 60
p1 = 3 + 2 + 3 + 1.5 + 2 + b3 + 1
print(f"  prereg-1 full: 3b {b3:.1f} min, total {p1:.1f} min = ${p1*rate:.2f}; ride-along 3a+3c+3d+3e about 3 min = ${3*rate:.2f}")
# prereg-2 step 5 per window, grouped served decode 8.78 -> 10.20 ms (+0.3 ms scoring)
step_ms = (8.78 + 10.20) / 2 + 0.3
dec = 2047 * step_ms / 1000
w2 = (17.03 + 1.0) + (1.3 + 1.0) + 8.20 + 0.55 + 2 * dec
w2x = (17.03 + 1.0) + (1.3 + 1.0) + 8.20 + 0.55 + 2 * 2047 * ((12.09 + 16.28) / 2 + 0.3) / 1000
w3 = w2 + (1.2 + 1.0) + 0.55 + dec
print(f"  step 5 a window: two arms {w2:.1f} s (grouped decode; {w2x:.1f} s if expanded); third arm adds {w3-w2:.1f} s")
print(f"  anchor gqa-ppl-r2: 123.5 s a window (measured) = 2 x ~18 s prefill paths + 4,095 lockstep steps at "
      f"{(123.5-36)/4095*1e3:.1f} ms")
fixed = 3 + 2 + 4 + 1.5 + 1.5 + 7
t2a, t3a = fixed + 24 * w2 / 60, fixed + 24 * w3 / 60
t2x = fixed + 24 * w2x / 60
print(f"  prereg-2 two arms: {t2a:.0f} min = ${t2a*rate:.2f} (expanded decode {t2x:.0f} min = ${t2x*rate:.2f}); "
      f"three arms: {t3a:.0f} min = ${t3a*rate:.2f}; timeout 90 min = ${90*rate:.2f}")
