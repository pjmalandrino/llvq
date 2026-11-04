# Verifier's arithmetic check for wave5-deep/attention.md, not repository code.
# Run once with python3 -I (a tiny arithmetic script, as the ground rules allow); it reads no file and starts no compute.
BW = 864.0  # GB/s
toks = {5: 134.5, 2048: 113.8, 4096: 98.0, 8192: 77.5}   # measured, served-grouped grouped arm
ms = {L: 1000.0 / t for L, t in toks.items()}
print("ms/token from tok/s:", {L: round(v, 3) for L, v in ms.items()})
f16 = 147456; q8 = 78336
for L in (2048, 4096, 8192):
    gb = f16 * L / 1e9
    d = ms[L] - ms[5]
    print(f"L={L}: bytes {gb:.4f} GB, extra {d:.3f} ms, rate {gb/d*1e3:.1f} GB/s, eta {gb/d*1e3/BW:.3f}")
    # journal's 2-decimal ms
jms = {5: 7.43, 2048: 8.78, 4096: 10.20, 8192: 12.90}
for L in (2048, 4096, 8192):
    gb = f16 * L / 1e9; d = jms[L] - jms[5]
    print(f"  (journal ms) L={L}: rate {gb/d*1e3:.1f} GB/s eta {gb/d*1e3/BW:.4f}")
fixed = 36 * 3e-3  # ms
print("attention-only table:")
for L in (2048, 4096, 8192):
    today = (jms[L] - jms[5]) + 6 * 36 * 2.2e-3
    k16 = [f16 * L / 1e9 / (e * BW) * 1e3 + fixed for e in (0.60, 0.75, 0.85)]
    k8 = [q8 * L / 1e9 / (e * BW) * 1e3 + fixed for e in (0.60, 0.75, 0.85)]
    print(f"  {L}: today {today:.2f}; f16 {[round(x,2) for x in k16]}; q8 {[round(x,2) for x in k8]}")
print("end-to-end table:")
for L in (2048, 4096, 8192):
    for base in (7.12, 6.80):
        r = [1000.0 / (base + f16 * L / 1e9 / (e * BW) * 1e3) for e in (0.60, 0.75, 0.85)]
        print(f"  {L} f16 step {base}: {[round(x,1) for x in r]}")
    r = [1000.0 / (7.12 + q8 * L / 1e9 / (e * BW) * 1e3) for e in (0.60, 0.75, 0.85)]
    print(f"  {L} q8  step 7.12: {[round(x,1) for x in r]}")
# same tables but WITH the kernel fixed cost at long context counted, and 2.6 us per launch
for lat in (2.1, 2.3, 2.6):
    s = jms[5] - 5 * 36 * lat * 1e-3 + fixed
    print(f"5-token step if launches cost {lat} us and kernel fixed 3 us/layer: {s:.3f} ms -> {1000/s:.1f} tok/s")
# prereg-2 in-situ ratio predictions
for e in (0.70, 0.75):
    print("eta", e, [round(jms[L] / (7.12 + f16 * L / 1e9 / (e * BW) * 1e3), 3) for L in (2048, 4096, 8192)], "5:", round(jms[5]/7.12, 4))
# prereg-1 R<400 band arithmetic
for e, base in ((0.30, 7.12), (0.50, 6.80), (0.30, 6.80), (0.50, 7.12)):
    print(f"R<400 band: eta {e} base {base}: ratio {jms[8192]/(base + 1.20796/(e*BW)*1e3):.3f}")
# prereg-1 predicted tg at depth with R=600
for t0 in (310, 85):
    t = 1000/t0 + f16 * (8192 + 64) / 1e9 / 600 * 1e3
    print(f"tg128 at d=8192 from {t0} at R=600: {1000/t:.1f}")
# ex2 placeholder accuracy (Taylor coefficients) at f in [0,1]
import math
C = [0.693147182, 0.240226507, 0.0555041087, 0.00961812911, 0.00133335581, 0.000154035304]
worst = 0
for i in range(0, 1001):
    f = i / 1000
    p = C[5]
    for c in reversed(C[:5]):
        p = p * f + c
    p = p * f + 1.0
    rel = abs(p - 2 ** f) / 2 ** f
    worst = max(worst, rel)
print(f"placeholder ex2 worst rel err on [0,1]: {worst:.3e} = {worst/2**-24:.1f} ulp(2^-24)")
# INV255 literal
import struct
v = struct.unpack('f', struct.pack('f', 1/255))[0]
lit = struct.unpack('f', struct.pack('f', 0.0039215688593685627))[0]
print("f32(1/255) =", repr(v), "literal ->", repr(lit), "equal:", v == lit)
# combine scratch, R1 bytes
print("scratch 8k:", 8*32*4*520/1e6, "MB; int8 per pos-layer:", 2*(128+8)*8, "x36 =", 2*(128+8)*8*36)
# prefill attention flops
pairs = 8192*8193/2
print("prefill attn TFLOP:", pairs*32*512*36/1e12, "s at 91.6:", pairs*32*512*36/91.6e12, "at 362:", pairs*32*512*36/362e12, "at 181:", pairs*32*512*36/181e12)
