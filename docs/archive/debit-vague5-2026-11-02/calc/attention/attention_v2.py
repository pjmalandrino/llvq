# Arithmetic for wave5-deep/attention.md, revision 2 (2026-11-02). Not repository code.
# Tiny arithmetic only: reads no file, starts no compute. Run once with `python3 -I`.
# Labels: inputs marked M are measured (journal named), E are estimated (judgment);
# every output is computed from them, hence *estimated* as soon as one E input enters.

BW = 864.0          # GB/s, L40S spec (format-noyau.md uses 864)
SMS = 142           # L40S SMs
CLK = 2.52e9        # Hz, boost
F16_TOK = 147456    # B per token, 4B, f16 cache (kvq.rs test)  M (code)
Q8_TOK = 78336      # B per token, 4B, q8 g64 (kvq.rs test)       M (code)

# ---- 1. today's slope, served-grouped-2026-11-02.txt medians (M)
ms = {5: 7.43, 2048: 8.78, 4096: 10.20, 8192: 12.90}
print("1. today's context slope (computed from served-grouped medians)")
for L in (2048, 4096, 8192):
    gb = F16_TOK * L / 1e9
    r = gb / ((ms[L] - ms[5]) / 1e3)
    print(f"   L={L}: {r:6.1f} GB/s, eta {r/BW:.3f}")

# ---- 2. the lead's own 5-token share (E: 2.1-2.3 us per in-situ launch; kernel fixed 3 us a layer)
print("2. 5-token step, lead's own share")
for removed, who in ((5, "kernel absorbs the KV write (6 -> 1 launches)"),
                     (3, "branch prologue already writes KV (4 -> 1)")):
    for lat in (2.1, 2.3):
        step = ms[5] - 36 * removed * lat * 1e-3 + 36 * 3e-3
        print(f"   {who}, {lat} us: step {step:.2f} ms, x{ms[5]/step:.3f}, {1000/step:.1f} tok/s")

# ---- 3. end-to-end projection at the 7.12 ms step (the lead's own launches only)
print("3. decode tok/s projection, step 7.12 ms + cache read at eta")
for L in (2048, 4096, 8192):
    f = [1000 / (7.12 + F16_TOK * L / 1e9 / (e * BW) * 1e3) for e in (0.60, 0.75, 0.85)]
    q = [1000 / (7.12 + Q8_TOK * L / 1e9 / (e * BW) * 1e3) for e in (0.60, 0.75, 0.85)]
    print(f"   L={L}: f16 {[round(x) for x in f]}  q8 {[round(x) for x in q]}  today {1000/ms[L]:.1f}")

# ---- 4. issue-count model, per lane, per tile of 8 positions, per row (E counts)
def per_lane_tile(G, sched, f32_tiles=False):
    kv = 8 + 64 + 8 + 64                      # K and V: LDS.128 + one conversion per element
    if f32_tiles:                              # k > 1: tiles widened once per CTA, rows read f32
        kv = 16 + 16 + 2
    dot, acc, d, ts = 64 * G, 72 * G, 2 * G, 8 * G
    ex2 = 13                                   # cmp/branch 2, floor 1, sub 1, 6 FFMA, F2I, SHL, IADD
    if sched == "allreduce":                   # the sketch: every lane holds every score
        return kv + dot + acc + d + ts + 64 * G + 8 * G + 8 * G + 9 * G * ex2
    # reduce-scatter: same xor 8,4,2,1 tree, each lane ends with 8G/16 scores (padded to even)
    n = 8 * G if (8 * G) % 32 == 0 else 48     # G=5: 40 padded to 48
    rs, m = 0, n
    while m > n // 16:
        rs += 2 * m                            # m/2 SHFL + m/2 FADD + m SEL
        m //= 2
    own = n // 16
    bcast = (own + 1) + (8 * G + G + 3) // 4   # STS of own p and m, LDS.128 of all p and carries
    return kv + dot + acc + d + ts + rs + own + 8 + own * ex2 + ex2 + bcast

print("4. warp instructions per (row, position, KV head) = per-lane-per-tile / 16")
W = {}
for G in (4, 5):
    for s, f32 in (("allreduce", False), ("reduce-scatter", False), ("reduce-scatter", True)):
        n = per_lane_tile(G, s, f32)
        W[(G, s, f32)] = n / 16
        tag = s + (" + f32 tiles (k>1)" if f32 else "")
        print(f"   G={G} {tag:32s}: {n:5d} per lane per tile, {n/16:6.1f} warp instr")

# ---- 5. compute time against DRAM time, per layer (E: IPC per resident warp 0.25-0.45, cap 4)
def compute_us(L, k, G, w, mapping):
    total = k * L * 8 * w                      # warp instructions per layer
    if k == 1:
        halfwarps = L // 64 * 8                # one unit per group in every mapping at k = 1
        if mapping == "split":                 # CTAs = splits x heads; spread over SMs
            active = min(SMS, (L // 256) * 8)
        else:
            active = min(SMS, (L // 64) * 8)
    else:
        if mapping == "split":                 # group r = row r, 4 units in sequence
            halfwarps = k * (L // 256) * 8
            active = min(SMS, (L // 256) * 8)
        else:                                  # one CTA per unit, rows across groups
            halfwarps = k * (L // 64) * 8
            active = min(SMS, (L // 64) * 8)
    warps_sm = halfwarps / 2 / active
    out = []
    for ipw in (0.45, 0.25):
        ipc = min(4.0, warps_sm * ipw)
        out.append(total / active / ipc / CLK * 1e6)
    return out, warps_sm

print("5. per layer, compute (us, fast/slow IPC) against DRAM (us at eta 0.85/0.60)")
for L in (2048, 4096, 8192):
    dram = [4096 * L / (e * BW * 1e9) * 1e6 for e in (0.85, 0.60)]
    print(f"   L={L}: DRAM {dram[0]:.1f}-{dram[1]:.1f} us")
    for k in (1, 8):
        for mapping in ("split", "unit"):
            for key in ((4, "allreduce", False), (4, "reduce-scatter", k > 1)):
                c, wsm = compute_us(L, k, 4, W[key], mapping)
                lo = max(c[0], dram[0]) / dram[0]
                hi = max(c[1], dram[1]) / dram[1]
                hi2 = max(c[1], dram[0]) / dram[0]
                print(f"     k={k} {mapping:5s} {key[1]:14s}: compute {c[0]:5.1f}-{c[1]:5.1f} us"
                      f" ({wsm:4.1f} warps/SM); t(k)/t(1) ~ {min(lo,hi):.2f}-{max(lo,hi,hi2):.2f}")

# ---- 6. the kernel's verify extra-row cost, against the speculation lead's 2.75 ms a row
print("6. extra verify row at 8k: a x t_attn(1), t_attn(1) ~ 2.0 ms (eta 0.75, 36 layers)")
for r8 in (1.1, 1.3, 1.7, 2.5):
    a = (r8 - 1) / 7
    print(f"   t(8)/t(1)={r8}: a={a:.3f}, {a*2.0:.3f} ms a row, {a*2.0/2.75*100:.0f} % of 2.75 ms")

# ---- 7. P1 prefill on CUDA cores at 8k (FLOP computed; overhead and IPC E)
pairs = 8192 * 8193 / 2
flop = pairs * 32 * 512 * 36
t_ffma = flop / (SMS * 128 * 2 * CLK)
print(f"7. P1: {flop/1e12:.1f} TFLOP, {t_ffma:.3f} s at the FFMA peak")
for ovh in (909 / 544, 1516 / 544):            # instructions over FFMA, RS and AR at G=4
    for ipc in (3.0, 1.5):
        print(f"   overhead {ovh:.2f}, IPC {ipc}: {t_ffma*ovh*4/ipc:.2f} s")

# ---- 8. int8 R1 dequant, per layer at 8k (E: 4 instr an element, IPC 2-4)
wi = 256 * 4 / 32 * 8192 * 8 / SMS
print(f"8. R1 dequant at 8k: {wi/1e3:.1f} K warp instr per SM, {wi/4/CLK*1e6:.1f}-{wi/2/CLK*1e6:.1f} us;"
      f" int8 DRAM {2176*8192/(0.78*BW*1e9)*1e6:.1f} us at eta 0.78")

# ---- 9. job prices at $1.80/h (E durations; anchors in docs/data/jobs.csv)
USD_MIN = 1.80 / 60
def usd(m):
    return round(m * USD_MIN, 2)
print("9. job prices")
# V6: three arms, decode path only for the kernel arms; served arm's prefill once per window
s4 = {"grouped": 8.8 + 0.5, "kernel": 7.6 + 0.5, "kernel_q8": 7.6 + 0.5 + 4.75}     # ms a step, ctx 0-4k
s8 = {"grouped": 11.6 + 0.5, "kernel": 8.6 + 0.5, "kernel_q8": 8.6 + 0.5 + 4.75}    # ms a step, ctx 4-8k
w4 = (sum(s4.values()) * 4095 / 1e3 + 17.4 + 2 * 17.4 / 12) / 60    # + one identity window
w8 = (sum(s8.values()) * 4096 / 1e3 + 3 * 17.4) / 60
v6 = 12 * w4 + 6 * w8 + 8
v6_two = 12 * ((s4["grouped"] + s4["kernel"]) * 4095 / 1e3 + 17.4) / 60 \
       + 6 * ((s8["grouped"] + s8["kernel"]) * 4096 / 1e3 + 2 * 17.4) / 60 + 8
print(f"   V6 three arms: {v6:.0f} min, ${usd(v6)}; two arms only: {v6_two:.0f} min, ${usd(v6_two)}")
old = 12 * 123.5 / 60 + 12 * (2 * 8192 * 4.25e-3 + 2 * 17.4 + (12.1 + 9.1) * 4096 / 1e3) / 60 + 8
print(f"   V6 as first written (both paths, both lengths): {old:.0f} min, ${usd(old)}")
v5 = 24 * 2 * 8192 * 4.25e-3 / 60 + 6 * (2 * 17.4 + (12.1 + 16.9) * 4096 / 1e3) / 60 + 8
print(f"   V5: {v5:.0f} min, ${usd(v5)}")
for lo, hi, name in ((5, 14, "V2 FlashInfer"), (6, 8, "V4a per run"), (13, 16, "V4b")):
    print(f"   {name}: {lo}-{hi} min, ${usd(lo)}-{usd(hi)}")
tot_lo = usd(5) + usd(6) + usd(13) + usd(v5) + usd(v6) + 1.9 + 0.55
tot_hi = usd(14) + 2 * usd(8) + usd(16) + usd(v5) + usd(v6) + 2.1 + 0.55
print(f"   4B ladder V2+V4a+V4b+V5+V6+V7+V8: ${tot_lo:.2f}-{tot_hi:.2f}; with V2b +$0.12-0.15")
