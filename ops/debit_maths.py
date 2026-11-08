# /// script
# dependencies = ["numpy"]
# ///
"""Every number of docs/recherche-debit-maths.md that is computed, at $0.

    uv run ops/debit_maths.py

Part 1 fits the instruction-linear model on the F1 floor arms (mesures/f1-rang-variantes-2026-09-05.txt) and
projects the batch-1 levers at the three served doors. Part 2 is the issue balance on two cards. Part 3 is the batch
arithmetic. Part 4 simulates the activation rounding of the int8 lever. Instruction counts come from
ops/tetra-sass-probe.sh (devel ptxas 12.4, sm_89).
"""
import math
# --- calibration: F1 floor arms, 2026-09-05, one process, 0.908 GB = 151.3M blocks (all 252 4B matrices)
meas = {"word": 2.825, "f1r": 5.515, "v1": 3.900, "v2": 5.625, "v3": 3.875}
inst = {"word": 50, "f1r": 499, "v1": 238, "v2": 523, "v3": 219}
c = (meas["v3"] - meas["word"]) / (inst["v3"] - inst["word"])
print(f"slope c = {c*1e3:.3f} us per instruction-per-block over 151.3M blocks")
for k in ("f1r", "v1", "v2"):
    p = meas["word"] + c * (inst[k] - inst["word"])
    print(f"  {k}: predicted {p:.3f} measured {meas[k]:.3f} ({(p/meas[k]-1)*100:+.1f} %)")
blocks_f1 = 0.908e9 / 6
print(f"  issue rate implied: {blocks_f1/32/(c*1e-3)/(142*2.52e9):.2f} warp-instr/clk/SM of 4")
share = c * (237 - 50) / (0.644 + c * (237 - 50))
print(f"instruction-driven share of above-floor Tetra time at 237: {share:.3f}")

# --- sizes: Tetra weights, door ms
sizes = {
 "4B": dict(tetra=36*(4096*2560+1024*2560+2*9728*2560)+24*2560*9728, door=1000/221.5,
            sites=36+36+24),
 "8B": dict(tetra=36*(4096*4096+1024*4096+4096*4096+2*12288*4096)+19*4096*12288, door=1000/142.5,
            sites=36*3+19),
 "14B": dict(tetra=40*(5120*5120+1024*5120+2*17408*5120)+21*5120*17408, door=1000/86.3,
            sites=40+40+21),
}
rows4 = 1.81  # ms, Tetra rows in situ at 4B (plan-300 section 4, 4E v6 stamps; v8 and v9 kept that loop)
# door tok/s: ETAT section 2 (served-v9 journals). Tetra weights: the sealed placements (how-it-works section 8,
# preregistration-sealed-8b-14b-2026-09-23: 8B v + down@17 layers int4, 14B v + o + down@10-28 int4).
# quant_us: the int8 rounding of the activation, a few hundred cycles a site (estimated).
for s, d in sizes.items():
    d["blocks"] = d["tetra"] / 24
    d["rows"] = rows4 * d["blocks"] / sizes["4B"]["tetra"] * 24
    d["dram"] = d["blocks"] * 6.45 / 864e9 * 1e3
# lane items
def items(nb): return math.ceil(nb / 32)
def lane_ratio(groups):
    a = sum(r * items(nb) for r, nb in groups); b = sum(r * nb / 32 for r, nb in groups); return b / a
ratios = {
 "4B": lane_ratio([(24576*36, 106), (2560*24, 405)]),
 "8B": lane_ratio([(33792*36, 170), (4096*19, 512)]),
 "14B": lane_ratio([(40960*40, 213), (5120*21, 725)]),
}
quant_us = 0.4
print()
for s, d in sizes.items():
    R, door = d["rows"], d["door"]
    ins = R * share
    def door_after(save): return 1000 / (door - save)
    sv = {
      "exact micro 237->222": ins * (1 - 172/187),
      "lane rotation": ins * (1 - ratios[s]),
      "exact both": ins * (1 - 172/187 * ratios[s]),
      "int8 dp4a 237->152": ins * (1 - 102/187) - d["sites"] * quant_us * 1e-3,
      "int8 + rotation": ins * (1 - 102/187 * ratios[s]) - d["sites"] * quant_us * 1e-3,
      "decode-free (rows at DRAM)": R - d["dram"],
    }
    print(f"{s}: Tetra {d['tetra']/1e9:.3f}e9 w, {d['blocks']/1e6:.1f}M blocks, rows {R:.2f} ms, DRAM {d['dram']:.2f} ms, "
          f"door {door:.3f} ms = {1000/door:.1f} tok/s, lane ratio {ratios[s]:.3f}")
    for k, v in sv.items():
        lo = 0.41 * v if "decode-free" not in k else v
        print(f"   {k:28s} save {v:.3f} ms (low {lo:.3f}) -> {door_after(v):6.1f} tok/s (low {door_after(lo):6.1f})"
              f"  +{(door/(door-v)-1)*100:4.1f} %")

# --- issue balance
for card, bw, sm, ghz in (("L40S", 864e9, 142, 2.52e9), ("RTX 5090", 1792e9, 170, 2.407e9)):
    bpc = bw / (sm * ghz)
    blk = bpc / 6.45
    print(f"{card}: {bpc:.3f} B/SM-clk, {blk:.4f} blocks/clk/SM at DRAM;"
          f" 237 -> {237*blk:.0f} lane-instr/clk ({237*blk/128*100:.0f} % of issue);"
          f" 222 -> {222*blk/128*100:.0f} %; 152 -> {152*blk:.0f} ({152*blk/128*100:.0f} %)")

# --- batch
W_t, W_a = 1.38, 2.67; BW = 864e9
F = 2 * (3.63e9 + 151936*2560)
for tc_name, tc in (("f16 362", 362e12), ("int8 733 x 0.75", 733e12*0.75)):
    per = F / tc * 1e3
    print(f"{tc_name}: MMA {per*1e3:.1f} us/seq; B* Tetra {W_t/864*1e3/ (per*1e3) * 1e3:.0f}, int4 {W_a/864*1e3/(per*1e3)*1e3:.0f}")
print(f"max format edge a step: {(W_a-W_t)/864*1e3:.2f} ms")
act = (3.63e9 + 151936*2560)
print(f"act reload f32 M=64: {act*4/64/1e9:.3f} GB/seq-step; f16 M=256: {act*2/256/1e6:.0f} MB; int8 M=256 {act/256/1e6:.0f} MB")
tet = {32: 1000*32/2102, 128: 1000*128/3997, 1: 1000/107}
awq = {32: 1000*32/5134, 128: 1000*128/10146, 1: 1000/203}
st = (tet[128]-tet[32])/96; sa = (awq[128]-awq[32])/96
print(f"steps Tetra {tet}, AWQ {awq}; slopes {st:.3f} / {sa:.3f} ms; excess {st-sa:.3f}; L2 BW to explain {act*4/64/((st-sa)*1e-3)/1e12:.2f} TB/s")
fixed = tet[32] - 32*st
for fx in (fixed, 5.0):
    for B in (32, 128):
        t = fx + B*0.08
        print(f"  fixed {fx:.2f} slope 0.08: B={B} step {t:.2f} ms -> {B*1000/t:.0f} tok/s, {W_t*1000/t:.0f} GB/s (whole model a step)")
# decode once a step in batch
dec = sizes["4B"]["blocks"]*152/32/(142*4*2.52e9*0.53)*1e3
print(f"decode once a step at 152 instr, 53 % issue: {dec:.2f} ms vs weights {W_t/864*1e3:.2f} ms")
# KV per seq at 256 positions
kv = 36*2*1024*2*256
print(f"KV read a seq a step at 256 pos: {kv/1e6:.1f} MB -> {kv/864e9*1e3:.3f} ms")

# --- part 4: activation rounding error. Tetra keeps 88.89 % of the signal on Gaussian blocks
# (how-it-works section 7), a relative weight error of sqrt(0.1111) = 0.333.
import numpy as np
rng = np.random.default_rng(0)
d, rows = 2560, 2048
def hadamard_rot(x):
    # randomized Walsh-Hadamard on 2048 coordinates + identity on the rest: enough to spread outliers
    n = 2048
    v = x[:, :n] * rng.choice([-1, 1], n)
    h = 1
    while h < n:
        v = v.reshape(v.shape[0], -1, 2 * h)
        a, b = v[:, :, :h].copy(), v[:, :, h:].copy()
        v[:, :, :h], v[:, :, h:] = a + b, a - b
        v = v.reshape(v.shape[0], n); h *= 2
    return np.concatenate([v / np.sqrt(n), x[:, n:]], axis=1)
def q8_block(x, blk=24):
    nb = x.shape[1] // blk
    xb = x[:, :nb * blk].reshape(x.shape[0], nb, blk)
    s = np.abs(xb).max(axis=2, keepdims=True) / 127.0
    s[s == 0] = 1
    q = np.clip(np.rint(xb / s), -127, 127) * s
    return np.concatenate([q.reshape(x.shape[0], -1), x[:, nb * blk:]], axis=1)
def q8_tensor(x):
    s = np.abs(x).max(axis=1, keepdims=True) / 127.0
    return np.clip(np.rint(x / s), -127, 127) * s
W = rng.standard_normal((rows, d)).astype(np.float64)
for name, gen in [("gauss", lambda: rng.standard_normal((64, d))),
                  ("outliers x100 on 8 ch", lambda: (lambda x: (x.__setitem__((slice(None), slice(0, 8)), x[:, :8] * 100), x)[1])(rng.standard_normal((64, d))))]:
    x = hadamard_rot(gen())
    y = x @ W.T
    for qn, qf in [("f16", lambda v: v.astype(np.float16).astype(np.float64)),
                   ("int8 per 24-block", q8_block), ("int8 per token", q8_tensor)]:
        e = (qf(x) @ W.T) - y
        print(f"{name:24s} {qn:20s} rel rms err of y = {np.sqrt((e**2).mean() / (y**2).mean()):.2e}")
print(f"int8 per 24-block noise power over the weight error's: {(5.09e-3 / 0.333) ** 2:.1e}")
