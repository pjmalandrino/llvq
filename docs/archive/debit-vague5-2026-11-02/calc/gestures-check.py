# Verification arithmetic for the revised gestures files (wave 5). Pure computation, no I/O beyond print.
import math, random, struct

def bitrev10(x):
    r = 0
    for b in range(10):
        r = (r << 1) | ((x >> b) & 1)
    return r

def f16bits(v):
    return struct.unpack('<H', struct.pack('<e', v))[0]

def f16val(b):
    return struct.unpack('<e', struct.pack('<H', b))[0]

def candle(vals):
    B = 1024; n = len(vals)
    shr = [float('-inf')] * B; idx = [0xFFFFFFFF] * B
    for t in range(B):
        i = t; ns = True
        while i < n:
            if ns or vals[i] > shr[t]:
                shr[t] = vals[i]; idx[t] = i; ns = False
            i += B
    s = B // 2
    while s > 0:
        for t in range(s):
            if shr[t + s] > shr[t]:
                shr[t] = shr[t + s]; idx[t] = idx[t + s]
        s //= 2
    return idx[0]

def ord16(b, m1=False):
    if not m1 and b == 0x8000:
        b = 0
    return ((~b) & 0xFFFF) if (b & 0x8000) else (b | 0x8000)

def key_argmax(bits, m1=False):
    best = None; bi = None
    for i, b in enumerate(bits):
        k = (ord16(b, m1) << 32) | ((1023 - bitrev10(i & 1023)) << 22) | (0x3FFFFF - (i >> 10))
        if best is None or k > best:
            best = k; bi = i
    return bi

N = 151936
print('bitrev10(300) =', bitrev10(300), ' bitrev10(276) =', bitrev10(276), ' (1300 is thread', 1300 - 1024, ')')
# Old case: -0 at 300, +0 at 1300 (thread 276), background -2.0
base = [f16bits(-2.0)] * N
old = list(base); old[300] = 0x8000; old[1300] = 0x0000
new = list(base); new[1300] = 0x8000; new[300] = 0x0000
for name, bits in [('old (-0@300, +0@1300)', old), ('new (-0@1300, +0@300)', new)]:
    vals = [f16val(b) for b in bits]
    c = candle(vals); k = key_argmax(bits); km1 = key_argmax(bits, m1=True)
    print(f'{name}: candle {c}, key {k}, m1-key {km1}, m1 killed: {km1 != c}')
# Probability thread 0 holds the max in the old random test
p_no = (7/8) ** 149
print('P(thread 0 has no 2.25 among its 149 elements) = %.3e' % p_no)
# Sparse cases: max at 2-6 random positions, background from small levels incl. +-0
random.seed(7)
levels = [f16bits(v) for v in (-3.0, -0.0, 0.0, 1.5)]
mism = 0; m1k = 0
for case in range(12):
    bits = [random.choice(levels) for _ in range(N)]
    top = f16bits(2.25)
    for j in random.sample(range(N), random.randint(2, 6)):
        bits[j] = top
    vals = [f16val(b) for b in bits]
    c = candle(vals); k = key_argmax(bits)
    mism += (c != k)
print('sparse cases checked: 12, key mismatches:', mism)
# Cost arithmetic
print('A2 eager 4B step ms = %.2f; today gqa-ab grouped step ms = %.2f; ratio %.3f' % (1000/99.2, 1000/134.4, (1000/134.4)/(1000/99.2)))
print('events trim ms = %.4f' % (1000/121.0 - 1000/122.4))
print('cast per launch in situ us = %.2f' % ((1000/119.6 - 1000/122.4) * 1000 / 84))
print('gsm8k 4b expanded: 71 min at 1.80/h = $%.2f' % (71/60*1.80))
print('tracking per-launch delta (a3-graph) us = %.2f' % (3.67 - 3.63))
for nargs in (1, 2, 3, 4):
    print('  929 launches x %d slice args x 0.04 us = %.3f ms' % (nargs, 929 * nargs * 0.04 / 1000))
print('fallback total pct at 7.44 ms: %.2f-%.2f' % (1.15 + 929*1*0.04/1000/7.44*100, 1.15 + 929*4*0.04/1000/7.44*100))
