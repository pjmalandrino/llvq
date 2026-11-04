"""DRAFT, never run. Written read-only on 2026-11-02 for the persistent-kernel
lead (wave5-deep/persistent.md, sections 3.9 and 5, step L0).

Host replay of candle 0.9.2's fast_argmax (candle-kernels reduce.cu:476-523,
one block of 1,024 threads as candle-core cuda_backend/mod.rs:324-331 launches
it for a 151,936-wide row) against the 64-bit key of exact_ops.cuh, on
adversarial tie patterns. The repository version of this check belongs in a
Rust host test; this file only pins the algorithm. Values are f16 bit patterns
compared through their exact float value, as candle compares the f32 widening.
"""
import random
import struct

BLOCK = 1024


def h2f(h):
    return struct.unpack("<e", struct.pack("<H", h))[0]


def candle_fast_argmax(bits):
    n = len(bits)
    shr = [float("-inf")] * BLOCK
    idx = [0xFFFFFFFF] * BLOCK
    for t in range(BLOCK):
        not_set = True
        i = t
        while i < n:
            v = h2f(bits[i])
            if not_set or v > shr[t]:  # strict >, the first element is taken unconditionally
                shr[t] = v
                idx[t] = i
                not_set = False
            i += BLOCK
    s = BLOCK // 2
    while s > 0:
        for t in range(s):
            if shr[t + s] > shr[t]:  # the left operand keeps ties
                shr[t] = shr[t + s]
                idx[t] = idx[t + s]
        s //= 2
    return idx[0]


def bitrev10(t):
    return int(f"{t:010b}"[::-1], 2)


def key(h, i):
    if (h & 0x7C00) == 0x7C00 and (h & 0x03FF) != 0:
        raise ValueError("NaN logit: the device path refuses and falls back")
    if h == 0x8000:
        h = 0
    o = (~h & 0xFFFF) if (h & 0x8000) else (h | 0x8000)
    return (o << 32) | ((1023 - bitrev10(i & 1023)) << 22) | (0x3FFFFF - (i >> 10))


def key_argmax(bits):
    best = max(range(len(bits)), key=lambda i: key(bits[i], i))
    return best


def adversarial(n, rng):
    base = [rng.choice([0x3C00, 0xBC00, 0x0000, 0x8000, 0x4500]) for _ in range(n)]
    top = 0x5A00  # the planted maximum
    k = rng.choice([2, 3, 5, 16, 64])
    for _ in range(k):  # ties placed to exercise thread order and in-thread order
        t = rng.choice([0, 1, 2, 256, 512, 768, 1023, rng.randrange(1024)])
        lap = rng.randrange(max(1, n // BLOCK))
        i = min(n - 1, lap * BLOCK + t)
        base[i] = top
    if rng.random() < 0.3:  # a negative-zero / positive-zero tie at the top
        base = [0x8000 if b == top else b for b in base]
        base[rng.randrange(n)] = 0x0000
    return base


if __name__ == "__main__":
    rng = random.Random(0x5EED)
    n = 151936
    bad = 0
    for case in range(300):
        b = adversarial(n, rng)
        if candle_fast_argmax(b) != key_argmax(b):
            bad += 1
    print(f"{bad} mismatches over 300 adversarial cases")
