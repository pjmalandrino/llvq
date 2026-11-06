# Arithmetic check for wave5-deep/brick.md §1 and §8 item 9, and prereg-brick-1.md §1 and §6.
#
# Not repository code. A tiny numpy script, run by the author with
#   nice -n 10 python3 -I synthetic_actq.py > synthetic_actq.out
# on 2026-11-02. No file or network I/O besides stdout, no repository access.
# It replaces the 6-vector figures of the first draft (no script was saved then).
#
# What it computes, per case and over N vectors per case:
#   relative rms error ||q(x) - x|| / ||x|| of
#     f16   : rounding the activation to f16 (what the served path already carries upstream)
#     i16s  : the brick quantizer, int16 per 384-coordinate slice, power-of-two exponent
#             e = floor(log2(amax)) - 14, spread capped at 3 binades below the token's top slice
#     i16u  : the same without the cap
#     i16v  : int16 per vector, power-of-two exponent
#     i24s  : int24 per slice, cap 3
#     i8v   : int8 per vector, step amax/127 (ExLlamaV3's plain mode)
#   the noise-power share (i16s / f16)^2, the exponent spread across slices and how often the
#   cap binds (spread > 3), and slice amax/rms.
# Cases (block coordinates only; the f16 tail columns are excluded as in the kernel):
#   gauss  : iid N(0, 1), the near-Gaussian picture a good rotation gives
#   t4, t3 : iid Student t with 4 and 3 degrees of freedom, a rotation that failed to Gaussianize
#   spikeR : a natural-basis Gaussian vector plus ONE channel carrying a share R of the energy
#            (a massive activation), rotated by the repository's construction
#            Q = (Q_odd (x) H_m) D (llvq-quant/src/rotation.rs), a fresh Q per vector;
#            f16 rounding is applied in the natural basis, where the activation is stored.
# Widths: 2,544 and 9,720 block coordinates (d_in 2,560 and 9,728 of the 4B, 106 and 405 blocks).
import sys
import numpy as np

SEED, N, CHUNK, SLICE = 20261102, 2000, 200, 384
rng = np.random.default_rng(SEED)


def exps(a):
    """floor(log2(a)) for a > 0, exactly, from frexp; -inf where a == 0."""
    m, e = np.frexp(a)
    out = (e - 1).astype(np.float64)
    out[a == 0] = -np.inf
    return out


def q_slices(x, bits, cap):
    """x: (B, n) float32 values held in float64. Power-of-two exponent per slice."""
    B, n = x.shape
    ns = -(-n // SLICE)
    pad = np.zeros((B, ns * SLICE))
    pad[:, :n] = x
    s = pad.reshape(B, ns, SLICE)
    amax = np.abs(s).max(axis=2)
    e = exps(amax) - (bits - 2)
    e_hi = e.max(axis=1, keepdims=True)
    finite = np.where(np.isfinite(e), e, np.nan)
    spread = (e_hi[:, 0] - np.nanmin(finite, axis=1))
    if cap is None:
        es = np.where(np.isfinite(e), e, e_hi)
    else:
        es = np.maximum(np.where(np.isfinite(e), e, -np.inf), e_hi - cap)
    step = np.exp2(es)[:, :, None]
    qmax = 2.0 ** (bits - 1) - 1
    q = np.clip(np.rint(s / step), -qmax, qmax) * step
    return q.reshape(B, ns * SLICE)[:, :n], spread, amax / np.sqrt((s * s).mean(axis=2))


def q_vec_pow2(x, bits):
    amax = np.abs(x).max(axis=1, keepdims=True)
    step = np.exp2(exps(amax) - (bits - 2))
    qmax = 2.0 ** (bits - 1) - 1
    return np.clip(np.rint(x / step), -qmax, qmax) * step


def q_int8_amax127(x):
    step = np.abs(x).max(axis=1, keepdims=True) / 127.0
    return np.clip(np.rint(x / step), -127, 127) * step


def rel(q, x):
    return np.linalg.norm(q - x, axis=1) / np.linalg.norm(x, axis=1)


def fwht(v):
    """Orthonormal Walsh-Hadamard along the last axis (length a power of two)."""
    v = v.copy()
    m = v.shape[-1]
    h = 1
    while h < m:
        v = v.reshape(*v.shape[:-1], m // (2 * h), 2, h)
        a, b = v[..., 0, :].copy(), v[..., 1, :].copy()
        v[..., 0, :], v[..., 1, :] = a + b, a - b
        v = v.reshape(*v.shape[:-3], m)
        h *= 2
    return v / np.sqrt(m)


def rotate(xn, d_full):
    """Q = (Q_odd (x) H_m) D on rows of xn (B, d_full), a fresh Q per row."""
    B = xn.shape[0]
    m = d_full & -d_full
    k = d_full // m
    D = rng.choice([-1.0, 1.0], size=(B, d_full))
    v = (xn * D).reshape(B, k, m)
    v = fwht(v)                                   # I_k (x) H_m
    G = rng.standard_normal((B, k, k))
    Qo, R = np.linalg.qr(G)
    Qo = Qo * np.sign(np.diagonal(R, axis1=1, axis2=2))[:, None, :]
    v = np.einsum('bij,bjm->bim', Qo, v)          # Q_odd (x) I_m
    return v.reshape(B, d_full)


def gen(case, B, d_full, nb):
    if case == 'gauss':
        x = rng.standard_normal((B, nb))
        return x.astype(np.float32).astype(np.float64), None
    if case in ('t4', 't3'):
        nu = 4 if case == 't4' else 3
        x = rng.standard_t(nu, size=(B, nb))
        return x.astype(np.float32).astype(np.float64), None
    share = {'spike50': 0.5, 'spike90': 0.9}[case]
    xn = rng.standard_normal((B, d_full))
    idx = rng.integers(0, d_full, size=B)
    rest = (xn * xn).sum(axis=1)
    xn[np.arange(B), idx] = np.sqrt(share / (1 - share) * rest)
    xn = xn.astype(np.float32).astype(np.float64)          # stored activation, f32
    f16n = rel(xn.astype(np.float16).astype(np.float64), xn)  # rounding where it is stored
    xr = rotate(xn, d_full)[:, :nb].astype(np.float32).astype(np.float64)
    return xr, f16n


def pct(v, p):
    return float(np.percentile(v, p))


print(f"numpy {np.__version__}, seed {SEED}, {N} vectors per case, slice {SLICE}")
for case in ('gauss', 't4', 't3', 'spike50', 'spike90'):
    for d_full, nb in ((2560, 2544), (9728, 9720)):
        acc = {k: [] for k in ('f16', 'i16s', 'i16u', 'i16v', 'i24s', 'i8v', 'spread', 'ratio')}
        sl = []
        for c in range(N // CHUNK):
            x, f16n = gen(case, CHUNK, d_full, nb)
            f16 = rel(x.astype(np.float16).astype(np.float64), x) if f16n is None else f16n
            q16, spread, slr = q_slices(x, 16, 3)
            i16s = rel(q16, x)
            acc['f16'].append(f16)
            acc['i16s'].append(i16s)
            acc['i16u'].append(rel(q_slices(x, 16, None)[0], x))
            acc['i16v'].append(rel(q_vec_pow2(x, 16), x))
            acc['i24s'].append(rel(q_slices(x, 24, 3)[0], x))
            acc['i8v'].append(rel(q_int8_amax127(x), x))
            acc['spread'].append(spread)
            acc['ratio'].append((i16s / f16) ** 2)
            sl.append(slr.ravel())
        a = {k: np.concatenate(v) for k, v in acc.items()}
        sl = np.concatenate(sl)
        bind = float((a['spread'] > 3).mean())
        print(f"\n{case:8s} n={nb:5d}")
        for k in ('f16', 'i16s', 'i16u', 'i16v', 'i24s', 'i8v'):
            print(f"  {k:5s} rel rms  p50 {pct(a[k], 50):.3e}  p99 {pct(a[k], 99):.3e}  max {a[k].max():.3e}")
        print(f"  (i16s/f16)^2 noise-power share  p50 {pct(a['ratio'], 50):.3f}  p99 {pct(a['ratio'], 99):.3f}  max {a['ratio'].max():.3f}")
        print(f"  exponent spread across slices    p50 {pct(a['spread'], 50):.0f}  p99 {pct(a['spread'], 99):.0f}  max {a['spread'].max():.0f}"
              f"   cap binds in {100 * bind:.1f} % of vectors")
        print(f"  i16s over i16u, p50 {pct(a['i16s'] / a['i16u'], 50):.4f}  max {(a['i16s'] / a['i16u']).max():.4f}")
        print(f"  slice amax/rms  p50 {pct(sl, 50):.2f}  p99 {pct(sl, 99):.2f}  max {sl.max():.2f}")
sys.stdout.flush()
