// The incoherence rotation, applied to the activation.
//
// ## Why this kernel has to exist at all
//
// The sealed artifact stores weights in a *rotated* basis: the quantizer
// computed `W' = W Qᵀ` and `format.rs::decode_matrix` un-rotates on the way
// out, which is why the dense path needs nothing extra. The fused matvec does
// not un-rotate — it reads `W'` as stored — so the identity it computes is
//
//     y = W x = (W Qᵀ)(Q x) = W' · rot(x)
//
// and `rot` is this file. Without it the fused kernel returns a correct
// product of the wrong two things, silently: every row is finite, plausible,
// and off. That failure mode is the reason this arrives with an executable
// test rather than a shape assertion.
//
// ## Construction, mirroring `llvq-quant/src/rotation.rs`
//
//     Q = (Q_odd ⊗ H_m) D     n = k·m, m the largest power of two dividing n
//
// so `rot(x)` is three passes: flip signs by `D`, a Walsh–Hadamard transform
// over each of the `k` contiguous groups of `m`, then `Q_odd` applied across
// the groups at each of the `m` positions. The Rust reference is the
// specification; `tests/rotation_matches_rust.rs` diffs this text against it.
//
// ## Why the phases are separate functions
//
// Every one of them takes `(tid, nthreads)` instead of reading `threadIdx`.
// That is what lets the host test *execute* them — the driver runs the kernel
// at `blockDim.x == 1`, where barriers are vacuous and the arithmetic is
// identical, then replays each phase at several thread counts to prove the
// work split does not change the answer. A separate host reimplementation
// would prove nothing: it could share a bug with the kernel, or drift from it.

#ifndef LLVQ_ROT_CUH
#define LLVQ_ROT_CUH

#ifndef LLVQ_SLOT_CUH
typedef unsigned int u32;
#endif

// Widest odd factor the mix handles.
//
// `col` is a per-thread array indexed by a fully unrolled loop, which is what
// keeps it in registers; a runtime bound would put it in local memory and the
// register report would show the spill. 32 covers every Qwen3-4B and 8B width
// (k = 1, 3, 5, 19). Qwen3-32B's `down_proj` is 25600 = 512 · 50 and does
// **not** fit — the host refuses it rather than the kernel truncating it.
#define LLVQ_ROT_KMAX 32u

// f16 → f32, exactly.
//
// Duplicated from `matvec.cu` rather than shared, for the same reason
// `f16_bits` is duplicated between `llvq-metal` and `llvq-cuda`: the two are
// separate NVRTC translation units and the shared home is lot K0.
//
// The host branch differs from `matvec.cu`'s, deliberately. There, `h2f`
// returns 0 under `LLVQ_HOST_BUILD` because that kernel is compile-only and a
// software conversion would flatter a *cost* baseline. Here the host branch
// must be right, because this kernel is executed on the host — and it can be:
// widening binary16 to binary32 is exact, so a correct software conversion is
// bit-identical to `cvt.f32.f16`. There is nothing to approximate and nothing
// to flatter.
__device__ __forceinline__ float rot_h2f(unsigned short h)
{
#ifdef LLVQ_HOST_BUILD
    u32 sign = (u32)(h & 0x8000u) << 16;
    u32 exp = (h >> 10) & 0x1fu;
    u32 man = h & 0x3ffu;
    u32 bits;
    if (exp == 0u) {
        if (man == 0u) {
            bits = sign;  // ±0
        } else {
            // Subnormal binary16 is normal in binary32: shift the mantissa up
            // until its leading 1 falls off, and charge the exponent for it.
            u32 e = 127u - 15u + 1u;
            while ((man & 0x400u) == 0u) {
                man <<= 1;
                --e;
            }
            bits = sign | (e << 23) | ((man & 0x3ffu) << 13);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7f800000u | (man << 13);  // inf / NaN, payload kept
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (man << 13);
    }
    float r;
    __builtin_memcpy(&r, &bits, sizeof r);
    return r;
#else
    float r;
    asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(h));
    return r;
#endif
}

// Phase 1 — widen, flip, stage.
//
// The signs are a bitmap, not `n` floats: one `u32` per 32 coordinates costs
// 3 % of the vector's own bytes instead of 100 %, and the whole table for a
// 36-block model then fits in a few hundred kilobytes of L2 rather than
// evicting the weights this kernel exists to make room for.
//
// Bit `i & 31` of word `i >> 5` set means **negative**, matching the host,
// which packs `signs[i] < 0.0`.
//
// `x_off` is the activation's start *in elements*, not a courtesy: inside an
// inference runtime the tensor handed over is a view into a larger buffer, and
// `CudaSlice::slice` yields a `CudaView` that the launch signature would have
// had to special-case. One add per thread, and the caller passes 0 whenever it
// owns the whole buffer.
__device__ __forceinline__ void rot_load(const unsigned short* __restrict__ xin,
                                         const u32* __restrict__ signbits,
                                         float* s,
                                         u32 n,
                                         u32 x_off,
                                         u32 tid,
                                         u32 nthreads)
{
    for (u32 i = tid; i < n; i += nthreads) {
        float v = rot_h2f(xin[x_off + i]);
        s[i] = ((signbits[i >> 5] >> (i & 31u)) & 1u) ? -v : v;
    }
}

// Phase 2 — one butterfly stage of the Walsh–Hadamard transform, over all `k`
// groups at once.
//
// The Rust reference walks `len = 1, 2, 4, …` and, within a stage, pairs
// `(j, j+len)` for every `j` whose `len`-bit is clear. Here the same stage is
// addressed by pair ordinal `p ∈ [0, n/2)`:
//
//     j = (p/len)·2len + (p mod len)
//
// **The groups need no separate term.** The obvious form carries one — take
// `g = p/(m/2)`, the pair `r = p − g·(m/2)` within it, and offset by `g·m` —
// and it is exactly equal to the line above, because `len ≤ m/2` and both are
// powers of two, so `m/2` is a multiple of `len` and the group offset
// distributes out of the division:
//
//     (g·(m/2) + r)/len · 2len + r mod len  =  g·m + (r/len)·2len + r mod len
//
// So the groups are traversed correctly by an addressing scheme that does not
// know they exist, and `m` is not an argument here at all. Found by mutation:
// pinning `g` to zero changed nothing any test could see, which per §5 of
// CLAUDE.md means either the test is weak or the code is dead. It was dead —
// verified algebraically above, and `M6` (flat addressing, i.e. dropping the
// `len` decomposition too) is still killed, so the surviving line is load-
// bearing.
//
// Every pair touches two disjoint slots, so the stage is race-free for any
// thread count — that independence is what the multi-width replay in the test
// checks.
//
// Not scaled here: the `1/√m` the orthogonal transform needs is folded into
// the last phase, so the vector is touched once less.
__device__ __forceinline__ void rot_wht_step(float* s, u32 n, u32 len, u32 tid, u32 nthreads)
{
    u32 npairs = n >> 1;
    for (u32 p = tid; p < npairs; p += nthreads) {
        u32 j = (p / len) * (len << 1) + (p % len);
        float a = s[j];
        float b = s[j + len];
        s[j] = a + b;
        s[j + len] = a - b;
    }
}

// Phase 3a — `k == 1`: there is no odd factor, so the transform is done and
// only the scale is owed.
//
// A separate path rather than a `Q_odd = [1]` special case of the mix: the
// branch is uniform across the block (`k` is a kernel argument), so it costs
// nothing, and it saves `KMAX²` fully-unrolled multiply-adds per position on
// the widest power-of-two layers — `o_proj` at 4096 is exactly that shape.
__device__ __forceinline__ void rot_scale_out(const float* s,
                                              float* __restrict__ xout,
                                              u32 n,
                                              float inv,
                                              u32 tid,
                                              u32 nthreads)
{
    for (u32 i = tid; i < n; i += nthreads) xout[i] = s[i] * inv;
}

// Phase 3b — `Q_odd` across the groups, plus the scale, plus the write.
//
// Thread `j` owns column `j`: the `k` values at `s[t·m + j]` are read by no
// other thread and written by no other thread, so this phase needs no barrier
// of its own and can write its results back over its own inputs. That is why
// `col` exists — the `g`-th output overwrites an input the `(g+1)`-th still
// needs, so the column is lifted into registers first.
//
// Both inner loops run to `LLVQ_ROT_KMAX`, not to `k`: a compile-time bound is
// what keeps `col` in registers, and `small` is zero-padded by the host so the
// extra terms contribute nothing. The waste is real and bounded — `KMAX²`
// multiply-adds per position against `k²` — and it measured too small to
// justify a templated kernel per width.
//
// **Which of the two padded operands is load-bearing.** Mutation says: the
// zeros in `small`, not the zeros in `col`. Setting the `t >= k` slots of
// `col` to 1.0f instead of 0.0f changes no result and no test, because
// `small[g·KMAX + t]` is already zero there — a surviving mutant, and an
// equivalent one. `col`'s initializer is therefore not a correctness guard
// but a definedness guard: reading an uninitialized automatic is undefined,
// and a register holding whatever the previous thread left there is exactly
// the kind of nondeterminism that reproduces once in ten runs. The host
// contract is that `small` is zero-padded, and `tests/rotation_matches_rust.rs`
// pads it; a host that forgot would produce garbage no assertion here catches.
__device__ __forceinline__ void rot_mix(const float* s,
                                        const float* __restrict__ small,
                                        float* __restrict__ xout,
                                        u32 m,
                                        u32 k,
                                        float inv,
                                        u32 tid,
                                        u32 nthreads)
{
    for (u32 j = tid; j < m; j += nthreads) {
        float col[LLVQ_ROT_KMAX];
#pragma unroll
        for (u32 t = 0; t < LLVQ_ROT_KMAX; ++t) col[t] = t < k ? s[t * m + j] * inv : 0.0f;
        for (u32 g = 0; g < k; ++g) {
            float acc = 0.0f;
#pragma unroll
            for (u32 t = 0; t < LLVQ_ROT_KMAX; ++t)
                acc = __fmaf_rn(small[g * LLVQ_ROT_KMAX + t], col[t], acc);
            xout[g * m + j] = acc;
        }
    }
}

#endif  // LLVQ_ROT_CUH
