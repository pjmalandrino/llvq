// The incoherence rotation in Metal Shading Language.
//
// A port of `llvq-cuda/kernels/llvq_rot.cuh` and `llvq-cuda/kernels/rotate.cu`.
// The arithmetic is the same and is meant to be BIT-IDENTICAL:
// `llvq-metal/tests/rot_matches_host.rs` compiles that CUDA text with
// `clang++` through `llvq-cuda/tests/host_rotate.cpp`, runs both, and demands
// equal f32. A rotation that is merely close is a wrong basis.
//
// ## Why this kernel has to exist at all
//
// The sealed artifact stores weights in a rotated basis. The quantizer
// computed `W' = W Qᵀ`, so the fused matvec computes
//
//     y = W x = (W Qᵀ)(Q x) = W' · rot(x)
//
// and `rot` is this file. Without it the fused path returns a correct product
// of the wrong two things. Every row stays finite and plausible and wrong.
// That failure mode is why this arrives with an executable gate.
//
// ## Construction, mirroring `llvq-quant/src/rotation.rs`
//
//     Q = (Q_odd ⊗ H_m) D     n = k·m, m the largest power of two dividing n
//
// So `rot(x)` is three passes. Flip signs by `D`. Run a Walsh-Hadamard
// transform over each of the `k` contiguous groups of `m`. Apply `Q_odd`
// across the groups at each of the `m` positions.
//
// ## The one structural change, and the measurement that forced it
//
// The staging area is a `device` buffer here, where CUDA uses dynamic shared
// memory. An M3 Max offers 32,768 B of threadgroup memory (*measured*,
// `MTLDevice.maxThreadgroupMemoryLength`). Qwen3-4B's `down_proj` input is
// 9728 wide, which needs 38,912 B. It does not fit, and that is the widest
// projection in the served object, so a threadgroup-staged kernel could not
// serve the model at all.
//
// The consequence is honest and bounded. Every Walsh-Hadamard stage reads and
// writes device memory instead of threadgroup memory. The vector is 39 KB and
// the matvec it feeds moves gigabytes, so the traffic is small. Whether the
// Apple cache hierarchy makes it free is a measurement nobody has taken.
//
// A threadgroup-staged variant is admissible for `n <= 8192`, which covers
// five of the six projection widths. It is a second code path with its own
// gate, so it is a later lot and not a guess made here.
//
// ## Why one threadgroup does a whole vector
//
// The Walsh-Hadamard transform is `log₂ m` stages separated by barriers, and
// Metal has no barrier across threadgroups. Splitting one vector would need a
// second dispatch per rotation site, on a decode already dominated by launch
// latency. The rows of `rot_apply_rows_metal` are independent, so the grid
// carries them and each threadgroup still owns a whole row.
//
// ## What the host owes these kernels
//
//   * `n = k · m`, `m` a power of two, `k <= LLVQ_ROT_KMAX`
//   * `small` zero-padded to `LLVQ_ROT_KMAX × LLVQ_ROT_KMAX`, row-major
//   * `inv = 1/sqrt(m)`, computed in f64 and narrowed once on the host.
//     Recomputing it here would put `rsqrt` or `sqrt` between the two sides
//     and make a last-bit difference unattributable.
//   * `scratch`, at least `n` floats for `rot_apply_metal` and `n_rows · n`
//     for `rot_apply_rows_metal`
//   * a `small` buffer of at least one element even when `k == 1`. Metal
//     refuses a zero-length buffer and hands back a null pointer, the same
//     wall cudarc puts up. The `k == 1` branch never reads it, so a
//     one-element dummy is enough.
//
// There is deliberately no bounds guard on `n`. An out-of-range `n` would
// overrun `scratch`, so it is a host-side assertion checked once per matrix at
// load time rather than once per token by every thread.
//
// ## The store is f32, and that is a decision
//
// `xout` is `device float*`. Narrowing to f16 is a separate lot with its own
// gate, because `f2h` is round-to-nearest-even and an untested one would be a
// new defect. `tv_tetra48_metal` made the same choice for the same reason.

#include <metal_stdlib>
using namespace metal;

// Every multiply-add in this file is EXACTLY what is written.
//
// Metal is clang, and clang contracts `a * b + c` into an `fma` unless told
// not to. `fma` rounds once where the written form rounds twice, so the two
// are different numbers. The gate here is equality against the CUDA original,
// which spells out `__fmaf_rn` at the one site that fuses and leaves the rest
// as plain operators. This pragma is what lets the MSL say the same thing.
//
// Turning Metal's fast math off is necessary and NOT sufficient: it stops the
// reassociation and leaves the contraction. This stops the contraction.
#pragma clang fp contract(off)

// Widest odd factor the mix handles.
//
// `col` is a per-thread array indexed by a fully unrolled loop, which is what
// keeps it in registers. A runtime bound would push it to thread-local memory.
// 32 covers every Qwen3-4B and 8B width (k = 1, 3, 5, 19). Qwen3-32B's
// `down_proj` is 25600 = 512 · 50 and does NOT fit. The host refuses it rather
// than the kernel truncating it.
//
// `rot_matches_host.rs` reads this line and the CUDA one and requires the same
// literal. The host pads `small` to this side, so a drift would read past the
// block it uploaded.
#define LLVQ_ROT_KMAX 32u

/// f16 bits to f32, exactly.
///
/// The CUDA twin emits `cvt.f32.f16`. Widening binary16 to binary32 is exact,
/// so there is nothing to approximate and the two cannot disagree. The
/// argument is `ushort` rather than `half` to mirror the CUDA signature, which
/// lets the gate hand both sides the same bytes.
inline float rot_h2f(ushort h)
{
    return float(as_type<half>(h));
}

/// Phase 1. Widen, flip, stage.
///
/// The signs are a bitmap, not `n` floats. One `uint` per 32 coordinates costs
/// 3 % of the vector's own bytes instead of 100 %, so the whole table for a
/// 36-block model stays small enough to sit in cache beside the weights.
///
/// Bit `i & 31` of word `i >> 5` set means NEGATIVE, matching the host, which
/// packs `signs[i] < 0.0`.
///
/// `x_off` is the activation's start in elements. Inside an inference runtime
/// the tensor handed over is a view into a larger buffer, and a Metal buffer
/// binding carries its own offset the caller may already have spent. One add
/// per thread, and the caller passes 0 when it owns the whole buffer.
inline void rot_load(const device ushort* xin,
                     const device uint* signbits,
                     device float* s,
                     uint n,
                     uint x_off,
                     uint tid,
                     uint nthreads)
{
    for (uint i = tid; i < n; i += nthreads) {
        float v = rot_h2f(xin[x_off + i]);
        s[i] = ((signbits[i >> 5] >> (i & 31u)) & 1u) ? -v : v;
    }
}

/// Phase 2. One butterfly stage of the Walsh-Hadamard transform, over all `k`
/// groups at once.
///
/// The Rust reference walks `len = 1, 2, 4, ...` and pairs `(j, j+len)` for
/// every `j` whose `len`-bit is clear. Here the same stage is addressed by
/// pair ordinal `p` in `[0, n/2)`:
///
///     j = (p/len)·2len + (p mod len)
///
/// The groups need no separate term. The obvious form carries one: take
/// `g = p/(m/2)`, the pair `r = p − g·(m/2)` within it, and offset by `g·m`.
/// That is equal to the line above, because `len <= m/2` and both are powers
/// of two, so `m/2` is a multiple of `len` and the group offset distributes
/// out of the division:
///
///     (g·(m/2) + r)/len · 2len + r mod len  =  g·m + (r/len)·2len + r mod len
///
/// So `m` is not an argument here at all. The CUDA header records the same
/// algebra and the mutation run that found it.
///
/// Every pair touches two disjoint slots, so the stage is race-free at any
/// thread count. `the_work_split_does_not_move_a_bit` is what checks it.
///
/// Not scaled here. The `1/sqrt(m)` the orthogonal transform needs is folded
/// into the last phase, so the vector is touched once less.
inline void rot_wht_step(device float* s, uint n, uint len, uint tid, uint nthreads)
{
    uint npairs = n >> 1;
    for (uint p = tid; p < npairs; p += nthreads) {
        uint j = (p / len) * (len << 1) + (p % len);
        float a = s[j];
        float b = s[j + len];
        s[j] = a + b;
        s[j + len] = a - b;
    }
}

/// Phase 3a. `k == 1`, so there is no odd factor and only the scale is owed.
///
/// A separate path rather than a `Q_odd = [1]` special case of the mix. The
/// branch is uniform across the threadgroup, since `k` is a kernel argument,
/// so it costs nothing. It saves `KMAX²` fully unrolled multiply-adds per
/// position on the widest power-of-two layers. `o_proj` at 4096 is that shape.
inline void rot_scale_out(const device float* s,
                          device float* xout,
                          uint n,
                          float inv,
                          uint tid,
                          uint nthreads)
{
    for (uint i = tid; i < n; i += nthreads) xout[i] = s[i] * inv;
}

/// Phase 3b. `Q_odd` across the groups, plus the scale, plus the write.
///
/// Thread `j` owns column `j`. The `k` values at `s[t·m + j]` are read and
/// written by no other thread, so this phase needs no barrier of its own.
/// `col` exists because the `g`-th output would overwrite an input the
/// `(g+1)`-th still needs, so the column is lifted into registers first.
///
/// Both inner loops run to `LLVQ_ROT_KMAX`, not to `k`. A compile-time bound
/// is what keeps `col` in registers, and the host zero-pads `small` so the
/// extra terms contribute nothing. The waste is `KMAX²` multiply-adds per
/// position against `k²`, and the CUDA side measured it too small to justify a
/// kernel per width.
///
/// Which padded operand is load-bearing: the zeros in `small`, not the zeros
/// in `col`. Setting the `t >= k` slots of `col` to 1.0f changes no result,
/// because `small[g·KMAX + t]` is already zero there. `col`'s initializer is a
/// definedness guard, and `the_mix_ignores_whatever_pads_the_small_block` is
/// what turns the host contract into an assertion.
///
/// The read index is CLAMPED, where the CUDA writes `t < k ? s[t*m+j] : 0`.
/// CUDA's `s` is a shared-memory array, so a speculated load past `t = k` is
/// harmless. Here `s` is a `device` pointer with no bounds the compiler can
/// prove, so the same speculation would touch unmapped pages hundreds of
/// kilobytes past the buffer. The clamp reads a slot that always exists and
/// the select throws the value away, so the arithmetic is unchanged.
inline void rot_mix(const device float* s,
                    const device float* small,
                    device float* xout,
                    uint m,
                    uint k,
                    float inv,
                    uint tid,
                    uint nthreads)
{
    for (uint j = tid; j < m; j += nthreads) {
        float col[LLVQ_ROT_KMAX];
#pragma clang loop unroll(full)
        for (uint t = 0u; t < LLVQ_ROT_KMAX; ++t) {
            uint src = (t < k ? t : 0u) * m + j;
            col[t] = t < k ? s[src] * inv : 0.0f;
        }
        for (uint g = 0u; g < k; ++g) {
            float acc = 0.0f;
#pragma clang loop unroll(full)
            for (uint t = 0u; t < LLVQ_ROT_KMAX; ++t)
                acc = fma(small[g * LLVQ_ROT_KMAX + t], col[t], acc);
            xout[g * m + j] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// The entry points. `rot_apply_metal` is one activation, one threadgroup;
// `rot_apply_rows_metal` is `n_rows` of them, one threadgroup each.
// ---------------------------------------------------------------------------

/// `x' = Q x` for one activation.
///
/// The host dispatches exactly one threadgroup. Any thread count works, and
/// the answer does not depend on it.
///
/// The barriers carry `mem_device` because the staging area is a device
/// buffer. `threadgroup_barrier` orders device memory across the threads of
/// one threadgroup, which is the only scope this kernel shares anything in.
kernel void rot_apply_metal(const device ushort* xin      [[buffer(0)]],
                            const device uint*   signbits [[buffer(1)]],
                            const device float*  small    [[buffer(2)]],
                            device float*        xout     [[buffer(3)]],
                            device float*        scratch  [[buffer(4)]],
                            constant uint&       n        [[buffer(5)]],
                            constant uint&       m        [[buffer(6)]],
                            constant uint&       k        [[buffer(7)]],
                            constant float&      inv      [[buffer(8)]],
                            constant uint&       x_off    [[buffer(9)]],
                            uint tid      [[thread_position_in_threadgroup]],
                            uint nthreads [[threads_per_threadgroup]])
{
    rot_load(xin, signbits, scratch, n, x_off, tid, nthreads);
    threadgroup_barrier(mem_flags::mem_device);

    // `m` is uniform across the threadgroup, so every thread runs the same
    // number of stages and reaches every barrier. That is the condition Metal
    // requires, and the reason the loop bound cannot become per-thread.
    for (uint len = 1u; len < m; len <<= 1) {
        rot_wht_step(scratch, n, len, tid, nthreads);
        threadgroup_barrier(mem_flags::mem_device);
    }

    if (k == 1u) {
        rot_scale_out(scratch, xout, n, inv, tid, nthreads);
    } else {
        rot_mix(scratch, small, xout, m, k, inv, tid, nthreads);
    }
}

/// `X' = Q X` for `n_rows` activations, ONE dispatch.
///
/// The rows are independent, so nothing crosses a threadgroup and nothing
/// needs a barrier between them. One threadgroup does one whole row, barriers
/// and all, exactly the work `rot_apply_metal` does.
///
/// The output lands contiguous, `[n_rows, n]` row-major, which is the shape
/// the row matvec wants. The CUDA host used to build that shape with a
/// `Tensor::cat` per row, and this removes those copies rather than moving
/// them.
///
/// What the host owes on top of `rot_apply_metal`:
///
///   * one threadgroup per row, so `threads == n_rows · nthreads`. There is no
///     bounds guard, for the reason given about `n`.
///   * `xout` at least `n_rows · n` floats, `scratch` the same.
///   * `row_stride` in ELEMENTS, the distance from one input row to the next.
///     It is not `n`: the activation handed over is a view into a larger
///     buffer and its rows are `d_in` apart there. The two are equal in every
///     model this repository serves, so writing `n` would be a coincidence
///     rather than a definition.
kernel void rot_apply_rows_metal(const device ushort* xin        [[buffer(0)]],
                                 const device uint*   signbits   [[buffer(1)]],
                                 const device float*  small      [[buffer(2)]],
                                 device float*        xout       [[buffer(3)]],
                                 device float*        scratch    [[buffer(4)]],
                                 constant uint&       n          [[buffer(5)]],
                                 constant uint&       m          [[buffer(6)]],
                                 constant uint&       k          [[buffer(7)]],
                                 constant float&      inv        [[buffer(8)]],
                                 constant uint&       x_off      [[buffer(9)]],
                                 constant uint&       row_stride [[buffer(10)]],
                                 uint tid      [[thread_position_in_threadgroup]],
                                 uint nthreads [[threads_per_threadgroup]],
                                 uint r        [[threadgroup_position_in_grid]])
{
    // `ulong`, not `uint`. `r * n` fits in 32 bits at every width this model
    // family has, and a cast that is right by coincidence stops being right in
    // silence.
    device float* s = scratch + (ulong)r * (ulong)n;

    rot_load(xin, signbits, s, n, x_off + r * row_stride, tid, nthreads);
    threadgroup_barrier(mem_flags::mem_device);

    // Same stage loop, same uniformity argument. `r` is uniform too, since it
    // is the threadgroup index, so no barrier is conditional on anything
    // divergent.
    for (uint len = 1u; len < m; len <<= 1) {
        rot_wht_step(s, n, len, tid, nthreads);
        threadgroup_barrier(mem_flags::mem_device);
    }

    device float* out = xout + (ulong)r * (ulong)n;
    if (k == 1u) {
        rot_scale_out(s, out, n, inv, tid, nthreads);
    } else {
        rot_mix(s, small, out, m, k, inv, tid, nthreads);
    }
}


// ---------------------------------------------------------------------------
// The same rotation with the transform staged in THREADGROUP memory.
//
// ## What it buys, and where it stops
//
// `rot_apply_metal` keeps its working set in device memory because the
// largest served width, n = 9728, needs 38,912 B against Apple's 32,768. So
// every one of the log2(m) butterfly stages crosses global memory: for
// n = 2560 that is 11 stages of 10 KB read and written, 220 KB, to transform
// 10 KB of data.
//
// Staged in threadgroup memory it is 10 KB in and 10 KB out, once. Eleven
// times less traffic.
//
// ⚠️ ADMISSIBLE ONLY FOR n <= 8192. The host checks it and falls back. Three
// of the four rotations a layer are n = 2560 under `rot_share=1`, so 108 of
// the served 144 launches a token take this path and 36 do not.
//
// The threadgroup barrier is `mem_threadgroup` here, not `mem_device`: what
// is being ordered is threadgroup memory. One threadgroup owns the whole
// transform, which is what makes a barrier sufficient at all.
// ---------------------------------------------------------------------------

inline void rot_mix_tg(const threadgroup float* s,
                    const device float* small,
                    device float* xout,
                    uint m,
                    uint k,
                    float inv,
                    uint tid,
                    uint nthreads)
{
    for (uint j = tid; j < m; j += nthreads) {
        float col[LLVQ_ROT_KMAX];
#pragma clang loop unroll(full)
        for (uint t = 0u; t < LLVQ_ROT_KMAX; ++t) {
            uint src = (t < k ? t : 0u) * m + j;
            col[t] = t < k ? s[src] * inv : 0.0f;
        }
        for (uint g = 0u; g < k; ++g) {
            float acc = 0.0f;
#pragma clang loop unroll(full)
            for (uint t = 0u; t < LLVQ_ROT_KMAX; ++t)
                acc = fma(small[g * LLVQ_ROT_KMAX + t], col[t], acc);
            xout[g * m + j] = acc;
        }
    }
}


kernel void rot_apply_tg_metal(const device ushort* xin      [[buffer(0)]],
                               const device uint*   signbits [[buffer(1)]],
                               const device float*  small    [[buffer(2)]],
                               device float*        xout     [[buffer(3)]],
                               constant uint&       n        [[buffer(5)]],
                               constant uint&       m        [[buffer(6)]],
                               constant uint&       k        [[buffer(7)]],
                               constant float&      inv      [[buffer(8)]],
                               constant uint&       x_off    [[buffer(9)]],
                               threadgroup float*   s        [[threadgroup(0)]],
                               uint tid      [[thread_position_in_threadgroup]],
                               uint nthreads [[threads_per_threadgroup]])
{
    for (uint i = tid; i < n; i += nthreads) {
        float v = rot_h2f(xin[x_off + i]);
        s[i] = ((signbits[i >> 5] >> (i & 31u)) & 1u) ? -v : v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint len = 1u; len < m; len <<= 1) {
        uint npairs = n >> 1;
        for (uint pp = tid; pp < npairs; pp += nthreads) {
            uint j = (pp / len) * (len << 1) + (pp % len);
            float a = s[j];
            float b = s[j + len];
            s[j] = a + b;
            s[j + len] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (k == 1u) {
        for (uint i = tid; i < n; i += nthreads) xout[i] = s[i] * inv;
    } else {
        rot_mix_tg(s, small, xout, m, k, inv, tid, nthreads);
    }
}
