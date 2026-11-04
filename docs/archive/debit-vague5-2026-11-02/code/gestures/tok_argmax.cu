// DRAFT, never compiled or run. Wave-5 deep dive "gestures", 2026-11-02, revised after review.
// Not part of the repository; a sketch for the operator to review.
//
// tok_argmax_h: the greedy token of one row of f16 logits, written into a
// device ring, with the tie order of candle 0.9.2's fast_argmax.
//
// candle's rule, read from the code (candle-kernels-0.9.2/src/reduce.cu:478-523,
// candle-core-0.9.2/src/cuda_backend/mod.rs:321-365):
//   * one block, block_dim = min(1024, n).next_power_of_two() = 1024 for n = 151,936;
//   * shr starts at -inf and shr_index at 0xFFFFFFFF; thread t scans
//     i = t, t+1024, ... with `not_set || src[i] > shr[t]` (strict), so inside a
//     thread the EARLIEST index of the max wins;
//   * the tree `if (tid < s && shr[tid+s] > shr[tid])`, s = 512 .. 1, keeps the
//     LEFT operand on ties. Bit 9 of t is decided first and bit 0 last, so among
//     tied threads the one with the smallest bitrev10(t) wins (512 beats 256,
//     2 beats 1);
//   * the f16 instantiation (reduce.cu:673) and the f32 one (reduce.cu:688) are
//     the same template, and f16 -> f32 widening is exact and order-preserving,
//     so argmax over the f16 logits equals argmax over their f32 cast.
//
// The key encodes that order, so ANY reduction order returns the same index:
//   key = ord16(v) << 32 | (1023 - bitrev10(i & 1023)) << 22 | (0x3FFFFF - (i >> 10))
// Valid for 1024 <= n < 1024 * 2^22 (block exactly 1024 wide; j fits 22 bits).
//
// NaN: in candle a NaN that is a thread's first element sticks, a NaN on the
// left of a tree merge absorbs and on the right loses. No commutative key can
// reproduce that, so a NaN is keyed 0, sets the sticky `flag`, and the ring
// gets token 0 (an in-bounds id, so the next gather cannot fault). The host
// reads `flag` with the tokens and refuses the generation (design choice for
// the operator: today candle would emit some token from broken logits).
//
// Geometry: 256 threads, grid G = ceil(n / 8 / 256) = 75 CTAs for n = 151,936
// (one 16-byte vector a thread, one wave on 142 SMs). Shared: 384 B. Registers:
// well under 64. Global state: slot (u64), ticket (u32), flag (u32), zeroed at
// load; the last CTA re-zeroes slot and ticket for the next token; the host
// zeroes flag when it arms a generation.
// Host preconditions (asserted in Rust): n % 8 == 0, x_off % 8 == 0 (16-byte
// alignment of the vector loads), n >= 1024, n < 2^32 / 2.
//
// Composition: stand-alone translation unit, its own NVRTC module; no
// dependency on the 10-file served unit. It follows the LLVQ_KERNEL_DIR
// contract like emb_q8.cu (fused_cuda.rs:63-75), but the card mutants below
// do NOT go through that variable: it is all or nothing over every kernel
// family (fused.rs:916-924), the runtime image carries no source (wave-3
// digest E7: the served unit is 10 files, 101,541 bytes), so eight complete
// directories would have to be shipped to the bucket. The A/B mode mutates
// the embedded text in memory instead (see MUTANTS at the end).

typedef unsigned int u32;
typedef unsigned long long u64;

__device__ __forceinline__ u32 ord16(u32 b)
{
    b &= 0xFFFFu;
    if (b == 0x8000u) b = 0u;                       // -0 ties +0, as `>` does
    return (b & 0x8000u) ? ((~b) & 0xFFFFu) : (b | 0x8000u);
}

__device__ __forceinline__ bool is_nan16(u32 b)
{
    return (b & 0x7C00u) == 0x7C00u && (b & 0x03FFu) != 0u;
}

__device__ __forceinline__ u32 bitrev10(u32 t)
{
    return __brev(t) >> 22;                         // reverse the low 10 bits
}

__device__ __forceinline__ u64 tie_key(u32 bits, u32 i)
{
    return ((u64)ord16(bits) << 32)
         | ((u64)(1023u - bitrev10(i & 1023u)) << 22)
         | (u64)(0x3FFFFFu - (i >> 10));
}

__device__ __forceinline__ u32 key_index(u64 k)
{
    u32 t = bitrev10(1023u - (u32)((k >> 22) & 0x3FFu));   // bitrev10 is an involution
    u32 j = 0x3FFFFFu - (u32)(k & 0x3FFFFFu);
    return (j << 10) | t;
}

__device__ __forceinline__ u64 umax64(u64 a, u64 b) { return a > b ? a : b; }

extern "C" __global__ void tok_argmax_h(const unsigned short* __restrict__ x,
                                        u32 x_off, u32 n,
                                        u64* slot, u32* ticket, u32* flag,
                                        u32* __restrict__ ring, u32 pos)
{
    __shared__ u64 wmax[32];
    __shared__ u32 wnan[32];
    u64 best = 0ull;
    u32 nan = 0u;

    const uint4* xv = reinterpret_cast<const uint4*>(x + x_off);
    const u32 nvec = n >> 3;
    for (u32 v = blockIdx.x * blockDim.x + threadIdx.x; v < nvec; v += gridDim.x * blockDim.x) {
        uint4 w = xv[v];
        u32 words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (u32 q = 0; q < 4u; ++q) {
#pragma unroll
            for (u32 h = 0; h < 2u; ++h) {
                u32 b = (words[q] >> (16u * h)) & 0xFFFFu;   // little-endian: low half first
                u32 i = (v << 3) + 2u * q + h;
                if (is_nan16(b)) { nan = 1u; continue; }
                best = umax64(best, tie_key(b, i));
            }
        }
    }

    // Warp, then block: max is associative and commutative, and keys are
    // unique per index, so the order of these merges cannot change the result.
    for (u32 o = 16u; o > 0u; o >>= 1)
        best = umax64(best, __shfl_xor_sync(0xffffffffu, best, o));
    nan = __any_sync(0xffffffffu, nan != 0u) ? 1u : 0u;
    const u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    if (lane == 0u) { wmax[warp] = best; wnan[warp] = nan; }
    __syncthreads();
    if (warp != 0u) return;                          // after the only barrier: safe
    const u32 nw = blockDim.x >> 5;
    best = lane < nw ? wmax[lane] : 0ull;
    nan = lane < nw ? wnan[lane] : 0u;
    for (u32 o = 16u; o > 0u; o >>= 1)
        best = umax64(best, __shfl_xor_sync(0xffffffffu, best, o));
    nan = __any_sync(0xffffffffu, nan != 0u) ? 1u : 0u;
    if (lane != 0u) return;

    atomicMax(slot, best);
    if (nan) atomicOr(flag, 1u);
    __threadfence();                                 // our max is visible before our ticket
    const u32 done = atomicAdd(ticket, 1u);
    if (done == gridDim.x - 1u) {                    // the last CTA sees every max
        const u64 k = atomicExch(slot, 0ull);        // read and reset for the next token
        *ticket = 0u;                                // ordered by the kernel boundary
        const bool any_nan = atomicAdd(flag, 0u) != 0u;   // sticky; the host reads it
        ring[pos] = any_nan ? 0u : key_index(k);
    }
}

// The spin kernel of the host clock (measurement only, never on a served path):
// one thread holds the stream for `ns` nanoseconds of %globaltimer, so the
// host can enqueue the launches behind it while the device is busy.
extern "C" __global__ void spin_ns(u64 ns)
{
    u64 t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do {
        __nanosleep(1000);
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < ns);
}

// ---------------------------------------------------------------------------
// THE CARD CHECK, run at load of the A/B mode (and of any path that serves the
// kernel later), against candle's own fast_argmax_f16 on the same device
// tensors: 41 calls IN ONE SEQUENCE through one slot/ticket/flag state, writing
// ring[0..41), one readback at the end. Order matters, and is part of the check:
//    1  +inf at n-1 (the largest key any case holds, so a slot that is not
//       reset, m6, poisons every later answer that lacks a +inf)
//    2  +inf at 10 and n-1         (candle: 10; bitrev 320 against 1018)
//    3  7.5 at 256 and 512         (candle: 512)
//    4  7.5 at 1 and 2             (candle: 2)
//    5  3.0 at 5 and 5 + 7*1024    (candle: 5, same thread)
//    6  -0 at 1300, +0 at 300      (candle: 1300; thread 276, bitrev 162, beats
//                                   thread 300, bitrev 210. The -0 sits at the
//                                   winning position, which is what kills m1;
//                                   the first draft had them the other way round
//                                   and did not, computed in calc/gestures-check.py)
//    7  all -inf                   (candle: 0)
//    8  tie of tiny subnormals
//    9..40  the first 32 sparse cases of argmax_key_ref::sparse_cases: the max at
//       2 to 6 random positions over a background of four levels with signed
//       zeros, so the cross-thread order decides (the first draft's dense levels
//       left the max in thread 0 with probability 1 - 2.3e-9, computed)
//    41 a quiet NaN at 1234: flag raised, ring[40] = 0 (last, because the flag
//       is sticky)
// Any index that differs stops the load and names the case. Cost: 41 vectors x
// 304 KB uploaded once, about 12.5 MB, a few ms of device time.
//
// ---------------------------------------------------------------------------
// MUTANTS (rule 10), applied by the A/B mode to the embedded text in memory.
// For each mutant: every `from` must occur EXACTLY once in the pristine text
// (the count is printed: a replace that matches nothing reads as a survivor),
// the mutated text compiles as its own NVRTC module, runs the 41-call sequence,
// and must fail at least one comparison or the flag check. Eight of eight must
// die, and the pristine text must pass, or the run is void.
//   m1  "if (b == 0x8000u) b = 0u;"            -> ""
//       killed by case 6 (and on the host by signed_zeros_tie)
//   m2  "return __brev(t) >> 22;"              -> "return t & 1023u;"
//       thread order by index instead of bit reversal; killed by cases 3 and 4
//   m3  "(u64)(0x3FFFFFu - (i >> 10));"        -> "(u64)(i >> 10);"
//       "u32 j = 0x3FFFFFu - (u32)(k & 0x3FFFFFu);" -> "u32 j = (u32)(k & 0x3FFFFFu);"
//       latest index inside a thread wins; killed by case 5
//   m4  "return __brev(t) >> 22;"              -> "return __brev(t) >> 23;"
//       a 9-bit reversal: the decode loses bit 9 of the thread, so case 3
//       returns 0 instead of 512. On the host it is killed by
//       key_index_inverts_tie_key (indices 512 and 1023). It is NOT killed by
//       random tie sets, as the first draft wrongly said.
//   m5  "*ticket = 0u;"                        -> ""
//       card only: from call 2 on, no CTA sees done == G - 1, ring[1..] keep 0
//   m6  "const u64 k = atomicExch(slot, 0ull);" -> "const u64 k = *slot;"
//       card only: the slot keeps case 2's +inf key, case 3 returns 10
//   m7  "atomicMax(slot, best);"               -> "*slot = best;"
//       card only: the last store wins; killed by the sparse cases
//   m8  "if (nan) atomicOr(flag, 1u);"         -> ""
//       killed by case 41 (and on the host by the NaN test)
// m5, m6 and m7 exist only on the card: the host reference mutates the Rust
// key, not this kernel, and cannot see them.
