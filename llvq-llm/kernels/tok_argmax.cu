// tok_argmax_h: the greedy token of one row of f16 logits, written into a
// device ring, with the tie order of candle 0.9.2's fast_argmax.
//
// candle's rule (candle-kernels-0.9.2/src/reduce.cu:478-523, dispatch
// candle-core-0.9.2/src/cuda_backend/mod.rs:321-365): one block of 1,024
// threads from 1,024 logits on; thread t scans i = t, t+1024, ... with a strict
// `>` from -inf, so inside a thread the EARLIEST index of the max wins; the
// tree keeps the LEFT operand on ties from s = 512 down to 1, so among tied
// threads the smallest bitrev10(t) wins.
//
// The key encodes that order, so ANY reduction order returns candle's index:
//   key = ord16(v) << 32 | (1023 - bitrev10(i & 1023)) << 22 | (0x3FFFFF - (i >> 10))
// Valid for 1024 <= n < 1024 * 2^22. The host reference and its tests are
// llvq-llm/src/argmax_key.rs; the card check runs both against candle's own
// fast_argmax_f16 at load.
//
// NaN: candle's answer then depends on the tree position, which no commutative
// key can follow. A NaN is skipped, raises the sticky `flag`, and the ring gets
// token 0 (in bounds, so the next gather cannot fault); the host refuses the
// generation.
//
// Geometry: 256 threads, grid ceil(n / 8 / 256) = 75 blocks at n = 151,936,
// one 16-byte vector a thread. State: slot (u64), ticket (u32), flag (u32),
// zeroed at load; the last block re-zeroes slot and ticket for the next call;
// the host zeroes flag when it arms a generation.
// Host preconditions (argmax_key::key_preconditions): n % 8 == 0,
// x_off % 8 == 0, 1024 <= n < 2^32.

typedef unsigned int u32;
typedef unsigned long long u64;

__device__ __forceinline__ u32 ord16(u32 b)
{
    b &= 0xFFFFu;
    if (b == 0x8000u) b = 0u;
    return (b & 0x8000u) ? ((~b) & 0xFFFFu) : (b | 0x8000u);
}

__device__ __forceinline__ bool is_nan16(u32 b)
{
    return (b & 0x7C00u) == 0x7C00u && (b & 0x03FFu) != 0u;
}

__device__ __forceinline__ u32 bitrev10(u32 t)
{
    return __brev(t) >> 22;
}

__device__ __forceinline__ u64 tie_key(u32 bits, u32 i)
{
    return ((u64)ord16(bits) << 32)
         | ((u64)(1023u - bitrev10(i & 1023u)) << 22)
         | (u64)(0x3FFFFFu - (i >> 10));
}

__device__ __forceinline__ u32 key_index(u64 k)
{
    u32 t = bitrev10(1023u - (u32)((k >> 22) & 0x3FFu));
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
            for (u32 hf = 0; hf < 2u; ++hf) {
                u32 b = (words[q] >> (16u * hf)) & 0xFFFFu;    // little-endian: low half first
                u32 i = (v << 3) + 2u * q + hf;
                if (is_nan16(b)) { nan = 1u; continue; }
                best = umax64(best, tie_key(b, i));
            }
        }
    }

    // Warp, then block: max is associative and commutative and the keys are
    // unique, so the order of these merges cannot change the result.
    for (u32 o = 16u; o > 0u; o >>= 1)
        best = umax64(best, __shfl_xor_sync(0xffffffffu, best, o));
    nan = __any_sync(0xffffffffu, nan != 0u) ? 1u : 0u;
    const u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    if (lane == 0u) { wmax[warp] = best; wnan[warp] = nan; }
    __syncthreads();
    if (warp != 0u) return;                          // after the only barrier
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
    if (done == gridDim.x - 1u) {                    // the last block sees every max
        const u64 k = atomicExch(slot, 0ull);
        *ticket = 0u;                                // ordered by the kernel boundary
        const bool any_nan = atomicAdd(flag, 0u) != 0u;
        ring[pos] = any_nan ? 0u : key_index(k);
    }
}
