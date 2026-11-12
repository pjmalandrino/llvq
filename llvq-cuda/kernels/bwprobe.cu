// bwprobe.cu: the card's read rate by a bare probe (decode-speed research phase, R2 item 4,
// docs/mesures/recherche-debit-r1-cadrage-2026-11-12.txt). No kernel of the repository had measured it: 864 GB/s
// is a datasheet figure, and the highest rate any of our kernels read is q4read's 671.
//
// The probe reads a buffer much larger than L2 and sums it. Three things keep it honest:
//   - the buffer is not constant: tv_bw_fill writes word i as bw_hash(i, seed), a 32-bit finaliser of i, so no
//     compression, no zero page and no constant folding can stand in for the reads;
//   - the sum is checked: every CTA writes its u64 partial sum, the host adds them and compares with its own sum of
//     the same hash (rowcand.rs::bw_expected_sum), and a configuration whose sum misses is refused, not printed;
//   - the result is used: the partials are written to global memory and read back after every timed launch.
// Variants (template parameters): VEC 1 (one u32 a load) or 4 (one 128-bit load), UNR independent loads in flight a
// thread a step (1 or 4). `pf_steps` > 0 adds a software prefetch, `prefetch.global.L2`, of the line the thread
// will read `pf_steps` grid strides later.

__device__ __forceinline__ u32 bw_hash(unsigned long long i, u32 seed)
{
    u32 h = (u32)i * 0x9e3779b9u + (u32)(i >> 32) * 0x85ebca6bu + seed;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

extern "C" __global__ void tv_bw_fill(u32* __restrict__ buf, unsigned long long n, u32 seed)
{
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x)
        buf[i] = bw_hash(i, seed);
}

__device__ __forceinline__ void bw_prefetch(const void* p)
{
#ifdef LLVQ_HOST_BUILD
    (void)p;
#else
    asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
#endif
}

template <u32 VEC>
__device__ __forceinline__ unsigned long long bw_load(const u32* p)
{
    if (VEC == 4u) {
        const uint4 v = *reinterpret_cast<const uint4*>(p);
        return (unsigned long long)v.x + v.y + v.z + v.w;
    }
    return *p;
}

// Reads n u32 (n a multiple of VEC), each element once, grid-stride by elements of VEC words; UNR loads a step.
template <u32 VEC, u32 UNR>
__device__ __forceinline__ void bw_read_body(const u32* __restrict__ buf, unsigned long long n,
                                             unsigned long long* __restrict__ part, u32 pf_steps)
{
    const unsigned long long ne = n / VEC;
    const unsigned long long t = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long S = (unsigned long long)gridDim.x * blockDim.x;
    unsigned long long sum = 0ull;
    unsigned long long i = t;
    for (; i + (UNR - 1u) * S < ne; i += UNR * S) {
        if (pf_steps != 0u) {
            const unsigned long long a = i + (unsigned long long)pf_steps * UNR * S;
            if (a < ne) bw_prefetch(buf + a * VEC);
        }
        unsigned long long v[UNR];
#pragma unroll
        for (u32 k = 0; k < UNR; ++k) v[k] = bw_load<VEC>(buf + (i + k * S) * VEC);
#pragma unroll
        for (u32 k = 0; k < UNR; ++k) sum += v[k];
    }
    for (; i < ne; i += S) sum += bw_load<VEC>(buf + i * VEC);
    // warp, then CTA: one u64 a CTA, no atomic
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    unsigned long long* red = reinterpret_cast<unsigned long long*>(PERS_SMEM_BASE() + PERS_SMEM_HEADER);
    const u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    if (lane == 0u) red[warp] = sum;
    __syncthreads();
    if (threadIdx.x == 0u) {
        unsigned long long s = 0ull;
        for (u32 w = 0; w < nw; ++w) s += red[w];
        part[blockIdx.x] = s;
    }
}

#define BW_KERNEL(sym, VEC, UNR)                                                                               \
    extern "C" __global__ void sym(const u32* __restrict__ buf, unsigned long long n,                         \
                                   unsigned long long* __restrict__ part, u32 pf_steps)                       \
    {                                                                                                          \
        bw_read_body<VEC, UNR>(buf, n, part, pf_steps);                                                        \
    }

BW_KERNEL(tv_bw_read_v1u1, 1u, 1u)
BW_KERNEL(tv_bw_read_v4u1, 4u, 1u)
BW_KERNEL(tv_bw_read_v4u4, 4u, 4u)
