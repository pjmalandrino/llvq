// DRAFT, never compiled or run. Step 3e of prereg-prefill-1.md, added in revision 2: the FP64
// instruction rate of the L40S, which settles U3 (the cost of the exact C-dense dequant) one job
// earlier than prereg-prefill-2, for seconds of card time. Written read-only against HEAD 06e1602.
//
// What it measures: DMUL and DADD instructions per second, the two operations the C-dense chain
// uses (deq_tetra48_f64.cu goes through __dmul_rn and __dadd_rn, never an FMA). Each thread runs
// FP64_CHAINS independent chains so that latency is hidden; the result is kept alive by a store
// that the compiler cannot prove dead. Instructions = gridDim.x * blockDim.x * iters * chains * 2.
//
// Host side (fusedrun measurement mode, sub-step 3e): grid 142 * 8 blocks of 256 threads,
// iters 4,096; 1 discarded burst, then 5 bursts of 10 launches, each burst between two fences
// (cudarc events off on the single stream, as LLVQ_GQA_PPL); rate = instructions / burst time.
// Printed: the median and range of the 5 bursts, in T instructions/s, beside the spec reading
// 142 SMs x 2 FP64 units x 2.52 GHz = 0.716 (computed; NVIDIA's L40S page lists no FP64 figure).
//
// It does NOT measure the dequant kernel's own efficiency (memory, barriers, the mix's loop): it
// bounds the arithmetic only. prereg-prefill-2 step 4 times one full dequant pass.

#ifndef FP64_CHAINS
#define FP64_CHAINS 4
#endif

extern "C" __global__ void fp64_probe(double* __restrict__ sink, unsigned iters, double a, double b) {
    double x[FP64_CHAINS];
#pragma unroll
    for (int c = 0; c < FP64_CHAINS; ++c) x[c] = 1.0 + 1e-3 * (double)(threadIdx.x + 37 * c);
    for (unsigned i = 0; i < iters; ++i) {
#pragma unroll
        for (int c = 0; c < FP64_CHAINS; ++c) x[c] = __dadd_rn(__dmul_rn(x[c], a), b); // 2 instructions, never fused
    }
    double s = 0.0;
#pragma unroll
    for (int c = 0; c < FP64_CHAINS; ++c) s = __dadd_rn(s, x[c]);
    // a = 0.999999 and b = 1e-6 keep x near 1: no overflow, no subnormal. The store happens for
    // one impossible value only, which the compiler cannot rule out, so the loop stays alive.
    if (s == 1234.5678) sink[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
